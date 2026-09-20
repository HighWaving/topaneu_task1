"""E09: E08's location-token head plus a dense voxel-level lesion head.

Why
---
Every experiment so far trained the classifier on 305 case-level binary labels spread over
52 classes. `data_topaneu26/location_masks/` has the actual voxel-level segmentation and
none of it was used. RSNA's 1st place carries exactly this kind of auxiliary task and
weights it at 1.0 against 0.1 for classification, with an ablation showing that equalising
the weights costs 0.902 -> 0.884 (`configs/experiment/251013-...yaml:75-77`).

Binary, not 52-class
--------------------
A lesion is ~82-135 voxels on the 128x256x256 input grid, so ~10-17 voxels on the
64x128x128 feature grid. Splitting that across 52 channels leaves nothing to learn from.
RSNA's auxiliary target is likewise a single binary channel (a sphere of radius 5). The
"which location" question is already carried by the 52 location tokens and their fixed
vessel attachment; what the dense head adds is "where in this volume is the lesion at all",
which is exactly the signal 305 weak labels cannot provide.

Head shape and initialisation follow RSNA's `_init_sphere_head`
(`src/models/components/anet_roi_net.py:184`): Conv3d(C,32,k3) -> InstanceNorm3d -> SiLU ->
Conv3d(32,1,k1) with the last layer zero-weight and bias -4.0, i.e. a prior of ~0.018.

Loss
----
balanced BCE + Tversky(alpha=0.3, beta=0.7), mirroring RSNA's
`BalancedBCEWithLogitsLoss + FocalTverskyPlusPlusLoss`, but with RSNA's balancing bug fixed:
their `balanced_bce.py:9` reduces to a scalar before splitting positives from negatives, so
the class balancing is a no-op and the term is just 2xBCE. Here the per-element loss is kept
(`reduction="none"`) so the positive and negative means are genuinely separate -- which
matters far more for us than for them, since our positive rate is ~1e-5.

The classification objective is NOT down-weighted. E08's `bce + 1.0*listwise` is a proven
configuration; this experiment adds a term rather than re-balancing a working one, so the
comparison against E08 stays a single change.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topaneu_loctoken import TopAneuLocationTokenClassifier


class TopAneuDenseSupervised(TopAneuLocationTokenClassifier):
    def __init__(self, *args, dense_lambda: float = 1.0, feature_channels: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense_lambda = float(dense_lambda)
        channels = int(feature_channels if feature_channels is not None else args[1])
        self.dense_head = nn.Sequential(
            nn.Conv3d(channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.dense_head[-1].weight)
        nn.init.constant_(self.dense_head[-1].bias, -4.0)
        self._dense_logits: torch.Tensor | None = None

    def forward(self, image: torch.Tensor, vessel_mask: torch.Tensor) -> torch.Tensor:
        try:
            out = self.backbone(image, vessel_seg=vessel_mask,
                                vessel_union=(vessel_mask > 0).to(vessel_mask.dtype))
        except TypeError:
            out = self.backbone(image)
        feat = out.get("feat", out.get("dec_feat")) if isinstance(out, dict) else out
        if feat is None or feat.dim() != 5:
            raise ValueError("backbone must return a 5D feature map or dict containing feat/dec_feat")

        masks = self._one_hot_vessels(vessel_mask, tuple(feat.shape[-3:]))
        dilated = F.max_pool3d(masks, kernel_size=3, stride=1, padding=1)
        pooled = self._memory_efficient_region_pool(feat, dilated)

        vessel_tokens = self.region_projection(pooled)
        ids = torch.arange(self.num_vessel_classes, device=image.device)
        vessel_tokens = vessel_tokens + self.vessel_embedding(ids).unsqueeze(0)
        vessel_tokens = vessel_tokens + self.spatial_projection(self._centroids(masks))

        location_tokens = torch.einsum("lk,bkd->bld", self.attachment.to(vessel_tokens.dtype), vessel_tokens)
        location_tokens = location_tokens + self.loc_token_embed.unsqueeze(0)

        global_feat = F.adaptive_avg_pool3d(feat, 1).flatten(1)
        cls = self.cls_token.expand(image.shape[0], -1, -1) + self.global_projection(global_feat).unsqueeze(1)
        encoded = self.context(torch.cat((cls, location_tokens), dim=1))

        # Kept on the module so `loss()` can use it without changing the forward signature,
        # which keeps every existing evaluation script working unmodified.
        self._dense_logits = self.dense_head(feat.float())
        return self.loc_head(encoded[:, 1:]).squeeze(-1)

    @staticmethod
    def _balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        positive, negative = target > 0.5, target <= 0.5
        loss = logits.sum() * 0.0
        if positive.any():
            loss = loss + per_element[positive].mean()
        if negative.any():
            loss = loss + per_element[negative].mean()
        return loss

    @staticmethod
    def _tversky(logits: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        tp = (probability * target).sum()
        fp = (probability * (1 - target)).sum()
        fn = ((1 - probability) * target).sum()
        return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)

    def dense_loss(self, lesion_mask: torch.Tensor) -> torch.Tensor:
        if self._dense_logits is None:
            raise RuntimeError("forward() must run before dense_loss()")
        logits = self._dense_logits
        target = (lesion_mask > 0).to(torch.float32)
        if target.dim() == 4:
            target = target.unsqueeze(1)
        target = F.interpolate(target, size=logits.shape[-3:], mode="nearest")
        return self._balanced_bce(logits, target) + self._tversky(logits, target)

    def loss(self, logits, targets, pos_weight=None, lesion_mask: torch.Tensor | None = None):
        classification = super().loss(logits, targets, pos_weight=pos_weight)
        parts = dict(self.last_loss_parts)
        if lesion_mask is None:
            self.last_loss_parts = {**parts, "dense": 0.0}
            return classification
        dense = self.dense_loss(lesion_mask)
        total = classification + self.dense_lambda * dense
        self.last_loss_parts = {**parts, "dense": float(dense.detach()), "total": float(total.detach())}
        return total

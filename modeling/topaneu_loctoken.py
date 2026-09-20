"""E08: 52 location tokens -- the correct port of RSNA's Stage-2 head.

What RSNA does (`src/models/aneurysm_vessel_seg_roi_module.py:1518-1609`):

    tokens     = loc_transformer(loc_proj(pooled) + loc_token_embed)   # (B, 13, 96)
    logits_all = loc_head(tokens).squeeze(-1)                          # (B, 13)

one token per LOCATION, one shared LayerNorm+Linear(96,1) read out per token.  That works
because RSNA's 13 locations *are* its 13 vessel mask channels, so token i carries location
i's identity by construction.

What E01-E07 did instead (`model_dilation.py:156-165`): TopAneu has 36 vessels and 52
locations, the correspondence was dropped, the 36 vessel tokens were MEAN-POOLED into one
vector and a flat Linear(2*96, 52) read all 52 logits off it.  Averaging deletes which
vessel is which, so "aneurysm on vessel k -> location j" has no path through the model.
Measured consequence: a constant, case-independent ranking built from the model's own
global mean scores beats H3 and E01 on within-case GT rank and is beaten by E02 by only
0.92 ranks.

The BCE + within-case listwise objective from E03/E07 is inherited unchanged, so E08
differs from E07 in exactly one thing: how the 52 logits are produced.

This module restores RSNA's structure using the curated 52x36 attachment
(`location_attachment.py`).  Each location token is the attachment-weighted mean of the
vessel tokens it anatomically belongs to, plus a learned per-location embedding; the shared
transformer and per-token head then match RSNA exactly.

Note on the `position` group (8 of 52 classes): those locations share a vessel segment with
a sibling and are separated only by position along it, so their attachment rows are
identical to their sibling's.  They are distinguishable only through the centroid term
already folded into the vessel tokens, plus their own `loc_token_embed`.  That is a known
limit of the vessel map, not of this head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topaneu_listwise import TopAneuVesselAwareListwise


class TopAneuLocationTokenClassifier(TopAneuVesselAwareListwise):
    def __init__(self, *args, attachment: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if attachment.shape != (self.num_outputs, self.num_vessel_classes):
            raise ValueError(
                f"attachment must be ({self.num_outputs},{self.num_vessel_classes}), got {attachment.shape}")
        weights = attachment.astype(np.float32)
        if not (weights.sum(axis=1) > 0).all():
            raise ValueError("every location must attach to at least one vessel")
        weights = weights / weights.sum(axis=1, keepdims=True)
        # Fixed anatomy, not learned: registered as a buffer so it travels with the
        # checkpoint and is restored on strict load.
        self.register_buffer("attachment", torch.from_numpy(weights))

        embed_dim = self.cls_token.shape[-1]
        self.loc_token_embed = nn.Parameter(torch.zeros(self.num_outputs, embed_dim))
        nn.init.trunc_normal_(self.loc_token_embed, std=0.02)
        self.loc_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        # `output_head` from the parent is now unused. It is left in place so that
        # load_compatible_backbone_weights and any parent-shaped checkpoint still load.

    def forward(self, image: torch.Tensor, vessel_mask: torch.Tensor) -> torch.Tensor:
        try:
            out = self.backbone(
                image,
                vessel_seg=vessel_mask,
                vessel_union=(vessel_mask > 0).to(vessel_mask.dtype),
            )
        except TypeError:
            out = self.backbone(image)
        feat = out.get("feat", out.get("dec_feat")) if isinstance(out, dict) else out
        if feat is None or feat.dim() != 5:
            raise ValueError("backbone must return a 5D feature map or dict containing feat/dec_feat")

        masks = self._one_hot_vessels(vessel_mask, tuple(feat.shape[-3:]))
        dilated_masks = F.max_pool3d(masks, kernel_size=3, stride=1, padding=1)
        pooled = self._memory_efficient_region_pool(feat, dilated_masks)

        vessel_tokens = self.region_projection(pooled)
        ids = torch.arange(self.num_vessel_classes, device=image.device)
        vessel_tokens = vessel_tokens + self.vessel_embedding(ids).unsqueeze(0)
        vessel_tokens = vessel_tokens + self.spatial_projection(self._centroids(masks))

        # (52,36) @ (B,36,D) -> (B,52,D): each location token is the mean of the vessel
        # tokens it attaches to. Vessel identity and centroid are already inside those
        # tokens, so a junction token sees both of its segments.
        location_tokens = torch.einsum("lk,bkd->bld", self.attachment.to(vessel_tokens.dtype), vessel_tokens)
        location_tokens = location_tokens + self.loc_token_embed.unsqueeze(0)

        global_feat = F.adaptive_avg_pool3d(feat, 1).flatten(1)
        cls = self.cls_token.expand(image.shape[0], -1, -1) + self.global_projection(global_feat).unsqueeze(1)
        encoded = self.context(torch.cat((cls, location_tokens), dim=1))
        return self.loc_head(encoded[:, 1:]).squeeze(-1)

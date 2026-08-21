"""Minimal TopAneu adaptation of the RSNA vessel-aware classifier.

The 36 vessel masks create anatomical feature tokens. They are deliberately
decoupled from the 52 independent aneurysm-location outputs: no vessel token
has a fixed output label. A transformer models vessel/global context and a
single multi-label head learns the 36-token -> 52-label relationship.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .region_mask_pooling import RegionMaskedPooling3D


class TopAneuVesselAwareClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: int,
        *,
        num_vessel_classes: int = 36,
        num_outputs: int = 52,
        embed_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_vessel_classes == num_outputs:
            raise ValueError("vessel regions and Task 1 outputs must be independently specified")
        self.backbone = backbone
        self.num_vessel_classes = int(num_vessel_classes)
        self.num_outputs = int(num_outputs)

        self.region_pool = RegionMaskedPooling3D(
            mask_pool_modes="mean",
            global_pool_modes="mean",
            mask_feat_channels=int(feature_channels),
            global_feat_channels=int(feature_channels),
            branch_norm=True,
        )
        pooled_dim = self.region_pool.output_dim()
        self.region_projection = nn.Linear(pooled_dim, embed_dim)
        self.global_projection = nn.Linear(int(feature_channels), embed_dim)
        self.vessel_embedding = nn.Embedding(self.num_vessel_classes, embed_dim)
        # z/y/x normalized centroid plus a presence bit.
        self.spatial_projection = nn.Linear(4, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, self.num_outputs),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _one_hot_vessels(
        self,
        vessel_mask: torch.Tensor,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        if vessel_mask.dim() != 5:
            raise ValueError("vessel_mask must be (B,1,D,H,W) labels or (B,36,D,H,W) masks")
        if vessel_mask.shape[1] == self.num_vessel_classes:
            masks = vessel_mask.float()
            if masks.shape[-3:] != target_size:
                masks = F.interpolate(masks, size=target_size, mode="nearest")
            return masks
        if vessel_mask.shape[1] != 1:
            raise ValueError(f"unexpected vessel channels: {vessel_mask.shape[1]}")
        labels = vessel_mask.float()
        if labels.shape[-3:] != target_size:
            labels = F.interpolate(labels, size=target_size, mode="nearest")
        labels = labels[:, 0].round().long()
        if labels.min() < 0 or labels.max() > self.num_vessel_classes:
            raise ValueError("vessel label outside background + 1..36")
        return F.one_hot(labels, num_classes=self.num_vessel_classes + 1)[..., 1:].permute(0, 4, 1, 2, 3).float()

    @staticmethod
    def _centroids(masks: torch.Tensor) -> torch.Tensor:
        b, k, d, h, w = masks.shape
        z = torch.linspace(-1, 1, d, device=masks.device, dtype=masks.dtype).view(1, 1, d, 1, 1)
        y = torch.linspace(-1, 1, h, device=masks.device, dtype=masks.dtype).view(1, 1, 1, h, 1)
        x = torch.linspace(-1, 1, w, device=masks.device, dtype=masks.dtype).view(1, 1, 1, 1, w)
        mass = masks.flatten(2).sum(-1)
        denom = mass.clamp_min(1e-6)
        cz = (masks * z).flatten(2).sum(-1) / denom
        cy = (masks * y).flatten(2).sum(-1) / denom
        cx = (masks * x).flatten(2).sum(-1) / denom
        present = (mass > 0).to(masks.dtype)
        coords = torch.stack((cz, cy, cx, present), dim=-1)
        return coords * present.unsqueeze(-1)

    @torch.autocast(device_type="cuda", enabled=False)
    def _memory_efficient_region_pool(self, feat: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Match RSNA masked/global mean pooling without a BxKxCxDxHxW tensor.

        The original implementation broadcasts the feature map over every
        region.  That is tolerable for 13 RSNA regions, but TA36 would require
        an additional ~9 GiB at the baseline resolution.  Batched matrix
        multiplication computes the same weighted means directly.
        """
        feat = feat.float()
        weights = masks.float().clamp_min(0)
        flat_feat = feat.flatten(2)
        flat_weights = weights.flatten(2)
        denominator = flat_weights.sum(-1, keepdim=True).clamp_min(self.region_pool.eps)
        masked_mean = torch.bmm(flat_weights, flat_feat.transpose(1, 2)) / denominator
        masked_mean = self.region_pool.mask_proj(self.region_pool.mask_norm(masked_mean))

        global_mean = flat_feat.mean(-1)
        global_mean = global_mean.unsqueeze(1).expand(-1, masks.shape[1], -1)
        global_mean = self.region_pool.global_proj(self.region_pool.global_norm(global_mean))
        return torch.cat((masked_mean, global_mean), dim=-1)

    def forward(self, image: torch.Tensor, vessel_mask: torch.Tensor) -> torch.Tensor:
        print(f"[*] Backbone input: shape={image.shape}, dtype={image.dtype}, device={image.device}", flush=True)
        try:
            out: Any = self.backbone(
                image,
                vessel_seg=vessel_mask,
                vessel_union=(vessel_mask > 0).to(vessel_mask.dtype),
            )
        except TypeError:
            out = self.backbone(image)
        if isinstance(out, dict):
            feat = out.get("feat", out.get("dec_feat"))
        else:
            feat = out
        if feat is None or feat.dim() != 5:
            raise ValueError("backbone must return a 5D feature map or dict containing feat/dec_feat")

        # Tokenize at feature-map resolution.  Building a 36-channel one-hot
        # mask at full 128x256x256 input resolution wastes more than 1 GiB.
        masks = self._one_hot_vessels(vessel_mask, tuple(feat.shape[-3:]))
        pooled = self._memory_efficient_region_pool(feat, masks)
        region_tokens = self.region_projection(pooled)
        ids = torch.arange(self.num_vessel_classes, device=image.device)
        region_tokens = region_tokens + self.vessel_embedding(ids).unsqueeze(0)
        region_tokens = region_tokens + self.spatial_projection(self._centroids(masks))

        global_feat = F.adaptive_avg_pool3d(feat, 1).flatten(1)
        cls = self.cls_token.expand(image.shape[0], -1, -1) + self.global_projection(global_feat).unsqueeze(1)
        encoded = self.context(torch.cat((cls, region_tokens), dim=1))
        case_repr = torch.cat((encoded[:, 0], encoded[:, 1:].mean(dim=1)), dim=-1)
        return self.output_head(case_repr)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        if logits.shape != targets.shape or logits.shape[-1] != self.num_outputs:
            raise ValueError("Task 1 targets must be a (B,52) multi-hot tensor")
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)

    def load_compatible_backbone_weights(self, checkpoint: str | Path) -> dict[str, int]:
        """Reuse shape-compatible RSNA Stage 2 representation tensors.

        The 13-token location embedding and 13-output heads are intentionally
        discarded.  In addition to the image backbone, the shape-compatible
        token projection and Transformer context layers are retained.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        target = self.backbone.state_dict()
        prefixes = ("model.net.", "model.module.net.", "net.", "backbone.", "")
        matched: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            for prefix in prefixes:
                candidate = key[len(prefix):] if prefix and key.startswith(prefix) else (key if not prefix else None)
                if candidate in target and target[candidate].shape == value.shape:
                    matched[candidate] = value
                    break
        self.backbone.load_state_dict(matched, strict=False)

        full_target = self.state_dict()
        compatible_context: dict[str, torch.Tensor] = {}
        rename_prefixes = {
            "loc_proj.": "region_projection.",
            "loc_transformer.": "context.",
        }
        for source_key, value in state.items():
            for source_prefix, target_prefix in rename_prefixes.items():
                # Lightning keys may be unprefixed, or nested below the model.
                marker = source_key.find(source_prefix)
                if marker < 0:
                    continue
                candidate = target_prefix + source_key[marker + len(source_prefix) :]
                if candidate in full_target and full_target[candidate].shape == value.shape:
                    compatible_context[candidate] = value
                break
        self.load_state_dict(compatible_context, strict=False)
        return {
            "matched": len(matched) + len(compatible_context),
            "matched_backbone": len(matched),
            "matched_context": len(compatible_context),
            "target_backbone_tensors": len(target),
            "source_tensors": len(state),
        }

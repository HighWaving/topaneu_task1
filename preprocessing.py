"""Stage 2 preprocessing used by the epoch-10 TopAneu baseline."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


INPUT_SIZE = (128, 256, 256)
VESSEL_MARGIN_MM = 15.0


def _center_pad_or_crop(x: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
    _, _, depth, height, width = x.shape
    target_depth, target_height, target_width = target
    pad_d = max(0, target_depth - depth)
    pad_h = max(0, target_height - height)
    pad_w = max(0, target_width - width)
    if pad_d or pad_h or pad_w:
        x = F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_d // 2,
                pad_d - pad_d // 2,
            ),
        )
        _, _, depth, height, width = x.shape
    start_d = max(0, (depth - target_depth) // 2)
    start_h = max(0, (height - target_height) // 2)
    start_w = max(0, (width - target_width) // 2)
    return x[
        :,
        :,
        start_d : start_d + target_depth,
        start_h : start_h + target_height,
        start_w : start_w + target_width,
    ]


def _resize_z_xy(
    image: torch.Tensor,
    vessel: torch.Tensor,
    target: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    _, depth, height, width = image.shape
    target_depth, target_height, target_width = target
    scale_z = target_depth / depth
    scale_xy = min(target_height / height, target_width / width)
    new_size = (
        max(1, round(depth * scale_z)),
        max(1, round(height * scale_xy)),
        max(1, round(width * scale_xy)),
    )
    image = F.interpolate(image[None].float(), size=new_size, mode="trilinear", align_corners=False)
    vessel = F.interpolate(vessel[None].float(), size=new_size, mode="nearest")
    image = _center_pad_or_crop(image, target)[0]
    vessel = _center_pad_or_crop(vessel, target)[0].round().to(torch.uint8)
    return image, vessel


def _vessel_bbox(
    mask_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    margin_mm: float,
) -> tuple[slice, ...]:
    points = np.argwhere(mask_zyx > 0)
    if points.size == 0:
        return tuple(slice(0, size) for size in mask_zyx.shape)
    lower = points.min(0)
    upper = points.max(0) + 1
    margin = np.asarray(
        [math.ceil(margin_mm / max(float(spacing), 1e-6)) for spacing in spacing_zyx]
    )
    lower = np.maximum(0, lower - margin)
    upper = np.minimum(np.asarray(mask_zyx.shape), upper + margin)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def preprocess_case(
    image_path: str | Path,
    vessel_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_nii = nib.load(str(image_path))
    vessel_nii = nib.load(str(vessel_path))
    if image_nii.shape != vessel_nii.shape:
        raise ValueError("TA36 output shape does not match the input image")
    if not np.allclose(image_nii.affine, vessel_nii.affine, rtol=1e-5, atol=1e-4):
        raise ValueError("TA36 output affine does not match the input image")
    if "".join(nib.aff2axcodes(image_nii.affine)) != "LPS":
        raise ValueError("Stage 2 expects the image and TA36 mask in LPS orientation")

    image_xyz = np.asarray(image_nii.dataobj, dtype=np.float32)
    vessel_xyz = np.asarray(vessel_nii.dataobj)
    if not np.isfinite(image_xyz).all():
        raise ValueError("input contains non-finite voxels")
    if not np.allclose(vessel_xyz, np.rint(vessel_xyz)):
        raise ValueError("TA36 output contains non-integer labels")
    vessel_xyz = np.rint(vessel_xyz).astype(np.uint8)
    if int(vessel_xyz.max(initial=0)) > 36:
        raise ValueError("TA36 output label is outside background + 1..36")

    image_zyx = image_xyz.transpose(2, 1, 0)
    vessel_zyx = vessel_xyz.transpose(2, 1, 0)
    spacing_zyx = tuple(reversed(image_nii.header.get_zooms()[:3]))
    crop = _vessel_bbox(vessel_zyx, spacing_zyx, VESSEL_MARGIN_MM)
    image_zyx = image_zyx[crop]
    vessel_zyx = vessel_zyx[crop]
    mean = float(image_zyx.mean())
    std = float(image_zyx.std())
    image_zyx = (image_zyx - mean) / max(std, 1e-6)
    image = torch.from_numpy(np.ascontiguousarray(image_zyx[None])).float()
    vessel = torch.from_numpy(np.ascontiguousarray(vessel_zyx[None]))
    return _resize_z_xy(image, vessel, INPUT_SIZE)

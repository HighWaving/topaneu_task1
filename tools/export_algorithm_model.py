#!/usr/bin/env python
"""Export an epoch checkpoint into the flat Grand Challenge model layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TA36_SUBDIRS = [
    "resEncM_ceW_bg0.75_max1.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3",
    "plain_conv_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3",
    "primusV3S_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_clsBalSamp_degree0.75_warm50_ep1000_DS",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch = int(payload["epoch"])
    exported = {
        "model": payload["model"],
        "epoch": epoch,
        "metrics": payload.get("metrics", {}),
        "architecture": payload.get("architecture", {}),
        "rsna_weight_reuse": payload.get("rsna_weight_reuse", {}),
    }
    torch.save(exported, args.output_dir / "task1_checkpoint.pt")
    config = {
        "submission_version": "v0.1.0",
        "checkpoint_epoch": epoch,
        "num_vessel_classes": 36,
        "num_outputs": 52,
        "input_size_zyx": [128, 256, 256],
        "vessel_margin_mm": 15.0,
        "normalization": "per-cropped-volume z-score",
        "resize": "keep_ratio=z-xy then center pad/crop",
        "nnunet_configuration": "3d_fullres",
        "num_truncate_stages": 1,
        "embed_dim": 96,
        "transformer_heads": 4,
        "transformer_layers": 2,
        "output_activation": "sigmoid",
        "output_threshold": 0.5,
        "modality_handling": "socket-specific CTA/MRA handlers; shared Stage 2 model",
        "vessel_source": "Official TA36 direct three-model inference",
        "ta36_model_subdirs": TA36_SUBDIRS,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    mapping = {
        "description": "Model logit index to TopAneu Task 1 location ID",
        "logit_index_to_location_id": {str(index): index + 1 for index in range(52)},
    }
    (args.output_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2))
    print(json.dumps({"epoch": epoch, "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

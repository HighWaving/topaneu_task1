#!/usr/bin/env python
"""Validate the unarchived Grand Challenge Algorithm Model layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    required = [
        "task1_checkpoint.pt",
        "config.json",
        "label_mapping.json",
        "stage2_nnunet/plans.json",
        "stage2_nnunet/dataset.json",
    ]
    missing = [name for name in required if not (args.model_dir / name).is_file()]
    config = json.loads((args.model_dir / "config.json").read_text())
    mapping = json.loads((args.model_dir / "label_mapping.json").read_text())
    for subdir in config["ta36_model_subdirs"]:
        model_dir = args.model_dir / "ta36_models" / subdir
        for relative in ("plans.json", "dataset.json", "fold_4/checkpoint_final.pth"):
            if not (model_dir / relative).is_file():
                missing.append(str(Path("ta36_models") / subdir / relative))
    if missing:
        raise FileNotFoundError(f"missing Algorithm Model resources: {missing}")
    if list(mapping["logit_index_to_location_id"].values()) != list(range(1, 53)):
        raise ValueError("52-label mapping is not identity index+1")
    checkpoint = torch.load(args.model_dir / "task1_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != config["checkpoint_epoch"]:
        raise ValueError("checkpoint/config epoch mismatch")
    if not isinstance(checkpoint.get("model"), dict) or not checkpoint["model"]:
        raise ValueError("checkpoint has no model state")
    print(
        json.dumps(
            {
                "gate": "ALGORITHM_MODEL_LAYOUT_PASS",
                "epoch": checkpoint["epoch"],
                "state_tensors": len(checkpoint["model"]),
                "labels": len(mapping["logit_index_to_location_id"]),
                "ta36_models": len(config["ta36_model_subdirs"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

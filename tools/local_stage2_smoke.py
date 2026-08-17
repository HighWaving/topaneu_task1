#!/usr/bin/env python
"""Local Stage 2 smoke using organizer-generated TA36 masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import inference
from preprocessing import preprocess_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--vessel", type=Path, action="append", required=True)
    parser.add_argument("--modality", choices=("ct", "mr"), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (len(args.image) == len(args.vessel) == len(args.modality)):
        raise ValueError("provide the same number of --image, --vessel and --modality values")
    model = inference._load_model()
    threshold = float(inference._load_config()["output_threshold"])
    cases = []
    with torch.inference_mode():
        for image_path, vessel_path, modality in zip(args.image, args.vessel, args.modality):
            image, vessel = preprocess_case(image_path, vessel_path)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(image[None].cuda(), vessel[None].cuda())
            probabilities = torch.sigmoid(logits)[0].float().cpu()
            labels = [index + 1 for index, value in enumerate(probabilities.tolist()) if value >= threshold]
            cases.append(
                {
                    "image": str(image_path),
                    "vessel": str(vessel_path),
                    "modality": modality,
                    "image_shape": list(image.shape),
                    "vessel_shape": list(vessel.shape),
                    "logit_shape": list(logits.shape),
                    "finite": bool(torch.isfinite(probabilities).all()),
                    "prediction": labels,
                }
            )
    report = {"gate": "CTA_MRA_STAGE2_SMOKE_PASS", "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

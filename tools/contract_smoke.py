#!/usr/bin/env python
"""Exercise the exact Grand Challenge Task 1 socket and JSON writer contract."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import SimpleITK as sitk

import main


def run_interface(socket_slug: str, image_subdir: str, handler_name: str) -> list[int]:
    with tempfile.TemporaryDirectory(prefix="topaneu-contract-", dir="/tmp") as temporary:
        root = Path(temporary)
        input_path = root / "input"
        output_path = root / "output"
        image_dir = input_path / "images" / image_subdir
        image_dir.mkdir(parents=True)
        output_path.mkdir()
        (input_path / "inputs.json").write_text(
            json.dumps([{"socket": {"slug": socket_slug}}])
        )
        image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
        sitk.WriteImage(image, str(image_dir / "sanity.mha"))
        main.INPUT_PATH = input_path
        main.OUTPUT_PATH = output_path
        setattr(main, handler_name, lambda _: [1, 52])
        if main.run() != 0:
            raise RuntimeError("Grand Challenge handler returned non-zero")
        result = json.loads((output_path / "detected-aneurysm-locations.json").read_text())
        main.validate_locations(result)
        return result


def main_smoke() -> int:
    report = {
        "gate": "GRAND_CHALLENGE_CONTRACT_PASS",
        "ct": run_interface("head-ct-angiography", "head-ct-angio", "infer_ct"),
        "mr": run_interface("head-mr-angiography", "head-mr-angio", "infer_mr"),
        "output": "detected-aneurysm-locations.json",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_smoke())

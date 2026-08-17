"""Grand Challenge Task 1 algorithm entrypoint, based on the official template."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import SimpleITK

from inference import infer_ct, infer_mr


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def run() -> int:
    interface_key = get_interface_key()
    handlers = {
        ("head-ct-angiography",): ("head-ct-angio", infer_ct),
        ("head-mr-angiography",): ("head-mr-angio", infer_mr),
    }
    if interface_key not in handlers:
        raise RuntimeError(f"unsupported Grand Challenge input sockets: {interface_key}")
    image_directory, infer = handlers[interface_key]
    image = load_image_file(INPUT_PATH / "images" / image_directory)
    prediction = infer(image)
    write_json_file(OUTPUT_PATH / "detected-aneurysm-locations.json", prediction)
    return 0


def get_interface_key() -> tuple[str, ...]:
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    return tuple(sorted(value["socket"]["slug"] for value in inputs))


def load_json_file(location: Path):
    return json.loads(location.read_text())


def load_image_file(location: Path) -> SimpleITK.Image:
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    if len(input_files) != 1:
        raise RuntimeError(f"expected exactly one image in {location}, found {len(input_files)}")
    return SimpleITK.ReadImage(input_files[0])


def validate_locations(content: list[int]) -> None:
    if not isinstance(content, list):
        raise TypeError("Task 1 output must be a JSON list")
    if any(type(value) is not int or not 1 <= value <= 52 for value in content):
        raise ValueError("Task 1 location IDs must be integers in [1, 52]")
    if len(content) != len(set(content)):
        raise ValueError("Task 1 location IDs must be unique")


def write_json_file(location: Path, content: list[int]) -> None:
    validate_locations(content)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(run())

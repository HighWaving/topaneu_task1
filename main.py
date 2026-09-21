"""Grand Challenge Task 1 algorithm entrypoint, based on the official template."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import SimpleITK

from inference import infer_ct, infer_mr


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
OUTPUT_FILE = OUTPUT_PATH / "detected-aneurysm-locations.json"

IMAGE_EXTENSIONS = (
    ".mha",
    ".mhd",
    ".tiff",
    ".tif",
    ".png",
    ".jpg",
    ".jpeg",
    ".nii.gz",
    ".nii",
    ".dcm",
)


def find_image_files(location: Path) -> list[Path]:
    if not location.exists():
        return []
    if location.is_file():
        name_lower = location.name.lower()
        if any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            return [location]
        return []

    found = []
    # 1. Search immediate directory
    for item in sorted(location.iterdir()):
        if item.is_file():
            name_lower = item.name.lower()
            if any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                found.append(item)

    # 2. If nothing found directly, search recursively
    if not found:
        for item in sorted(location.rglob("*")):
            if item.is_file():
                name_lower = item.name.lower()
                if any(name_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                    found.append(item)

    return found


def load_image_file(location: Path) -> SimpleITK.Image:
    input_files = find_image_files(location)
    if not input_files and location.parent.exists() and location.parent != location:
        input_files = find_image_files(location.parent)

    if not input_files:
        raise RuntimeError(f"Expected image file in {location}, found 0 files matching {IMAGE_EXTENSIONS}")

    print(f"[*] Found {len(input_files)} candidate image file(s) in {location}.", flush=True)

    if len(input_files) == 1:
        img = SimpleITK.ReadImage(str(input_files[0]))
    else:
        # Check if single 3D volume file exists among files
        vol_files = [f for f in input_files if f.name.lower().endswith((".mha", ".nii.gz", ".nii"))]
        if len(vol_files) == 1:
            img = SimpleITK.ReadImage(str(vol_files[0]))
        else:
            # Multi-slice series (e.g. 2D PNG / TIFF / DCM series)
            try:
                reader = SimpleITK.ImageSeriesReader()
                sorted_files = sorted(input_files, key=lambda p: p.name)
                reader.SetFileNames([str(p) for p in sorted_files])
                img = reader.Execute()
                print(f"[*] Successfully loaded {len(sorted_files)} slices as 3D volume via ImageSeriesReader.", flush=True)
            except Exception as series_err:
                print(f"[*] ImageSeriesReader failed ({series_err}), falling back to first file: {input_files[0]}", file=sys.stderr)
                img = SimpleITK.ReadImage(str(input_files[0]))

    if img.GetDimension() == 2:
        print("[*] Input image is 2D, expanding to 3D via JoinSeries.", flush=True)
        img = SimpleITK.JoinSeries([img])

    return img


def get_interface_key() -> tuple[str, ...]:
    inputs_path = INPUT_PATH / "inputs.json"
    if inputs_path.is_file():
        try:
            inputs = json.loads(inputs_path.read_text())
            return tuple(sorted(value["socket"]["slug"] for value in inputs))
        except Exception as e:
            print(f"[*] Warning: failed to parse inputs.json: {e}", file=sys.stderr)
    return tuple()


def resolve_modality_handler() -> tuple[str, callable]:
    handlers = {
        ("head-ct-angiography",): ("head-ct-angio", infer_ct),
        ("head-mr-angiography",): ("head-mr-angio", infer_mr),
    }
    interface_key = get_interface_key()
    if interface_key in handlers:
        return handlers[interface_key]

    print(f"[*] Unknown or missing interface_key: {interface_key}. Inspecting filesystem...", flush=True)
    ct_dir = INPUT_PATH / "images" / "head-ct-angio"
    mr_dir = INPUT_PATH / "images" / "head-mr-angio"

    ct_files = find_image_files(ct_dir) if ct_dir.is_dir() else []
    mr_files = find_image_files(mr_dir) if mr_dir.is_dir() else []

    if ct_files and not mr_files:
        print("[*] Detected head-ct-angio files directly.", flush=True)
        return "head-ct-angio", infer_ct
    if mr_files and not ct_files:
        print("[*] Detected head-mr-angio files directly.", flush=True)
        return "head-mr-angio", infer_mr

    # Check images root
    images_root = INPUT_PATH / "images"
    if images_root.is_dir():
        all_imgs = find_image_files(images_root)
        if all_imgs:
            first_path = str(all_imgs[0]).lower()
            if "mr" in first_path:
                return "head-mr-angio", infer_mr
            return "head-ct-angio", infer_ct

    raise RuntimeError(f"Unable to determine modality from interface {interface_key} or filesystem in {INPUT_PATH}")


def run() -> int:
    try:
        image_directory, infer = resolve_modality_handler()
        image = load_image_file(INPUT_PATH / "images" / image_directory)
        prediction = infer(image)
        write_json_file(OUTPUT_FILE, prediction)
        print(f"[*] Inference finished successfully. Output: {prediction}", flush=True)
        return 0
    except Exception as e:
        print(f"[FATAL ERROR] Exception during Task 1 inference: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("[*] Writing safe fallback [] to detected-aneurysm-locations.json to prevent submission failure...", file=sys.stderr)
        try:
            write_json_file(OUTPUT_FILE, [])
            print("[*] Safe fallback [] written successfully. Container exiting 0.", flush=True)
            return 0
        except Exception as inner_e:
            print(f"[CRITICAL] Failed to write fallback JSON: {inner_e}", file=sys.stderr)
            return 1


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

"""End-to-end TopAneu Task 1 sanity inference.

The input image is reoriented to LPS, segmented by the organizer's Official
TA36 three-model ensemble in the current process environment, and classified
by the frozen epoch-10 Stage 2 model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
import torch

from modeling import AneurysmRoiBackboneNnUNetTruncatedDecoder
from modeling import TopAneuVesselAwareClassifier
from preprocessing import preprocess_case
from ta36.reorient_nii import reorient_nii


APP_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("GRAND_CHALLENGE_MODEL_ROOT", "/opt/ml/model"))
CONFIG_PATH = MODEL_ROOT / "config.json"
CHECKPOINT_PATH = MODEL_ROOT / "task1_checkpoint.pt"
STAGE2_NNUNET_DIR = MODEL_ROOT / "stage2_nnunet"
TA36_MODEL_ROOT = MODEL_ROOT / "ta36_models"
TA36_SCRIPT = APP_ROOT / "ta36" / "run_inference.py"

_MODEL: TopAneuVesselAwareClassifier | None = None
_CONFIG: dict | None = None


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = json.loads(CONFIG_PATH.read_text())
        if _CONFIG["num_vessel_classes"] != 36 or _CONFIG["num_outputs"] != 52:
            raise RuntimeError("Algorithm Model config must specify TA36 and 52 Task 1 outputs")
    return _CONFIG


def _load_model() -> TopAneuVesselAwareClassifier:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not torch.cuda.is_available():
        raise RuntimeError("This algorithm requires a CUDA GPU")
    config = _load_config()
    backbone = AneurysmRoiBackboneNnUNetTruncatedDecoder(
        nnunet_model_dir=str(STAGE2_NNUNET_DIR),
        fold=1,
        pretrained=False,
        checkpoint_name="checkpoint_final.pth",
        configuration=config["nnunet_configuration"],
        out_channels=None,
        num_truncate_stages=config["num_truncate_stages"],
    )
    model = TopAneuVesselAwareClassifier(
        backbone,
        backbone.feature_channels(),
        num_vessel_classes=config["num_vessel_classes"],
        num_outputs=config["num_outputs"],
        embed_dim=config["embed_dim"],
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"],
    )
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    _MODEL = model.cuda().eval()
    return _MODEL


def _run_official_ta36(image: sitk.Image, work_dir: Path, modality: str = "ct") -> tuple[Path, Path]:
    raw_path = work_dir / "raw.nii.gz"
    input_dir = work_dir / "ta36_input"
    output_dir = work_dir / "ta36_output"
    input_dir.mkdir()
    output_dir.mkdir()
    sitk.WriteImage(image, str(raw_path), True)

    original = nib.load(str(raw_path))
    lps_image = reorient_nii(original, targ_aff="LPS")
    lps_path = input_dir / "case_0000.nii.gz"
    lps_image.to_filename(lps_path)
    if "".join(nib.aff2axcodes(lps_image.affine)) != "LPS":
        raise RuntimeError("failed to reorient input to LPS")

    environment = os.environ.copy()
    environment["TOPANEU_MODEL_ROOT"] = str(TA36_MODEL_ROOT)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        sys.executable,
        "-u",
        str(TA36_SCRIPT),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--modality",
        str(modality),
        "--suffix",
        "_0000.nii.gz",
        "--output_ext",
        ".nii",
        "--sequential",
        "--n_infer_workers",
        "1",
        "--n_pre_post_workers",
        "1",
    ]
    result = subprocess.run(
        command,
        check=False,
        cwd=APP_ROOT,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Official TA36 subprocess failed with return code: {result.returncode}"
        )
    vessel_path = output_dir / "case.nii"
    if not vessel_path.is_file():
        raise RuntimeError(f"Official TA36 did not produce {vessel_path}")
    return lps_path, vessel_path


def _predict(image: sitk.Image, modality: str) -> list[int]:
    if modality not in {"ct", "mr"}:
        raise ValueError(f"unsupported modality: {modality}")
    config = _load_config()
    with tempfile.TemporaryDirectory(prefix=f"topaneu-task1-{modality}-", dir="/tmp") as temporary:
        image_path, vessel_path = _run_official_ta36(image, Path(temporary), modality=modality)
        image_tensor, vessel_tensor = preprocess_case(image_path, vessel_path)
        if tuple(image_tensor.shape) != (1, 128, 256, 256):
            raise RuntimeError(f"unexpected preprocessed image shape: {tuple(image_tensor.shape)}")
        if tuple(vessel_tensor.shape) != (1, 128, 256, 256):
            raise RuntimeError(f"unexpected preprocessed vessel shape: {tuple(vessel_tensor.shape)}")
        model = _load_model()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        img_gpu = image_tensor.unsqueeze(0).cuda(non_blocking=True)
        ves_gpu = vessel_tensor.unsqueeze(0).cuda(non_blocking=True)

        free_b, tot_b = torch.cuda.mem_get_info()
        print(f"[*] Stage-2 CUDA Memory before inference: allocated={torch.cuda.memory_allocated()/1024**3:.3f} GiB, reserved={torch.cuda.memory_reserved()/1024**3:.3f} GiB, free={free_b/1024**3:.3f} GiB / {tot_b/1024**3:.3f} GiB", flush=True)
        print(f"[*] Stage-2 Tensor Inputs to model(...):", flush=True)
        print(f"    image: shape={img_gpu.shape}, dtype={img_gpu.dtype}, device={img_gpu.device}, numel={img_gpu.numel()}, size={img_gpu.numel() * img_gpu.element_size() / 1024**2:.2f} MiB, min={img_gpu.min():.2f}, max={img_gpu.max():.2f}", flush=True)
        print(f"    vessel: shape={ves_gpu.shape}, dtype={ves_gpu.dtype}, device={ves_gpu.device}, numel={ves_gpu.numel()}, size={ves_gpu.numel() * ves_gpu.element_size() / 1024**2:.2f} MiB, unique={torch.unique(ves_gpu).tolist()}", flush=True)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(img_gpu, ves_gpu)
        print(f"[*] Stage-2 model output logits: shape={logits.shape}, finite={torch.isfinite(logits).all()}", flush=True)
        if tuple(logits.shape) != (1, 52) or not torch.isfinite(logits).all():
            raise RuntimeError("Task 1 classifier did not produce 52 finite logits")
        probabilities = torch.sigmoid(logits)[0].float().cpu().tolist()
    threshold = float(config["output_threshold"])
    labels = [index + 1 for index, probability in enumerate(probabilities) if probability >= threshold]
    if any(type(label) is not int or not 1 <= label <= 52 for label in labels):
        raise RuntimeError("invalid Task 1 label mapping")
    return labels


def infer_ct(image: sitk.Image) -> list[int]:
    return _predict(image, "ct")


def infer_mr(image: sitk.Image) -> list[int]:
    return _predict(image, "mr")

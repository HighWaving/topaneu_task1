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


def _resolve_ta36_model_root() -> Path:
    subdirs = [
        "resEncM_ceW_bg0.75_max1.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3",
        "plain_conv_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3",
        "primusV3S_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_clsBalSamp_degree0.75_warm50_ep1000_DS",
    ]
    candidates = [
        TA36_MODEL_ROOT,
        MODEL_ROOT / "ta36",
        MODEL_ROOT / "Dataset572_TopAneu_Vessel_36fgCls_wLRSwap",
        MODEL_ROOT,
        APP_ROOT / "ta36" / "data" / "results" / "Dataset572_TopAneu_Vessel_36fgCls_wLRSwap",
    ]
    for cand in candidates:
        if cand.is_dir() and all((cand / d).is_dir() for d in subdirs):
            print(f"[TA36] Located model weights at: {cand}", flush=True)
            return cand

    print("[TA36] Candidate path search details:", flush=True)
    for cand in candidates:
        print(f"  candidate: {cand} (exists={cand.exists()}, is_dir={cand.is_dir()})", flush=True)
    if MODEL_ROOT.exists():
        print(f"[TA36] Files in MODEL_ROOT ({MODEL_ROOT}):", flush=True)
        for item in sorted(MODEL_ROOT.rglob("*"))[:50]:
            print(f"  {item.relative_to(MODEL_ROOT)} (is_dir={item.is_dir()})", flush=True)
    return TA36_MODEL_ROOT


def _run_official_ta36(image: sitk.Image, work_dir: Path) -> tuple[Path, Path]:
    raw_path = work_dir / "raw.nii.gz"
    input_dir = work_dir / "ta36_input"
    output_dir = work_dir / "ta36_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(raw_path), True)

    original = nib.load(str(raw_path))
    lps_image = reorient_nii(original, targ_aff="LPS")
    lps_path = input_dir / "case_0000.nii.gz"
    lps_image.to_filename(lps_path)
    if "".join(nib.aff2axcodes(lps_image.affine)) != "LPS":
        raise RuntimeError("failed to reorient input to LPS")

    resolved_ta36_root = _resolve_ta36_model_root()
    environment = os.environ.copy()
    environment["TOPANEU_MODEL_ROOT"] = str(resolved_ta36_root)
    environment["PYTHONUNBUFFERED"] = "1"
    pythonpath = str(APP_ROOT)
    if "PYTHONPATH" in environment:
        pythonpath = f"{pythonpath}:{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = pythonpath

    command = [
        sys.executable,
        str(TA36_SCRIPT),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--suffix",
        "_0000.nii.gz",
        "--sequential",
        "--n_infer_workers",
        "1",
        "--n_pre_post_workers",
        "1",
    ]
    print(f"[TA36] Executing TA36 inference: {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=APP_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(f"[TA36 STDOUT]\n{result.stdout}", flush=True)
    if result.stderr:
        print(f"[TA36 STDERR]\n{result.stderr}", file=sys.stderr, flush=True)
    if result.returncode != 0:
        error_msg = (
            f"Official TA36 subprocess failed with return code {result.returncode}.\n"
            f"=== SUBPROCESS STDOUT ===\n{result.stdout}\n"
            f"=== SUBPROCESS STDERR ===\n{result.stderr}\n"
        )
        raise RuntimeError(error_msg)

    vessel_path = output_dir / "case.nii.gz"
    if not vessel_path.is_file():
        raise RuntimeError(f"Official TA36 did not produce {vessel_path}")
    return lps_path, vessel_path


def _predict(image: sitk.Image, modality: str) -> list[int]:
    if modality not in {"ct", "mr"}:
        raise ValueError(f"unsupported modality: {modality}")
    config = _load_config()
    with tempfile.TemporaryDirectory(prefix=f"topaneu-task1-{modality}-", dir="/tmp") as temporary:
        image_path, vessel_path = _run_official_ta36(image, Path(temporary))
        image_tensor, vessel_tensor = preprocess_case(image_path, vessel_path)
        if tuple(image_tensor.shape) != (1, 128, 256, 256):
            raise RuntimeError(f"unexpected preprocessed image shape: {tuple(image_tensor.shape)}")
        if tuple(vessel_tensor.shape) != (1, 128, 256, 256):
            raise RuntimeError(f"unexpected preprocessed vessel shape: {tuple(vessel_tensor.shape)}")
        model = _load_model()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                image_tensor.unsqueeze(0).cuda(non_blocking=True),
                vessel_tensor.unsqueeze(0).cuda(non_blocking=True),
            )
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

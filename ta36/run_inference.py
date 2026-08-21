#!/usr/bin/env python
"""TopAneu vessel segmentation inference entry point (Docker).

Runs the 3-model ensemble (resEncM + plain_conv + primusV3S, Dataset572,
36 fg classes, fold 4) on a single NIfTI file or on all NIfTI files found
recursively in a folder. Results are written to the output directory with
the same sub-folder structure as the input.

Examples (inside the container):

    # single file
    python run_inference.py -i /input/case_001.nii.gz -o /output

    # whole folder (nested NIfTI files are found automatically)
    python run_inference.py -i /input -o /output

    # with napari screenshot rendering enabled
    python run_inference.py -i /input -o /output --vis
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import torch

torch.backends.cudnn.benchmark = True

REPO_ROOT = Path(__file__).resolve().parent

# Model weights baked into the image. Override with TOPANEU_MODEL_ROOT if you
# mount different weights.
MODEL_ROOT = os.environ.get(
    'TOPANEU_MODEL_ROOT',
    str(REPO_ROOT / 'data/results/Dataset572_TopAneu_Vessel_36fgCls_wLRSwap'),
)

# nnUNet v2.8 models trained on OMEN, patched in-place by
# 20260611_patch_omen_v2.8_models_for_v2.5_infer.py (checkpoint trainer_name
# rewritten to a repo trainer; _primus added to Primus plans.json), so they
# need NO `overwrite_trainer_name` and NO PRIMUS_* env vars.
MR_MODEL_SUBDIRS = [
    'resEncM_ceW_bg0.75_max1.5_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep1000_DS3',
    'primusV3S_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_clsBalSamp_degree0.75_warm50_ep1000_DS',
]

CT_MODEL_SUBDIRS = [
    'primusV3S_ceW_bg0.5_max2.5_diff_cluster_noMirror_bs2_ps80_192_128_clsBalSamp_degree0.75_warm50_ep1000_DS',
]

VIS_SCRIPT = REPO_ROOT / 'nnunetv2/houjing_scripts/vis_label_screenshots_napari_multi_view.py'


def build_model_cfg(modality='ct', tile_step_size=None):
    is_ct = str(modality).lower() == 'ct'
    subdirs = CT_MODEL_SUBDIRS if is_ct else MR_MODEL_SUBDIRS
    if tile_step_size is None:
        tile_step_size = 0.95 if is_ct else 0.75
    return [
        dict(
            base_model_dir=MODEL_ROOT,
            subdir=subdir,
            model_fold=4,
            ckpt_name='checkpoint_final.pth',
            patch_size=None,
            tile_step_size=tile_step_size,
        )
        for subdir in subdirs
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description='TopAneu vessel segmentation inference (36-class ensemble).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input', required=True, type=Path,
                   help='Input NIfTI file, or a folder searched recursively for *<suffix> files.')
    p.add_argument('-o', '--output', required=True, type=Path,
                   help='Output directory. Sub-folder structure mirrors the input folder.')
    p.add_argument('--modality', default='ct', choices=['ct', 'mr'],
                   help='Input imaging modality (ct: PrimusV3S only, mr: resEncM + PrimusV3S).')
    p.add_argument('--tile_step_size', type=float, default=None,
                   help='Tile step size for sliding window inference (default: 0.90 for CT, 0.75 for MR).')
    p.add_argument('--suffix', default='.nii.gz',
                   help="Input filename suffix to match; it is stripped from the case name "
                        "(e.g. '_0000.nii.gz' strips the channel suffix from output names).")
    p.add_argument('--output_ext', default='.nii.gz',
                   help='Extension of the saved segmentations.')
    p.add_argument('--sequential', action='store_true',
                   help='Run preprocess/inference/postprocess sequentially in one process '
                        'instead of the parallel pipeline.')
    p.add_argument('--n_infer_workers', type=int, default=1,
                   help='Number of GPU inference workers.')
    p.add_argument('--n_pre_post_workers', type=int, default=1,
                   help='Number of preprocessing workers and of post-processing workers. '
                        'The default 1 is validated to fit in 64 GB container memory; '
                        'more workers are faster but hold more cases in RAM at once.')
    p.add_argument('--n_gpus', type=int, default=1,
                   help='Number of GPUs to spread inference workers over.')
    p.add_argument('--gpu_limit_GB', type=float, default=None,
                   help='Limit GPU memory usage to approximately this many gigabytes '
                        '(default: no limit). Applied inside each process that runs '
                        'inference, so it works with both --sequential and the '
                        'parallel pipeline.')
    p.add_argument('--fuse_logits', action='store_true',
                   help='Ensemble by averaging model logits on the preprocessed grid and '
                        'resampling once, instead of resampling each model\'s output '
                        '(~3x faster export stage). Changes fusion from mean-of-softmax '
                        'to softmax-of-mean-logits; segmentations may differ slightly '
                        'at structure boundaries. Requires all ensemble models to share '
                        'the same preprocessing target spacing (true for the built-in '
                        '3-model ensemble; enforced at runtime).')
    p.add_argument('--fp16', default=True, action=argparse.BooleanOptionalAction,
                   help='Keep the ensemble probability accumulator (and with --fuse_logits '
                        'the fused logits) in float16, roughly halving its RAM. Argmax '
                        'output can differ from float32 only at near-exact probability '
                        'ties, far below the run-to-run GPU nondeterminism level. '
                        'Disable with --no-fp16.')
    p.add_argument('--overwrite_existing', action='store_true',
                   help='Re-run cases whose output file already exists (default: skip them).')
    p.add_argument('--vis', action='store_true',
                   help='Render napari multi-view screenshots after inference.')
    p.add_argument('--vis_views', nargs='+', default=['sagittal', 'coronal', 'axial'],
                   help='Views to render when --vis is enabled.')
    p.add_argument('--vis_grid_cols', type=int, default=3,
                   help='Number of columns in the visualization screenshot grid.')
    return p.parse_args()


def find_cases(input_path: Path, suffix: str):
    if input_path.is_file():
        if not input_path.name.endswith(suffix):
            sys.exit(f"Input file {input_path} does not end with suffix '{suffix}'")
        return input_path.parent, [input_path.name[:-len(suffix)]]
    if input_path.is_dir():
        fnames = sorted(
            str(p.relative_to(input_path))[:-len(suffix)]
            for p in input_path.rglob('*')
            if p.is_file() and p.name.endswith(suffix)
        )
        return input_path, fnames
    sys.exit(f"Input path does not exist: {input_path}")


def run_vis(args, in_dir: Path, fnames: list[str]):
    vis_out_root = args.output / 'visualizations'
    rel_dirs = sorted({str(Path(f).parent) if Path(f).parent != Path('.') else '' for f in fnames})
    for rel in rel_dirs:
        labels_dir = args.output / rel if rel else args.output
        images_dir = in_dir / rel if rel else in_dir
        vis_out = vis_out_root / rel if rel else vis_out_root
        cmd = [
            'xvfb-run', '-a', sys.executable, str(VIS_SCRIPT),
            '--labels_dir', str(labels_dir),
            '--images_dir', str(images_dir),
            '--out_dir', str(vis_out),
            '--views', *args.vis_views,
            '--grid_cols', str(args.vis_grid_cols),
            '--skip_existing',
        ]
        print(f"\n=== Rendering screenshots: {labels_dir} -> {vis_out} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: screenshot rendering failed for {labels_dir} "
                  f"(exit code {result.returncode})", file=sys.stderr)
    print(f"\nScreenshots saved to: {vis_out_root}")


def main():
    args = parse_args()

    in_dir, fnames = find_cases(args.input, args.suffix)
    if not fnames:
        sys.exit(f"No files matching '*{args.suffix}' found under {args.input}")
    print(f"Found {len(fnames)} case(s) under {in_dir}")

    target_subdirs = CT_MODEL_SUBDIRS if args.modality == 'ct' else MR_MODEL_SUBDIRS
    missing = [d for d in target_subdirs if not (Path(MODEL_ROOT) / d).is_dir()]
    if missing:
        sys.exit(f"Model weights not found under {MODEL_ROOT}: {missing}")

    from nnunetv2.houjing_scripts.infer_ppl_parallel_npz import infer_folder

    infer_folder(
        in_dir=str(in_dir),
        out_dir=str(args.output),
        fnames=fnames,
        suffix=args.suffix,
        output_ext=args.output_ext,
        model_cfg=build_model_cfg(args.modality, args.tile_step_size),
        sequential=args.sequential,
        n_preprocess_workers=args.n_pre_post_workers,
        n_infer_workers=args.n_infer_workers,
        n_post_inference_workers=args.n_pre_post_workers,
        queue1_size=args.n_pre_post_workers,
        queue2_size=args.n_pre_post_workers,
        n_gpus=args.n_gpus,
        gpu_limit_GB=args.gpu_limit_GB,
        fuse_logits=args.fuse_logits,
        fp16=args.fp16,
        use_mirroring=False,
        post_process=False,
        skip_existing=not args.overwrite_existing,
    )

    print(f"\nSegmentations saved to: {args.output}")

    if args.vis:
        run_vis(args, in_dir, fnames)


if __name__ == '__main__':
    main()

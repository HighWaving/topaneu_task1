# TopAneu 2026 Task 1 sanity submission

This repository is the code-only Grand Challenge Algorithm container for the
first TopAneu Task 1 sanity submission. It follows the organizer's Task 1
template sockets and output contract.

Pipeline:

1. Read one CTA (`head-ct-angiography`) or MRA (`head-mr-angiography`) image.
2. Reorient the temporary NIfTI representation to LPS.
3. Run the organizer's Official TA36 three-model ensemble directly in Python.
4. Crop/normalize/resize the image and 36-class vessel mask.
5. Run the frozen epoch-10 vessel-aware Stage 2 classifier.
6. Convert 52 sigmoid outputs to unique location IDs in `1..52` and write
   `/output/detected-aneurysm-locations.json`.

The container does not start Docker. Official TA36 inference source is
vendored under `ta36/` and `vendor/nnunetv2/`; its three weights and the Task 1
checkpoint are supplied separately as a Grand Challenge Algorithm Model at
`/opt/ml/model/`.

## Algorithm Model layout

The uploaded tarball must unpack directly to:

```text
/opt/ml/model/
├── task1_checkpoint.pt
├── config.json
├── label_mapping.json
├── stage2_nnunet/
│   ├── plans.json
│   └── dataset.json
└── ta36_models/
    └── <three Official TA36 model directories>/
```

## Version

- Branch: `main`
- Sanity tag: `v0.1.0`
- Output policy: sigmoid threshold `0.5` (score is not the objective of this version)

## Provenance

- Grand Challenge I/O is based on the downloaded official TopAneu Task 1 template.
- TA36 inference source and model configuration are taken from the organizer's
  `20260814_topaneu_vesselseg_uzh.tar` release on Zenodo record `21959166`.
- The TA36 model weights are not committed to Git and are delivered only through
  the Algorithm Model tarball.

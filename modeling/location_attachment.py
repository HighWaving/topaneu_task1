"""Fixed (52, 36) location -> vessel attachment used to build 52 location tokens.

Why this exists
---------------
RSNA's Stage-2 head works because its 13 locations ARE its 13 vessel mask channels, so
token i carries location i's identity by construction
(`aneurysm_vessel_seg_roi_module.py:1518-1609`).  TopAneu has 36 vessel segments and 52
locations, so the port up to E07 gave up on the correspondence and mean-pooled the 36
tokens before a flat 52-way head -- deleting the "aneurysm is on vessel k -> location j"
path entirely, which is what the prevalence-collapse diagnostic measures.

This module restores the correspondence with the curated map in
`scripts/phase2b_location_class_to_vessel_map.py`, whose oracle study
(`artifacts/phase2b_geometric_attachment/`) shows the attachment is real: given a
ground-truth lesion mask, nearest-vessel attachment recovers the location class 76.6 % of
the time (87.8 % on junction classes).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

# Inside the container the mapping JSONs ship in the Algorithm Model.
PROJECT_ROOT = Path(os.environ.get("TOPANEU_PROJECT_ROOT", "/opt/ml/model"))
# The trainers read the mapping from topaneu_release; the data_topaneu26 copy renamed
# ids 51/52 from "5.4 Distal-M2M3" to "5.3 Distal-M2M3", colliding with ids 49/50.
# Accept both spellings so the matrix builds against either copy.
NAME_ALIASES = {"R-5.3 Distal-M2M3": "R-5.4 Distal-M2M3",
                "L-5.3 Distal-M2M3": "L-5.4 Distal-M2M3"}


def _load_curated_map() -> dict:
    path = Path(os.environ.get(
        "TOPANEU_LOCATION_MAP",
        str(Path(__file__).resolve().parent / "phase2b_location_class_to_vessel_map.py")))
    spec = importlib.util.spec_from_file_location("phase2b_map", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LOCATION_TO_VESSEL


def build_attachment(
    location_mapping: Path,
    vessel_mapping: Path,
    num_locations: int = 52,
    num_vessels: int = 36,
) -> tuple[np.ndarray, list[str]]:
    """Return a (52, 36) float32 indicator matrix and the per-location group labels."""
    curated = _load_curated_map()
    loc_labels = json.loads(Path(location_mapping).read_text())["labels"]
    ves_labels = json.loads(Path(vessel_mapping).read_text())["labels"]

    loc_by_id = {int(v): k for k, v in loc_labels.items() if int(v) != 0}
    ves_by_name = {k: int(v) for k, v in ves_labels.items() if int(v) != 0}
    if sorted(loc_by_id) != list(range(1, num_locations + 1)):
        raise ValueError("location mapping must cover ids 1..52")
    if sorted(ves_by_name.values()) != list(range(1, num_vessels + 1)):
        raise ValueError("vessel mapping must cover ids 1..36")

    matrix = np.zeros((num_locations, num_vessels), dtype=np.float32)
    groups: list[str] = []
    for location_id in range(1, num_locations + 1):
        name = loc_by_id[location_id]
        key = name if name in curated else NAME_ALIASES.get(name)
        if key is None or key not in curated:
            raise ValueError(f"location {location_id} ({name!r}) has no curated vessel attachment")
        group, vessel_names = curated[key]
        groups.append(group)
        for vessel_name in vessel_names:
            if vessel_name not in ves_by_name:
                raise ValueError(f"{name!r} attaches to unknown vessel {vessel_name!r}")
            matrix[location_id - 1, ves_by_name[vessel_name] - 1] = 1.0

    if not (matrix.sum(axis=1) > 0).all():
        raise ValueError("every location must attach to at least one vessel")
    return matrix, groups

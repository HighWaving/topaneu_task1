"""Curated location(52) -> vessel-segment(36) mapping, by class semantics not voxel overlap.

Peer-review-driven: phase2's voxel-overlap purity (0.66 mean) measures the
wrong geometric quantity -- an aneurysm bulges OFF a vessel, so most of its
own voxels sit outside the vessel_masks segmentation by construction. The
correct question is attachment (nearest vessel segment), not overlap.

Every one of the 52 class NAMES already encodes its anatomical relationship
to vessel_mapping.json's 36 segments:

* SINGLE  (22 classes): named after exactly one vessel segment
  (e.g. "R-1.1 VA trunk" -> R-VA). A geometric method should identify this
  one segment.
* JUNCTION (20 classes): named as a junction between two segments
  (e.g. "R-1.3 VA-PICA junction" -> {R-VA, R-PICA}). A geometric method
  should find both segments represented near the lesion.
* POSITION (10 classes): share a vessel_masks segment with a sibling class,
  distinguished only by proximal/distal position along it (e.g. "R-4.2 A1"
  and "R-4.3 A2" are both on segment R-A1A2). Vessel-segment identity ALONE
  cannot disambiguate these -- they need an additional along-vessel-position
  feature, not attempted here. Flagged, not silently scored as solvable.

This is a judgement call on ambiguous cases (BA tip, C6-nonOA, C7-terminus,
etc.) -- documented inline. Not claimed to be the only reasonable reading;
built to be checked and adjusted, not accepted blind.
"""

from __future__ import annotations

# name -> (group, frozenset of vessel_mapping.json segment names it should attach to)
# group in {"single", "junction", "position"}
LOCATION_TO_VESSEL = {
    # --- Vertebrobasilar / posterior circulation ---
    "R-1.1 VA trunk": ("single", frozenset({"R-VA"})),
    "L-1.1 VA trunk": ("single", frozenset({"L-VA"})),
    "R-1.2 PICA trunk": ("single", frozenset({"R-PICA"})),
    "L-1.2 PICA trunk": ("single", frozenset({"L-PICA"})),
    "R-1.3 VA-PICA junction": ("junction", frozenset({"R-VA", "R-PICA"})),
    "L-1.3 VA-PICA junction": ("junction", frozenset({"L-VA", "L-PICA"})),
    "1.4 BA trunk": ("single", frozenset({"BA"})),
    # VA-BA junction has no side in its own name -- either vertebral artery
    # joining the basilar is a hit; scored as attaching to BA + (R-VA or L-VA).
    "1.5 VA-BA junction": ("junction", frozenset({"BA", "R-VA", "L-VA"})),
    "R-1.6 AICA trunk": ("single", frozenset({"R-AICA"})),
    "L-1.6 AICA trunk": ("single", frozenset({"L-AICA"})),
    "R-1.7 BA-AICA junction": ("junction", frozenset({"BA", "R-AICA"})),
    "L-1.7 BA-AICA junction": ("junction", frozenset({"BA", "L-AICA"})),
    "R-1.8 SCA trunk": ("single", frozenset({"R-SCA"})),
    "L-1.8 SCA trunk": ("single", frozenset({"L-SCA"})),
    "R-1.9 BA-SCA junction": ("junction", frozenset({"BA", "R-SCA"})),
    "L-1.9 BA-SCA junction": ("junction", frozenset({"BA", "L-SCA"})),
    # BA tip is the basilar trifurcation into both P1P2s -- a 3-way junction.
    "1.10 BA tip": ("junction", frozenset({"BA", "R-P1P2", "L-P1P2"})),
    "R-2.1 P1P2": ("single", frozenset({"R-P1P2"})),
    "L-2.1 P1P2": ("single", frozenset({"L-P1P2"})),
    "R-2.2 P3P4": ("single", frozenset({"R-P3P4"})),
    "L-2.2 P3P4": ("single", frozenset({"L-P3P4"})),

    # --- ICA ---
    "R-3.1 ICA infraclinoid C1-C5": ("single", frozenset({"R-ICA-C1-C5"})),
    "L-3.1 ICA infraclinoid C1-C5": ("single", frozenset({"L-ICA-C1-C5"})),
    "R-3.2 ICA C6-OA-junction": ("junction", frozenset({"R-ICA-C6-C7", "R-OA"})),
    "L-3.2 ICA C6-OA-junction": ("junction", frozenset({"L-ICA-C6-C7", "L-OA"})),
    # "nonOA" = plain C6 segment away from the OA junction -- single segment.
    "R-3.3 ICA C6-nonOA": ("single", frozenset({"R-ICA-C6-C7"})),
    "L-3.3 ICA C6-nonOA": ("single", frozenset({"L-ICA-C6-C7"})),
    "R-3.4 ICA C7-Pcom-junction": ("junction", frozenset({"R-ICA-C6-C7", "R-Pcom"})),
    "L-3.4 ICA C7-Pcom-junction": ("junction", frozenset({"L-ICA-C6-C7", "L-Pcom"})),
    "R-3.5 ICA C7-AChA-junction": ("junction", frozenset({"R-ICA-C6-C7", "R-AChA"})),
    "L-3.5 ICA C7-AChA-junction": ("junction", frozenset({"L-ICA-C6-C7", "L-AChA"})),
    # "nonBranch" = plain C7 segment, no named branch nearby -- single segment,
    # but shares R-ICA-C6-C7 with 3.3/3.2/3.4/3.5 -> effectively a position
    # class relative to those (can't distinguish "plain C7" from "plain C6"
    # by segment ID alone either). Treated as position, not single.
    "R-3.6 ICA C7-nonBranch": ("position", frozenset({"R-ICA-C6-C7"})),
    "L-3.6 ICA C7-nonBranch": ("position", frozenset({"L-ICA-C6-C7"})),
    # Terminus = the ICA bifurcation into M1/A1 -- a junction with both.
    "R-3.7 ICA C7-terminus": ("junction", frozenset({"R-ICA-C6-C7", "R-M1", "R-A1A2"})),
    "L-3.7 ICA C7-terminus": ("junction", frozenset({"L-ICA-C6-C7", "L-M1", "L-A1A2"})),

    # --- ACA ---
    "4.1 Acom complex": ("single", frozenset({"Acom"})),
    "R-4.2 A1": ("position", frozenset({"R-A1A2"})),
    "L-4.2 A1": ("position", frozenset({"L-A1A2"})),
    "R-4.3 A2": ("position", frozenset({"R-A1A2"})),
    "L-4.3 A2": ("position", frozenset({"L-A1A2"})),
    "R-4.4 A3": ("single", frozenset({"R-A3"})),
    "L-4.4 A3": ("single", frozenset({"L-A3"})),
    "R-4.5 Distal ACA branches": ("single", frozenset({"3rd-A2", "3rd-A3"})),
    "L-4.5 Distal ACA branches": ("single", frozenset({"3rd-A2", "3rd-A3"})),

    # --- MCA ---
    "R-5.1 M1 trunk": ("single", frozenset({"R-M1"})),
    "L-5.1 M1 trunk": ("single", frozenset({"L-M1"})),
    # "early bifurcation" is a position along M1, same segment as trunk.
    "R-5.2 M1 early bifurcation": ("position", frozenset({"R-M1"})),
    "L-5.2 M1 early bifurcation": ("position", frozenset({"L-M1"})),
    "R-5.3 M1-M2 junction": ("junction", frozenset({"R-M1", "R-M2"})),
    "L-5.3 M1-M2 junction": ("junction", frozenset({"L-M1", "L-M2"})),
    "R-5.4 Distal-M2M3": ("single", frozenset({"R-M2", "R-M3"})),
    "L-5.4 Distal-M2M3": ("single", frozenset({"L-M2", "L-M3"})),
}

assert len(LOCATION_TO_VESSEL) == 52, len(LOCATION_TO_VESSEL)
GROUP_COUNTS = {"single": 0, "junction": 0, "position": 0}
for _, (group, _) in LOCATION_TO_VESSEL.items():
    GROUP_COUNTS[group] += 1

if __name__ == "__main__":
    print(GROUP_COUNTS)

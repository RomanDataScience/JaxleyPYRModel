"""Validation and extraction of static compartment features."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .records import StaticFeatures


def extract_features(cell, required_groups: Iterable[str]) -> StaticFeatures:
    groups = {}
    for group in required_groups:
        if group not in cell.nodes.columns:
            raise ValueError(f"Morphology is missing required group: {group}")
        mask = cell.nodes[group].to_numpy(dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Morphology group is empty: {group}")
        groups[group] = mask

    covered = np.sum(np.stack(tuple(groups.values())), axis=0)
    if np.any(covered != 1):
        indices = np.flatnonzero(covered != 1)[:10].tolist()
        raise ValueError(
            "Required morphology groups must partition compartments exactly; "
            f"bad indices include {indices}."
        )
    if "dist_from_soma" not in cell.nodes:
        raise ValueError("Morphology does not provide dist_from_soma.")
    distance = cell.nodes["dist_from_soma"].to_numpy(dtype=float)
    assignment = (
        cell.nodes["hoc_assignment_distance_um"].to_numpy(dtype=float)
        if "hoc_assignment_distance_um" in cell.nodes
        else distance.copy()
    )
    if not np.isfinite(distance).all() or not np.isfinite(assignment).all():
        raise ValueError("Morphology distances contain non-finite values.")
    sections = (
        cell.nodes["hoc_section_index"].to_numpy(dtype=int)
        if "hoc_section_index" in cell.nodes
        else None
    )
    return StaticFeatures(
        distance_um=distance,
        assignment_distance_um=assignment,
        group_masks=groups,
        hoc_section_index=sections,
    )


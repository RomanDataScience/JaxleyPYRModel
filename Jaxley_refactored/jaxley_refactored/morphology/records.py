"""Value objects shared by morphology providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StaticFeatures:
    """Compartment features used by distribution rules."""

    distance_um: np.ndarray
    assignment_distance_um: np.ndarray
    group_masks: Mapping[str, np.ndarray]
    hoc_section_index: np.ndarray | None = None


@dataclass(frozen=True)
class MorphologyResult:
    """A built cell plus immutable features and provenance."""

    cell: Any
    features: StaticFeatures
    fingerprint: str
    provenance: Mapping[str, Any]


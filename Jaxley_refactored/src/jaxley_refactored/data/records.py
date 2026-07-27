"""Immutable experimental trace records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TraceRecord:
    cell_id: str
    trace_id: str
    protocol: str
    voltage_mV: np.ndarray
    current_nA: np.ndarray
    time_ms: np.ndarray
    score_mask: np.ndarray
    dt_ms: float
    initial_voltage_mV: float
    weight: float
    metadata: Mapping[str, object]
    checksums: Mapping[str, str]

    def __post_init__(self):
        arrays = (
            self.voltage_mV,
            self.current_nA,
            self.time_ms,
            self.score_mask,
        )
        size = len(self.time_ms)
        if size == 0 or any(len(array) != size for array in arrays):
            raise ValueError(
                f"{self.trace_id}/{self.protocol}: arrays must have one non-empty shape."
            )
        for array in arrays:
            array.setflags(write=False)

    @property
    def trace_key(self) -> str:
        return f"{self.cell_id}/{self.trace_id}/{self.protocol}"

    def with_weight(self, weight: float) -> "TraceRecord":
        if weight < 0.0:
            raise ValueError("Trace weight cannot be negative.")
        return replace(self, weight=float(weight))


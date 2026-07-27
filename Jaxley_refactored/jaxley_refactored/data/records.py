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

    def with_max_steps(self, max_steps: int) -> "TraceRecord":
        """Return a prefix for fast end-to-end fitting smoke tests.

        Experimental score windows often start hundreds of milliseconds into a
        trace. If the requested prefix ends before that window, every retained
        sample is scored so a short test produces a finite, useful loss.
        """
        if max_steps <= 1:
            raise ValueError("max_steps must be greater than one.")
        if len(self.time_ms) <= max_steps:
            return self
        score_mask = np.array(self.score_mask[:max_steps], copy=True)
        score_window_replaced = not np.any(score_mask)
        if score_window_replaced:
            score_mask[:] = True
        return replace(
            self,
            voltage_mV=np.array(self.voltage_mV[:max_steps], copy=True),
            current_nA=np.array(self.current_nA[:max_steps], copy=True),
            time_ms=np.array(self.time_ms[:max_steps], copy=True),
            score_mask=score_mask,
            metadata={
                **self.metadata,
                "truncated_for_smoke_test": True,
                "original_n_steps": len(self.time_ms),
                "score_window_replaced": score_window_replaced,
            },
        )

"""Load the segmented Vitale current-clamp traces.

This module intentionally does not import anything from ``NEURON_optim`` or
the rest of the repository.  The segmented files are plain vectors produced
by ``Experimental_currentClamp_Analysis/segment_current_clamp_traces.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trace:
    name: str
    protocol: str
    time_ms: np.ndarray
    voltage_mV: np.ndarray
    current_nA: np.ndarray
    epoch_start_ms: float
    epoch_stop_ms: float

    @property
    def dt_ms(self) -> float:
        dt = np.diff(self.time_ms)
        if dt.size == 0 or not np.allclose(dt, dt[0], rtol=0.0, atol=1.0e-9):
            raise ValueError(f"{self.name} has a non-uniform time grid")
        return float(dt[0])


def _vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float, ndmin=1)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid trace vector: {path}")
    return values


def _trace(directory: Path, name: str, protocol: str, start: float, stop: float) -> Trace:
    prefix = directory / protocol / f"{name}_{protocol}"
    time = _vector(prefix.with_name(prefix.name + "_t_ms.txt"))
    voltage = _vector(prefix.with_name(prefix.name + "_v.txt"))
    current_pa = _vector(prefix.with_name(prefix.name + "_i.txt"))
    if not (time.size == voltage.size == current_pa.size):
        raise ValueError(f"Length mismatch for {prefix}")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Time is not strictly increasing for {prefix}")
    return Trace(name, protocol, time, voltage, current_pa * 1.0e-3, start, stop)


def load_dataset(root: Path, cell: str, depolarizing: list[str], hyperpolarizing: str) -> tuple[list[Trace], Trace]:
    """Load the four depolarizing traces and one selected hyperpolarizing trace.

    Epoch coordinates are derived from the stable segmented-trace convention:
    depolarizing segments contain 200 ms before and 700 ms after the 300 ms
    pulse, while hyperpolarizing segments contain 500 ms before and 500 ms
    after the 50 ms pulse.  The values are also checked against the current
    waveform so a mislabeled dataset fails early.
    """
    root = Path(root)
    depol = [_trace(root / cell, name, "depolarizing_step", 200.0, 500.0) for name in depolarizing]
    hyper = _trace(root / cell, hyperpolarizing, "hyperpolarizing_pulse", 500.0, 550.0)
    for trace in [*depol, hyper]:
        if trace.time_ms[0] > 1.0e-6:
            raise ValueError(f"{trace.name} does not start at zero: {trace.time_ms[0]}")
        if trace.epoch_stop_ms >= trace.time_ms[-1]:
            raise ValueError(f"{trace.name} has no post-step recovery samples")
    return depol, hyper


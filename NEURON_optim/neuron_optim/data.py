"""Load and prepare Combe segmented current-clamp traces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Trace:
    cell: str
    trace: str
    protocol: str
    time_ms: np.ndarray
    voltage_mV: np.ndarray
    current_nA: np.ndarray
    epoch_start_ms: float
    epoch_stop_ms: float

    @property
    def epoch_duration_ms(self) -> float:
        return self.epoch_stop_ms - self.epoch_start_ms


def _read_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float, ndmin=1)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid trace vector: {path}")
    return values


def _resolve_path(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        # Metadata in the repository contains author-machine paths. Use the
        # stable suffix beginning at the segmented-trace directory instead.
        marker = "Segmented_Traces"
        if marker in candidate.parts:
            candidate = root / Path(*candidate.parts[candidate.parts.index(marker) + 1:])
        else:
            candidate = root / candidate.name
    return candidate if candidate.exists() else root / Path(raw).name


def load_manifest(root: Path, manifest: Path | None = None) -> list[dict[str, str]]:
    path = manifest or root / "segment_metadata.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_trace(row: dict[str, str], time: np.ndarray, voltage: np.ndarray,
                current_pA: np.ndarray, *, start_ms: float | None = None,
                stop_ms: float | None = None, center_current: bool = False) -> Trace:
    if not (time.size == voltage.size == current_pA.size):
        raise ValueError(f"Length mismatch in {row['trace']} {row['segment']}")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"Time is not strictly increasing in {row['trace']}")
    segment_start = float(row.get("segment_start_ms", 0.0))
    epoch_start = float(row["epoch_start_ms"]) - segment_start
    epoch_stop = float(row["epoch_stop_ms"]) - segment_start
    start = epoch_start if start_ms is None else start_ms
    stop = epoch_stop if stop_ms is None else stop_ms
    mask = (time >= start - 1e-9) & (time <= stop + 1e-9)
    if mask.sum() < 2:
        raise ValueError(f"Requested window is empty for {row['trace']} {row['segment']}")
    selected_time = time[mask] - start
    selected_current_nA = current_pA[mask] * 1e-3
    relative_epoch_start = epoch_start - start
    if center_current:
        pre_mask = selected_time <= relative_epoch_start + 1e-9
        if not pre_mask.any():
            raise ValueError(f"No pre-stimulus samples available to center {row['trace']}")
        selected_current_nA = selected_current_nA - float(np.median(selected_current_nA[pre_mask]))

    return Trace(
        cell=row["cell"], trace=row["trace"], protocol=row["segment"],
        time_ms=selected_time, voltage_mV=voltage[mask],
        current_nA=selected_current_nA,
        epoch_start_ms=relative_epoch_start,
        epoch_stop_ms=epoch_stop - start,
    )


def load_trace(root: Path, row: dict[str, str], *, start_ms: float | None = None,
               stop_ms: float | None = None, center_current: bool = False) -> Trace:
    time = _read_vector(_resolve_path(root, row["time_output"]))
    voltage = _read_vector(_resolve_path(root, row["voltage_output"]))
    current_pA = _read_vector(_resolve_path(root, row["current_output"]))
    return _make_trace(row, time, voltage, current_pA, start_ms=start_ms,
                       stop_ms=stop_ms, center_current=center_current)


def load_protocol_traces(root: Path, *, cell: str, protocol: str,
                         trace_names: Iterable[str], pre_ms: float = 0.0,
                         post_ms: float | None = None, full_trial: bool = True,
                         center_current: bool = False) -> list[Trace]:
    rows = load_manifest(root)
    wanted = set(trace_names)
    selected = [row for row in rows if row["cell"] == cell and
                row["segment"] == protocol and row["trace"] in wanted]
    if len(selected) != len(wanted):
        found = {row["trace"] for row in selected}
        raise ValueError(f"Missing {protocol} traces: {sorted(wanted - found)}")
    result: list[Trace] = []
    for row in sorted(selected, key=lambda item: item["trace"]):
        raw_time = _read_vector(_resolve_path(root, row["time_output"]))
        voltage = _read_vector(_resolve_path(root, row["voltage_output"]))
        current_pA = _read_vector(_resolve_path(root, row["current_output"]))
        if not (raw_time.size == voltage.size == current_pA.size):
            raise ValueError(f"Length mismatch in {row['trace']} {row['segment']}")
        if np.any(np.diff(raw_time) <= 0):
            raise ValueError(f"Time is not strictly increasing in {row['trace']}")
        segment_start = float(row.get("segment_start_ms", 0.0))
        epoch_start = float(row["epoch_start_ms"]) - segment_start
        epoch_stop = float(row["epoch_stop_ms"]) - segment_start
        start = (max(float(raw_time[0]), 0.0, epoch_start - pre_ms)
                 if pre_ms > 0.0 else float(raw_time[0]))
        if full_trial:
            stop = (min(float(raw_time[-1]), epoch_stop + post_ms)
                    if post_ms is not None else float(raw_time[-1]))
        else:
            stop = epoch_stop
        result.append(_make_trace(
            row, raw_time, voltage, current_pA,
            start_ms=max(start, float(raw_time[0])),
            stop_ms=stop,
            center_current=center_current,
        ))
    return result

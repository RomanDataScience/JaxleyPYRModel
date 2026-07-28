"""Manifest-driven loader for segmented current-clamp recordings."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from jaxley_refactored.config.hashing import file_sha256
from jaxley_refactored.config.schema import DatasetSpec

from .records import TraceRecord


SEGMENT_ALIASES = {
    "depolarizing_pulse": "depolarizing_step",
    "hyperpolarizing_step": "hyperpolarizing_pulse",
}


class SegmentedTraceLoader:
    """Load, validate, resample, mask, and sort selected trace triplets."""

    def load(self, spec: DatasetSpec) -> tuple[TraceRecord, ...]:
        if not spec.manifest.is_file():
            raise FileNotFoundError(spec.manifest)
        with spec.manifest.open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
        indexed_trace_ids = {
            protocol: self._select_indices(
                tuple(
                    dict.fromkeys(
                        row["trace"]
                        for row in rows
                        if row["cell"] == spec.cell_id
                        and self._canonical(row["segment"]) == protocol
                    )
                ),
                spec.trace_indices,
                spec.cell_id,
                protocol,
            )
            for protocol in spec.segments
        }
        selected = [
            row
            for row in rows
            if row["cell"] == spec.cell_id
            and self._canonical(row["segment"]) in spec.segments
            and (
                row["trace"]
                in indexed_trace_ids[self._canonical(row["segment"])]
                if spec.trace_indices
                else self._selected(row["trace"], spec.traces)
            )
        ]
        if not selected:
            raise ValueError(f"No traces selected for cell {spec.cell_id}.")
        records = tuple(self._load_row(spec, row) for row in selected)
        keys = [record.trace_key for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("Dataset manifest contains duplicate selected traces.")
        return tuple(sorted(records, key=lambda item: (item.protocol, item.trace_id)))

    @staticmethod
    def _selected(trace_id: str, selectors: Iterable[str]) -> bool:
        selectors = tuple(selectors)
        return "*" in selectors or trace_id in selectors

    @staticmethod
    def _select_indices(
        trace_ids: tuple[str, ...],
        indices: tuple[int, ...],
        cell_id: str,
        protocol: str,
    ) -> frozenset[str]:
        if not indices:
            return frozenset()
        unavailable = tuple(index for index in indices if index > len(trace_ids))
        if unavailable:
            joined = ", ".join(str(index) for index in unavailable)
            raise ValueError(
                f"Trace index/indices {joined} unavailable for "
                f"{cell_id}/{protocol}; the protocol has {len(trace_ids)} trace(s)."
            )
        return frozenset(trace_ids[index - 1] for index in indices)

    @staticmethod
    def _canonical(segment: str) -> str:
        return SEGMENT_ALIASES.get(segment, segment)

    def _load_row(self, spec: DatasetSpec, row: dict[str, str]) -> TraceRecord:
        protocol = self._canonical(row["segment"])
        stem = f"{row['trace']}_{protocol}"
        directory = spec.root / spec.cell_id / protocol
        paths = {
            "voltage": directory / f"{stem}_v.txt",
            "current": directory / f"{stem}_i.txt",
            "time": directory / f"{stem}_t_ms.txt",
        }
        for kind, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"{spec.cell_id}/{row['trace']}/{protocol}: missing {kind} file {path}"
                )
        voltage = self._read(paths["voltage"], "voltage", row)
        current = self._read(paths["current"], "current", row)
        time = self._read(paths["time"], "time", row)
        if not (len(voltage) == len(current) == len(time)):
            raise ValueError(
                f"{spec.cell_id}/{row['trace']}/{protocol}: v/i/t length mismatch "
                f"{len(voltage)}/{len(current)}/{len(time)}."
            )
        dt = self._validate_time(time, row)
        current = current * spec.current_scale_to_nA
        if not np.isclose(dt, spec.target_dt_ms, rtol=1e-9, atol=1e-12):
            voltage, current, time = self._resample(
                voltage, current, time, spec.target_dt_ms
            )
        dt = spec.target_dt_ms
        time = np.arange(len(time), dtype=float) * dt

        epoch_start = float(row["epoch_start_ms"]) - float(row["segment_start_ms"])
        epoch_stop = float(row["epoch_stop_ms"]) - float(row["segment_start_ms"])
        score_start = max(float(time[0]), epoch_start - spec.score_pre_ms)
        score_stop = min(float(time[-1]), epoch_stop + spec.score_post_ms)
        mask = (time >= score_start) & (time <= score_stop)
        if not np.any(mask):
            raise ValueError(
                f"{spec.cell_id}/{row['trace']}/{protocol}: score mask is empty."
            )
        expected = int(row["n_samples"])
        if expected != len(self._read(paths["time"], "time", row)):
            raise ValueError(
                f"{spec.cell_id}/{row['trace']}/{protocol}: manifest n_samples "
                f"{expected} does not match time file."
            )
        return TraceRecord(
            cell_id=spec.cell_id,
            trace_id=row["trace"],
            protocol=protocol,
            voltage_mV=np.asarray(voltage, dtype=float),
            current_nA=np.asarray(current, dtype=float),
            time_ms=np.asarray(time, dtype=float),
            score_mask=np.asarray(mask, dtype=bool),
            dt_ms=float(dt),
            initial_voltage_mV=float(voltage[0]),
            weight=0.0,
            metadata={
                "epoch_start_ms": epoch_start,
                "epoch_stop_ms": epoch_stop,
                "was_clipped": row["was_clipped"].lower() == "true",
                "epoch_current_delta_pA": float(row["epoch_current_delta_pA"]),
            },
            checksums={kind: file_sha256(path) for kind, path in paths.items()},
        )

    @staticmethod
    def _read(path: Path, kind: str, row: dict[str, str]) -> np.ndarray:
        values = np.atleast_1d(np.loadtxt(path, dtype=float))
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(
                f"{row['cell']}/{row['trace']}/{row['segment']}: "
                f"{kind} must be a finite one-dimensional array."
            )
        return values

    @staticmethod
    def _validate_time(time: np.ndarray, row: dict[str, str]) -> float:
        if len(time) < 2:
            raise ValueError(
                f"{row['cell']}/{row['trace']}/{row['segment']}: "
                "time needs at least two samples."
            )
        differences = np.diff(time)
        if np.any(differences <= 0.0):
            raise ValueError(
                f"{row['cell']}/{row['trace']}/{row['segment']}: "
                "time must be strictly increasing."
            )
        dt = float(np.median(differences))
        if not np.allclose(differences, dt, rtol=1e-7, atol=1e-10):
            raise ValueError(
                f"{row['cell']}/{row['trace']}/{row['segment']}: "
                "time must be uniformly sampled."
            )
        return dt

    @staticmethod
    def _resample(voltage, current, time, target_dt):
        target_size = int(round((time[-1] - time[0]) / target_dt)) + 1
        target_time = time[0] + np.arange(target_size) * target_dt
        target_voltage = np.interp(target_time, time, voltage)
        source_indices = np.searchsorted(time, target_time, side="right") - 1
        source_indices = np.clip(source_indices, 0, len(current) - 1)
        return target_voltage, current[source_indices], target_time

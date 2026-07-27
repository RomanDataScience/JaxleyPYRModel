"""Deterministic weighting and static-shape trace buckets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .records import TraceRecord


@dataclass(frozen=True)
class TraceBucket:
    """Same-shape records suitable for one JAX compilation."""

    key: tuple[float, int]
    records: tuple[TraceRecord, ...]
    currents_nA: np.ndarray
    observed_mV: np.ndarray
    score_masks: np.ndarray
    weights: np.ndarray
    initial_voltage_mV: np.ndarray

    @property
    def dt_ms(self) -> float:
        return self.key[0]

    @property
    def n_steps(self) -> int:
        return self.key[1]


def weight_records(
    records: Iterable[TraceRecord],
    *,
    aggregation: str,
    protocol_weights: Mapping[str, float],
) -> tuple[TraceRecord, ...]:
    records = tuple(records)
    if aggregation == "protocol_mean":
        groups: dict[str, list[TraceRecord]] = defaultdict(list)
        for record in records:
            groups[record.protocol].append(record)
        missing = set(groups) - protocol_weights.keys()
        if missing:
            raise ValueError(f"Missing protocol weights: {sorted(missing)}")
        weighted = [
            record.with_weight(protocol_weights[record.protocol] / len(groups[record.protocol]))
            for record in records
        ]
    elif aggregation == "trace_mean":
        weighted = [record.with_weight(1.0 / len(records)) for record in records]
    elif aggregation == "sample_mean":
        total = sum(int(record.score_mask.sum()) for record in records)
        weighted = [
            record.with_weight(int(record.score_mask.sum()) / total)
            for record in records
        ]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    total_weight = sum(record.weight for record in weighted)
    if not np.isclose(total_weight, 1.0, atol=1e-12):
        raise ValueError(f"Trace weights sum to {total_weight}, expected 1.")
    return tuple(weighted)


def bucket_records(
    records: Iterable[TraceRecord], *, pad_to_longest: bool = False
) -> tuple[TraceBucket, ...]:
    """Stack traces into static-shape batches.

    With ``pad_to_longest=True``, records sharing a time step are padded to the
    longest trace. Padded score-mask entries are false, so they contribute
    exactly zero loss while enabling a single ``vmap`` across all such traces.
    """
    grouped: dict[tuple[float, int], list[TraceRecord]] = defaultdict(list)
    for record in records:
        length_key = 0 if pad_to_longest else len(record.time_ms)
        grouped[(record.dt_ms, length_key)].append(record)
    buckets = []
    for grouping_key, items in sorted(grouped.items()):
        records_in_bucket = tuple(
            sorted(items, key=lambda item: (item.protocol, item.trace_id))
        )
        n_steps = max(len(record.time_ms) for record in records_in_bucket)
        key = (grouping_key[0], n_steps)

        def padded(values, *, fill):
            missing = n_steps - len(values)
            return np.pad(values, (0, missing), constant_values=fill)

        buckets.append(
            TraceBucket(
                key=key,
                records=records_in_bucket,
                currents_nA=np.stack(
                    [
                        padded(record.current_nA, fill=0.0)
                        for record in records_in_bucket
                    ]
                ),
                observed_mV=np.stack(
                    [
                        padded(record.voltage_mV, fill=0.0)
                        for record in records_in_bucket
                    ]
                ),
                score_masks=np.stack(
                    [
                        padded(record.score_mask, fill=False)
                        for record in records_in_bucket
                    ]
                ),
                weights=np.asarray(
                    [record.weight for record in records_in_bucket], dtype=float
                ),
                initial_voltage_mV=np.asarray(
                    [record.initial_voltage_mV for record in records_in_bucket],
                    dtype=float,
                ),
            )
        )
    return tuple(buckets)

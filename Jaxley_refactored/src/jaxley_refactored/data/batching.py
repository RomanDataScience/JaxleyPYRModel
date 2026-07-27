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


def bucket_records(records: Iterable[TraceRecord]) -> tuple[TraceBucket, ...]:
    grouped: dict[tuple[float, int], list[TraceRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.dt_ms, len(record.time_ms))].append(record)
    buckets = []
    for key, items in sorted(grouped.items()):
        records_in_bucket = tuple(
            sorted(items, key=lambda item: (item.protocol, item.trace_id))
        )
        buckets.append(
            TraceBucket(
                key=key,
                records=records_in_bucket,
                currents_nA=np.stack(
                    [record.current_nA for record in records_in_bucket]
                ),
                observed_mV=np.stack(
                    [record.voltage_mV for record in records_in_bucket]
                ),
                score_masks=np.stack(
                    [record.score_mask for record in records_in_bucket]
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


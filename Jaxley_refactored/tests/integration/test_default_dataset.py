from dataclasses import replace
from pathlib import Path

import numpy as np

from jaxley_refactored.config import load_config
from jaxley_refactored.data import SegmentedTraceLoader, bucket_records, weight_records


PROJECT = Path(__file__).resolve().parents[2]


def test_default_cell_loads_first_and_third_segmented_traces():
    config = load_config(PROJECT / "configs/runtimes/cpu_x64.yaml")
    records = SegmentedTraceLoader().load(config.dataset)
    weighted = weight_records(
        records,
        aggregation=config.fit.aggregation,
        protocol_weights=config.fit.protocol_weights,
    )
    buckets = bucket_records(
        weighted, pad_to_longest=config.fit.pad_to_longest
    )

    assert len(records) == 4
    assert {record.trace_id for record in records} == {"v75ctrl", "v77ctrl"}
    assert {(bucket.n_steps, len(bucket.records)) for bucket in buckets} == {
        (13_000, 2),
        (24_000, 2),
    }
    assert np.isclose(sum(record.weight for record in weighted), 1.0)
    assert all(np.isclose(record.dt_ms, 0.05) for record in records)

    smoke_records = tuple(record.with_max_steps(100) for record in weighted)
    smoke_buckets = bucket_records(smoke_records)
    assert [(bucket.n_steps, len(bucket.records)) for bucket in smoke_buckets] == [
        (100, 4)
    ]
    assert smoke_buckets[0].score_masks.all()


def test_trace_indices_select_cell_specific_first_and_third_traces():
    config = load_config(PROJECT / "configs/runtimes/cpu_x64.yaml")
    selection = replace(config.dataset, traces=(), trace_indices=(1, 3))

    first_cell = SegmentedTraceLoader().load(selection)
    second_cell = SegmentedTraceLoader().load(
        replace(selection, cell_id="m20260331b")
    )

    assert {record.trace_id for record in first_cell} == {"v75ctrl", "v77ctrl"}
    assert {record.trace_id for record in second_cell} == {"v34ctrl", "v43ctrl"}
    assert len(first_cell) == len(second_cell) == 4

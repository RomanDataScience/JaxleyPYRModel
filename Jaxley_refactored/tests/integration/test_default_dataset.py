from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jaxley_refactored.config import load_config
from jaxley_refactored.data import SegmentedTraceLoader, bucket_records, weight_records


PROJECT = Path(__file__).resolve().parents[2]


def test_default_cell_loads_first_and_third_segmented_traces():
    config = load_config(PROJECT / "configs/runtimes/cpu_x64.yaml")
    assert config.dataset.simulation_post_ms is None
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


def test_lsu_applies_protocol_specific_simulation_horizons():
    config = load_config(PROJECT / "configs/losses/LSU_1.yaml")
    training = replace(config.dataset, cell_id="m20260331b")
    records = SegmentedTraceLoader().load(training)

    assert training.simulation_post_ms == 500.0
    assert training.simulation_post_ms_by_protocol == {
        "hyperpolarizing_pulse": 150.0,
    }
    assert training.score_post_ms == 500.0
    assert {(record.protocol, len(record.time_ms)) for record in records} == {
        ("depolarizing_step", 10_001),
        ("hyperpolarizing_pulse", 7_001),
    }
    assert all(record.score_mask[-1] for record in records)
    for record in records:
        expected_post_ms = (
            500.0 if record.protocol == "depolarizing_step" else 150.0
        )
        expected_stop_ms = (
            1_000.0 if record.protocol == "depolarizing_step" else 700.0
        )
        post_stimulus_ms = (
            record.time_ms[-1] - record.metadata["epoch_stop_ms"]
        )
        assert np.isclose(post_stimulus_ms, expected_post_ms, atol=record.dt_ms)
        assert np.isclose(record.time_ms[-1], expected_stop_ms)
        assert np.isclose(
            record.metadata["simulation_actual_post_stimulus_ms"],
            expected_post_ms,
            atol=record.dt_ms,
        )

    buckets = bucket_records(records)
    for bucket in buckets:
        recovery_samples = (
            5_000
            if bucket.records[0].protocol == "depolarizing_step"
            else 1_500
        )
        np.testing.assert_array_equal(
            bucket.window_masks["recovery"].sum(axis=1),
            np.full(len(bucket.records), recovery_samples),
        )
        assert bucket.window_masks["outside_stimulus"][:, -1].all()

    depolarizing = next(
        record for record in records if record.protocol == "depolarizing_step"
    )
    assert depolarizing.with_max_steps(12_000) is depolarizing
    smoke_prefix = depolarizing.with_max_steps(100)
    assert len(smoke_prefix.time_ms) == 100
    assert smoke_prefix.metadata["simulation_post_stimulus_ms"] == 500.0
    assert smoke_prefix.metadata["original_n_steps"] == 10_001

    validation = replace(training, trace_indices=(2, 4))
    validation_records = SegmentedTraceLoader().load(validation)
    assert {record.trace_id for record in validation_records} == {
        "v42ctrl",
        "v44ctrl",
    }
    assert all(
        np.isclose(
            record.time_ms[-1] - record.metadata["epoch_stop_ms"],
            500.0 if record.protocol == "depolarizing_step" else 150.0,
            atol=record.dt_ms,
        )
        for record in validation_records
    )


def test_lsu_rejects_a_protocol_horizon_when_the_recording_is_shorter():
    config = load_config(PROJECT / "configs/losses/LSU_1.yaml")

    with pytest.raises(
        ValueError,
        match=r"requested 150 ms after stimulus, but only .* ms are available",
    ):
        SegmentedTraceLoader().load(
            replace(config.dataset, cell_id="m20240527cd")
        )

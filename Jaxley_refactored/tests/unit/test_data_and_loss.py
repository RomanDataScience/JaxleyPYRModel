import numpy as np

from jaxley_refactored.data import TraceRecord, bucket_records, weight_records
from jaxley_refactored.fitting.losses import masked_mse, weighted_bucket_loss


def _record(trace, protocol, size, score_samples):
    time = np.arange(size, dtype=float) * 0.05
    return TraceRecord(
        cell_id="cell",
        trace_id=trace,
        protocol=protocol,
        voltage_mV=np.zeros(size),
        current_nA=np.zeros(size),
        time_ms=time,
        score_mask=np.arange(size) < score_samples,
        dt_ms=0.05,
        initial_voltage_mV=-70.0,
        weight=0.0,
        metadata={},
        checksums={},
    )


def test_protocol_mean_is_invariant_to_trace_count():
    records = (
        _record("a", "depolarizing_step", 3, 3),
        _record("b", "depolarizing_step", 3, 3),
        _record("c", "hyperpolarizing_pulse", 5, 5),
    )
    weighted = weight_records(
        records,
        aggregation="protocol_mean",
        protocol_weights={
            "depolarizing_step": 0.5,
            "hyperpolarizing_pulse": 0.5,
        },
    )

    np.testing.assert_allclose([item.weight for item in weighted], [0.25, 0.25, 0.5])
    assert [bucket.key for bucket in bucket_records(weighted)] == [(0.05, 3), (0.05, 5)]
    padded = bucket_records(weighted, pad_to_longest=True)
    assert [(bucket.key, len(bucket.records)) for bucket in padded] == [
        ((0.05, 5), 3)
    ]
    assert not padded[0].score_masks[0, 3:].any()


def test_masked_loss_uses_only_scored_samples_and_trace_weights():
    predicted = np.asarray([[1.0, 10.0], [2.0, 4.0]])
    observed = np.zeros_like(predicted)
    mask = np.asarray([[True, False], [True, True]])

    np.testing.assert_allclose(masked_mse(predicted, observed, mask), [1.0, 10.0])
    assert np.isclose(
        float(weighted_bucket_loss(predicted, observed, mask, [0.25, 0.75])),
        7.75,
    )


def test_short_smoke_prefix_replaces_an_out_of_range_score_window():
    record = _record("a", "depolarizing_step", 1000, 0)
    shortened = record.with_max_steps(100)

    assert len(shortened.time_ms) == 100
    assert shortened.score_mask.all()
    assert shortened.metadata["original_n_steps"] == 1000
    assert shortened.metadata["score_window_replaced"] is True


def test_outside_stimulus_window_excludes_step_and_padded_samples():
    short = TraceRecord(
        cell_id="cell",
        trace_id="short",
        protocol="depolarizing_step",
        voltage_mV=np.zeros(6),
        current_nA=np.zeros(6),
        time_ms=np.arange(6, dtype=float),
        score_mask=np.ones(6, dtype=bool),
        dt_ms=1.0,
        initial_voltage_mV=-65.0,
        weight=0.5,
        metadata={"epoch_start_ms": 2.0, "epoch_stop_ms": 4.0},
        checksums={},
    )
    long = TraceRecord(
        cell_id="cell",
        trace_id="long",
        protocol="depolarizing_step",
        voltage_mV=np.zeros(8),
        current_nA=np.zeros(8),
        time_ms=np.arange(8, dtype=float),
        score_mask=np.ones(8, dtype=bool),
        dt_ms=1.0,
        initial_voltage_mV=-65.0,
        weight=0.5,
        metadata={"epoch_start_ms": 2.0, "epoch_stop_ms": 4.0},
        checksums={},
    )

    bucket = bucket_records((short, long), pad_to_longest=True)[0]
    short_row = next(
        index
        for index, record in enumerate(bucket.records)
        if record.trace_id == "short"
    )
    np.testing.assert_array_equal(
        bucket.window_masks["outside_stimulus"][short_row],
        [True, True, False, False, False, True, False, False],
    )

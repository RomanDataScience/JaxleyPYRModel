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

from types import SimpleNamespace

import numpy as np

from jaxley_refactored.fitting.metrics import spike_feature_metrics


def test_spike_feature_metrics_store_per_trace_counts_and_rates():
    observed = np.asarray([-65.0, 20.0, -65.0, 20.0, -65.0, -65.0])
    simulated = np.asarray([-65.0, 20.0, -65.0, -65.0, -65.0, -65.0])
    record = SimpleNamespace(
        trace_key="cell/trace/depolarizing_step",
        protocol="depolarizing_step",
        time_ms=np.arange(6.0),
        voltage_mV=observed,
        dt_ms=1.0,
    )
    bucket = SimpleNamespace(
        key=(1.0, 6),
        records=(record,),
        window_masks={
            "stimulus": np.ones((1, 6), dtype=bool),
            "outside_stimulus": np.zeros((1, 6), dtype=bool),
        },
    )

    metrics = spike_feature_metrics(
        (bucket,), {bucket.key: simulated[None, :]}, threshold_mV=-20.0
    )["spiking"]
    trace = metrics["traces"][0]

    assert trace["experimental_spike_count"] == 2
    assert trace["simulated_spike_count"] == 1
    assert trace["experimental_rate_hz"] == 400.0
    assert trace["simulated_rate_hz"] == 200.0
    assert trace["rate_error_hz"] == -200.0
    assert metrics["rms_rate_error_hz"] == 200.0
    assert metrics["mean_absolute_rate_error_hz"] == 200.0


def test_spike_feature_metrics_mark_forbidden_spikes_ineligible():
    depolarizing = SimpleNamespace(
        trace_key="cell/trace/depolarizing_step",
        protocol="depolarizing_step",
        time_ms=np.arange(6.0),
        voltage_mV=np.full(6, -65.0),
        dt_ms=1.0,
    )
    hyperpolarizing = SimpleNamespace(
        trace_key="cell/trace/hyperpolarizing_pulse",
        protocol="hyperpolarizing_pulse",
        time_ms=np.arange(6.0),
        voltage_mV=np.full(6, -65.0),
        dt_ms=1.0,
    )
    bucket = SimpleNamespace(
        key=(1.0, 6),
        records=(depolarizing, hyperpolarizing),
        window_masks={
            "stimulus": np.asarray(
                [[False, False, True, True, False, False]] * 2
            ),
            "outside_stimulus": np.asarray(
                [[True, True, False, False, True, True]] * 2
            ),
        },
    )
    predictions = np.asarray(
        [
            [-65.0, -65.0, -65.0, -65.0, 20.0, -65.0],
            [-65.0, -65.0, 20.0, -65.0, -65.0, -65.0],
        ]
    )

    metrics = spike_feature_metrics((bucket,), {bucket.key: predictions})

    assert metrics["constraints"] == {
        "eligible": False,
        "forbidden_spike_count": 2,
    }
    assert [
        trace["forbidden_spike_count"]
        for trace in metrics["spiking"]["traces"]
    ] == [1, 1]

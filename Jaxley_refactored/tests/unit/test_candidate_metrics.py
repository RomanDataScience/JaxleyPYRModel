from types import SimpleNamespace

import numpy as np

from jaxley_refactored.fitting.metrics import spike_feature_metrics


def test_spike_feature_metrics_store_per_trace_counts_and_rates():
    observed = np.asarray([-65.0, 20.0, -65.0, 20.0, -65.0, -65.0])
    simulated = np.asarray([-65.0, 20.0, -65.0, -65.0, -65.0, -65.0])
    record = SimpleNamespace(
        trace_key="cell/trace/depolarizing_step",
        time_ms=np.arange(6.0),
        voltage_mV=observed,
        dt_ms=1.0,
    )
    bucket = SimpleNamespace(
        key=(1.0, 6),
        records=(record,),
        window_masks={"stimulus": np.ones((1, 6), dtype=bool)},
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

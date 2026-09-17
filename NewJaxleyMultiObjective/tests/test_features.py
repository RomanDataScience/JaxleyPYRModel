from types import SimpleNamespace

import numpy as np

from newjaxley_multiobjective.features import FeatureConfig, extract_features


def record(protocol, time, start, stop, current):
    return SimpleNamespace(
        trace_key=f"cell/trace/{protocol}",
        protocol=protocol,
        time_ms=np.asarray(time, dtype=float),
        voltage_mV=np.zeros(len(time), dtype=float),
        current_nA=np.asarray(current, dtype=float),
        score_mask=np.ones(len(time), dtype=bool),
        dt_ms=float(time[1] - time[0]),
        metadata={"epoch_start_ms": start, "epoch_stop_ms": stop},
    )


def test_hyperpolarizing_resistance_and_sag_are_measured_with_positive_sign():
    time = np.arange(0.0, 100.0, 1.0)
    current = np.zeros_like(time)
    current[50:81] = -0.1
    voltage = np.full_like(time, -60.0)
    voltage[50:55] = -70.0
    voltage[70:81] = -65.0
    item = record("hyperpolarizing_pulse", time, 50.0, 80.0, current)
    item.voltage_mV = voltage

    result = extract_features(item, voltage, FeatureConfig())

    assert result["features"]["peak_input_resistance"] == 100.0
    assert result["features"]["steady_state_input_resistance"] == 50.0
    assert result["features"]["sag"] == 0.5


def test_depolarizing_count_and_last_isi_are_measured():
    time = np.arange(0.0, 150.0, 0.1)
    voltage = np.full_like(time, -65.0)
    for center in (50.0, 80.0, 110.0):
        indexes = np.abs(time - center) < 1.0
        voltage[indexes] = 30.0 - 95.0 * np.abs(time[indexes] - center)
    current = np.zeros_like(time)
    current[(time >= 20.0) & (time <= 120.0)] = 0.1
    item = record("depolarizing_step", time, 20.0, 120.0, current)
    item.voltage_mV = voltage

    result = extract_features(item, voltage, FeatureConfig())

    assert result["features"]["ap_count"] == 3.0
    assert abs(result["features"]["last_stabilized_isi"] - 30.0) < 0.2
    assert np.isfinite(result["features"]["last_ap_half_width"])

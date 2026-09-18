from types import SimpleNamespace

import numpy as np

from newjaxley_multiobjective.config import FeatureConfig, ObjectiveConfig
from newjaxley_multiobjective.objectives import evaluate_objectives, trajectory_overlap


def test_trajectory_overlap_penalizes_shift_and_offset_without_mean_subtraction():
    time = np.arange(0.0, 10.0, 1.0)
    observed = np.sin(time)
    record = SimpleNamespace(
        time_ms=time,
        voltage_mV=observed,
        score_mask=np.ones(time.size, dtype=bool),
        metadata={"epoch_start_ms": 4.0, "epoch_stop_ms": 8.0},
    )
    config = ObjectiveConfig(overlap_sigma_mV=0.5)
    exact = trajectory_overlap(record, observed, config)
    offset = trajectory_overlap(record, observed + 2.0, config)

    assert exact["loss"] == 0.0
    assert offset["loss"] > exact["loss"]


def test_spike_guard_returns_maximum_loss():
    time = np.arange(0.0, 100.0, 0.1)
    voltage = np.full(time.size, -65.0)
    spike = np.abs(time - 90.0) < 1.0
    voltage[spike] = 30.0 - 95.0 * np.abs(time[spike] - 90.0)
    record = SimpleNamespace(
        trace_key="cell/trace/depolarizing_step",
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=np.full(time.size, -65.0),
        current_nA=np.zeros(time.size),
        score_mask=np.ones(time.size, dtype=bool),
        dt_ms=0.1,
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 80.0},
    )
    bucket = SimpleNamespace(key="bucket", records=(record,))
    experimental = {record.trace_key: {"features": {label: np.nan for label in (
        "resting_membrane_potential", "average_fahp", "time_mahp", "amplitude_mahp",
        "ap_count", "last_stabilized_isi", "last_ap_half_width",
        "peak_input_resistance", "steady_state_input_resistance", "sag",
    )}}}

    values, details = evaluate_objectives(
        (record,),
        {"bucket": voltage[None, :]},
        (bucket,),
        experimental,
        FeatureConfig(),
        ObjectiveConfig(spike_violation_loss=100000.0),
    )

    assert values == (100000.0,) * 11
    assert details["spike_violations"][0]["reason"] == "depolarizing_spike_outside_stimulus"

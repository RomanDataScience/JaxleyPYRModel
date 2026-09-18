from types import SimpleNamespace

import numpy as np

from newjaxley_multiobjective.config import FeatureConfig, ObjectiveConfig
from newjaxley_multiobjective.features import extract_features
from newjaxley_multiobjective.objectives import evaluate_objectives, robust_trajectory_loss, trajectory_overlap


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


def test_ap_count_guard_returns_maximum_loss_when_count_is_far_off():
    time = np.arange(0.0, 100.0, 0.1)
    voltage = np.full(time.size, -65.0)
    record = SimpleNamespace(
        trace_key="cell/trace/depolarizing_step",
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=voltage,
        current_nA=np.zeros(time.size),
        score_mask=np.ones(time.size, dtype=bool),
        dt_ms=0.1,
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 80.0},
    )
    bucket = SimpleNamespace(key="bucket", records=(record,))
    experimental = {record.trace_key: {"features": {
        "resting_membrane_potential": -65.0,
        "ap_count": 5.0,
    }}}

    values, details = evaluate_objectives(
        (record,),
        {"bucket": voltage[None, :]},
        (bucket,),
        experimental,
        FeatureConfig(),
        ObjectiveConfig(ap_count_tolerance=1.0, ap_count_violation_loss=100000.0),
    )

    assert values == (100000.0,) * 11
    assert details["ap_count_violations"][0]["absolute_error"] == 5.0
    assert details["objectives"]["ap_count"]["type"] == "hard_ap_count_constraint"


def test_ap_count_only_mode_keeps_count_mismatch_graded():
    time = np.arange(0.0, 200.0, 0.1)
    observed = np.full(time.size, -65.0)
    candidate = np.full(time.size, -65.0)
    for center in (40.0, 70.0, 100.0, 130.0, 160.0):
        indexes = np.abs(time - center) < 1.0
        observed[indexes] = 30.0 - 95.0 * np.abs(time[indexes] - center)
    for center in (40.0, 100.0, 160.0):
        indexes = np.abs(time - center) < 1.0
        candidate[indexes] = 30.0 - 95.0 * np.abs(time[indexes] - center)
    record = SimpleNamespace(
        trace_key="cell/trace2/depolarizing_step",
        trace_id="trace2",
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=observed,
        current_nA=np.zeros(time.size),
        score_mask=np.ones(time.size, dtype=bool),
        dt_ms=0.1,
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 180.0},
    )
    bucket = SimpleNamespace(key="bucket", records=(record,))
    experimental = {record.trace_key: extract_features(record, observed, FeatureConfig())}

    values, details = evaluate_objectives(
        (record,),
        {"bucket": candidate[None, :]},
        (bucket,),
        experimental,
        FeatureConfig(),
        ObjectiveConfig(mode="ap_count_only"),
    )

    assert values == (4.0,)
    assert details["objectives"]["ap_count"]["type"] == "ap_count_only"
    assert details["ap_count_by_trace"][record.trace_key]["absolute_error"] == 2.0


def test_robust_trajectory_loss_is_time_locked_and_zero_for_exact_trace():
    time = np.arange(0.0, 100.0, 0.1)
    observed = -65.0 + 2.0 * np.sin(time / 10.0)
    record = SimpleNamespace(
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=observed,
        score_mask=np.ones(time.size, dtype=bool),
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 80.0},
    )
    config = ObjectiveConfig(mode="depolarizing_fidelity", trajectory_huber_delta_mV=2.0)
    exact = robust_trajectory_loss(record, observed, config, FeatureConfig())
    shifted = robust_trajectory_loss(record, np.roll(observed, 10), config, FeatureConfig())

    assert exact["loss"] == 0.0
    assert shifted["loss"] > exact["loss"]


def test_depolarizing_mode_returns_one_voltage_objective_per_trace():
    time = np.arange(0.0, 100.0, 0.1)
    voltage = np.full(time.size, -65.0)
    record = SimpleNamespace(
        trace_key="cell/trace2/depolarizing_step",
        trace_id="trace2",
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=voltage,
        current_nA=np.zeros(time.size),
        score_mask=np.ones(time.size, dtype=bool),
        dt_ms=0.1,
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 80.0},
    )
    bucket = SimpleNamespace(key="bucket", records=(record,))
    experimental = {record.trace_key: {"features": {"ap_count": 0.0}}}

    values, details = evaluate_objectives(
        (record,),
        {"bucket": voltage[None, :]},
        (bucket,),
        experimental,
        FeatureConfig(),
        ObjectiveConfig(mode="depolarizing_fidelity"),
    )

    assert len(values) == 4
    assert "depolarizing_trace_trace2_voltage" in details["objectives"]
    assert "spike_timing_count" in details["objectives"]


def test_mocma_four_objectives_are_zero_for_an_exact_trace():
    time = np.arange(0.0, 220.0, 0.1)
    voltage = np.full_like(time, -65.0)
    for center in (60.0, 100.0, 140.0):
        indexes = np.abs(time - center) < 1.0
        voltage[indexes] = 30.0 - 95.0 * np.abs(time[indexes] - center)
    voltage[(time > 141.0) & (time < 150.0)] = -62.0
    current = np.zeros_like(time)
    current[(time >= 20.0) & (time <= 160.0)] = 0.1
    record = SimpleNamespace(
        trace_key="cell/trace2/depolarizing_step",
        trace_id="trace2",
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=voltage,
        current_nA=current,
        score_mask=np.ones(time.size, dtype=bool),
        dt_ms=0.1,
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 160.0},
    )
    bucket = SimpleNamespace(key="bucket", records=(record,))
    experimental = {record.trace_key: extract_features(record, voltage, FeatureConfig())}

    values, details = evaluate_objectives(
        (record,),
        {"bucket": voltage[None, :]},
        (bucket,),
        experimental,
        FeatureConfig(),
        ObjectiveConfig(mode="mocma_four_objectives"),
    )

    assert values == (0.0, 0.0, 0.0, 0.0)
    assert tuple(details["objectives"]) == (
        "firing_rate",
        "spike_shape",
        "post_stimulus_recovery",
        "depolarized_plateau",
    )


def test_mocma_four_objectives_penalize_a_post_spike_hyperpolarizing_dip():
    time = np.arange(0.0, 220.0, 0.1)
    experimental_voltage = np.full_like(time, -65.0)
    for center in (60.0, 100.0, 140.0):
        indexes = np.abs(time - center) < 1.0
        experimental_voltage[indexes] = 30.0 - 95.0 * np.abs(time[indexes] - center)
    current = np.zeros_like(time)
    current[(time >= 20.0) & (time <= 160.0)] = 0.1
    record = SimpleNamespace(
        trace_key="cell/trace2/depolarizing_step",
        protocol="depolarizing_step",
        time_ms=time,
        voltage_mV=experimental_voltage,
        current_nA=current,
        score_mask=np.ones(time.size, dtype=bool),
        dt_ms=0.1,
        metadata={"epoch_start_ms": 20.0, "epoch_stop_ms": 160.0},
    )
    candidate = experimental_voltage.copy()
    # Place the dip outside the ±8 ms AP-shape window so this isolates the
    # plateau objective.
    candidate[(time > 109.0) & (time < 115.0)] = -85.0
    bucket = SimpleNamespace(key="bucket", records=(record,))
    experimental = {record.trace_key: extract_features(record, experimental_voltage, FeatureConfig())}

    values, _ = evaluate_objectives(
        (record,),
        {"bucket": candidate[None, :]},
        (bucket,),
        experimental,
        FeatureConfig(),
        ObjectiveConfig(mode="mocma_four_objectives"),
    )

    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] == 0.0
    assert values[3] > 0.0

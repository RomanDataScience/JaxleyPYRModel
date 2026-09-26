import numpy as np

from neuron_optim.data import Trace
from neuron_optim.objective import (
    SimulationOutput,
    _spike_height_loss,
    build_objective_context,
    depolarizing_objective,
    detect_spikes,
    hyperpolarizing_objective,
)


def trace(protocol="hyperpolarizing_pulse"):
    time = np.arange(0.0, 300.0 if protocol.startswith("depolarizing") else 200.0, 0.1)
    voltage = np.full(time.size, -70.0)
    start, stop = ((100.0, 130.0) if protocol.startswith("hyper") else (50.0, 150.0))
    voltage[(time >= start) & (time <= stop)] = -80.0 if protocol.startswith("hyper") else -45.0
    return Trace("cell", "v75ctrl", protocol, time, voltage,
                 np.zeros_like(time), start, stop)


def test_detect_spikes_finds_threshold_crossing():
    time = np.arange(0.0, 20.0, 0.1)
    voltage = np.full(time.size, -70.0)
    voltage[50:53] = [-10.0, 30.0, -10.0]
    assert len(detect_spikes(time, voltage, prominence_mV=1.0)) == 1


def test_detect_spikes_rejects_sustained_depolarized_plateau():
    time = np.arange(0.0, 100.0, 0.1)
    voltage = np.full(time.size, -70.0)
    voltage[100:800] = -10.0
    assert len(detect_spikes(
        time, voltage, prominence_mV=1.0, max_spike_width_ms=10.0
    )) == 0


def test_hyper_exponential_loss_is_zero_for_matching_trace():
    observed = trace()
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV.copy())
    result = hyperpolarizing_objective([observed], [simulated])
    assert result.value == 0.0


def test_hyper_objective_penalizes_absolute_baseline_offset():
    observed = trace()
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV + 5.0)
    result = hyperpolarizing_objective([observed], [simulated])
    assert result.value > 0.0
    offset = result.details["traces"][0]["voltage_offset_mV"]
    assert offset["error"] == 5.0
    assert offset["loss"] > 0.0


def test_hyper_objective_can_separate_delta_v_from_voltage_offset():
    observed = trace()
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV + 5.0)
    shape_only = hyperpolarizing_objective(
        [observed], [simulated], voltage_offset_weight=0.0
    )
    combined = hyperpolarizing_objective([observed], [simulated])
    assert shape_only.value == 0.0
    assert combined.value > shape_only.value


def test_hyper_objective_reports_explicit_voltage_deflection():
    observed = trace()
    simulated = SimulationOutput(observed.time_ms, np.full(observed.time_ms.size, -70.0))
    result = hyperpolarizing_objective([observed], [simulated])
    deflection = result.details["traces"][0]["deflection_mV"]
    assert deflection["experimental"] == -10.0
    assert deflection["simulated"] == 0.0
    assert deflection["loss"] > 0.0


def test_hyper_spike_penalty_overrides_voltage_loss():
    observed = trace()
    voltage = observed.voltage_mV.copy()
    voltage[1050:1053] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, voltage)
    result = hyperpolarizing_objective([observed], [simulated], prominence_mV=1.0,
                                       spike_penalty=1234.0)
    assert result.value == 1234.0


def test_hyper_axonal_spike_uses_same_penalty():
    observed = trace()
    axon = observed.voltage_mV.copy()
    axon[1050:1053] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV.copy(), axon)
    result = hyperpolarizing_objective([observed], [simulated], prominence_mV=1.0,
                                       spike_penalty=1234.0)
    assert result.value == 1234.0


def test_depolarizing_extra_spikes_penalty():
    observed = trace("depolarizing_step")
    voltage = observed.voltage_mV.copy()
    for center in (2500, 2600):
        voltage[center:center + 3] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, voltage)
    result = depolarizing_objective([observed], [simulated], prominence_mV=1.0,
                                    extra_spike_penalty=4321.0)
    assert result.value == 4321.0


def test_depolarizing_axonal_spike_outside_step_uses_same_penalty():
    observed = trace("depolarizing_step")
    axon = observed.voltage_mV.copy()
    axon[200:203] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV.copy(), axon)
    result = depolarizing_objective([observed], [simulated], prominence_mV=1.0,
                                    extra_spike_penalty=4321.0)
    assert result.value == 4321.0


def test_depolarizing_requires_matching_axon_and_soma_counts():
    observed = trace("depolarizing_step")
    soma = observed.voltage_mV.copy()
    soma[600:603] = [-10.0, 30.0, -10.0]
    axon = soma.copy()
    axon[800:803] = [-10.0, 30.0, -10.0]
    result = depolarizing_objective(
        [observed], [SimulationOutput(observed.time_ms, soma, axon)],
        prominence_mV=1.0, extra_spike_penalty=4321.0,
    )
    assert result.value == 4321.0
    transmission = result.details["traces"][0]["transmission"]
    assert transmission["axon_count"] == 2
    assert transmission["soma_count"] == 1


def test_depolarizing_bursting_train_uses_extra_spike_penalty():
    observed = trace("depolarizing_step")
    observed_voltage = observed.voltage_mV.copy()
    for center in (600, 800, 1000):
        observed_voltage[center:center + 3] = [-10.0, 30.0, -10.0]
    observed = Trace(observed.cell, observed.trace, observed.protocol,
                     observed.time_ms, observed_voltage, observed.current_nA,
                     observed.epoch_start_ms, observed.epoch_stop_ms)
    simulated_voltage = observed_voltage.copy()
    for center in (600, 650, 1000):
        simulated_voltage[center:center + 3] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, simulated_voltage)
    result = depolarizing_objective([observed], [simulated], prominence_mV=1.0,
                                    extra_spike_penalty=4321.0)
    assert result.value == 4321.0


def test_depolarizing_firing_rate_match_is_zero():
    observed = trace("depolarizing_step")
    voltage = observed.voltage_mV.copy()
    voltage[600:603] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, voltage)
    result = depolarizing_objective([observed], [simulated], prominence_mV=1.0)
    rate = result.details["traces"][0]["firing_rate_hz"]
    assert rate["experimental"] == 0.0
    assert rate["simulated"] == 10.0
    assert rate["loss"] > 0.0


def test_depolarizing_firing_rate_component_matches_equal_rates():
    observed = trace("depolarizing_step")
    voltage = observed.voltage_mV.copy()
    voltage[600:603] = [-10.0, 30.0, -10.0]
    observed = Trace(observed.cell, observed.trace, observed.protocol, observed.time_ms,
                     voltage, observed.current_nA, observed.epoch_start_ms,
                     observed.epoch_stop_ms)
    simulated = SimulationOutput(observed.time_ms, voltage.copy())
    result = depolarizing_objective([observed], [simulated], prominence_mV=1.0)
    rate = result.details["traces"][0]["firing_rate_hz"]
    assert rate["experimental"] == 10.0
    assert rate["simulated"] == 10.0
    assert rate["loss"] == 0.0


def test_depolarizing_spike_symmetry_penalizes_rise_fall_mismatch():
    observed = trace("depolarizing_step")
    center = 600
    experimental = observed.voltage_mV.copy()
    experimental[center - 3:center + 5] = [-45.0, -25.0, 10.0, 35.0,
                                             5.0, -10.0, -30.0, -45.0]
    simulated = observed.voltage_mV.copy()
    simulated[center - 3:center + 5] = [-45.0, -25.0, 10.0, 35.0,
                                          10.0, -25.0, -45.0, -45.0]
    observed = Trace(observed.cell, observed.trace, observed.protocol,
                     observed.time_ms, experimental, observed.current_nA,
                     observed.epoch_start_ms, observed.epoch_stop_ms)
    result = depolarizing_objective(
        [observed], [SimulationOutput(observed.time_ms, simulated)],
        prominence_mV=1.0, unmatched_spike_penalty=0.0,
        weights={"trajectory": 0.0, "spike_shape": 0.0,
                 "spike_symmetry": 1.0, "plateau": 0.0,
                 "return_baseline": 0.0, "firing_rate": 0.0},
    )
    symmetry = result.details["traces"][0]["spike_symmetry"]
    assert symmetry["matched"] == 1
    assert symmetry["loss"] > 0.0


def test_depolarizing_spike_height_penalizes_amplitude_mismatch():
    observed = trace("depolarizing_step")
    center = 600
    experimental = observed.voltage_mV.copy()
    experimental[center - 2:center + 4] = [-25.0, 10.0, 35.0, 5.0, -10.0, -30.0]
    simulated = observed.voltage_mV.copy()
    simulated[center - 2:center + 4] = [-25.0, 10.0, 20.0, 5.0, -10.0, -30.0]
    observed = Trace(observed.cell, observed.trace, observed.protocol,
                     observed.time_ms, experimental, observed.current_nA,
                     observed.epoch_start_ms, observed.epoch_stop_ms)
    result = depolarizing_objective(
        [observed], [SimulationOutput(observed.time_ms, simulated)],
        prominence_mV=1.0, unmatched_spike_penalty=0.0,
        weights={"trajectory": 0.0, "spike_shape": 0.0,
                 "spike_symmetry": 0.0, "spike_height": 1.0,
                 "plateau": 0.0, "return_baseline": 0.0,
                 "firing_rate": 0.0},
    )
    height = result.details["traces"][0]["spike_height_mV"]
    assert height["experimental"] == [105.0]
    assert height["simulated"] == [90.0]
    assert height["loss"] > 0.0


def test_spike_height_loss_does_not_saturate_for_large_errors():
    moderate_error = _spike_height_loss(
        np.asarray([90.0]), np.asarray([105.0]), sigma_mV=5.0
    )
    large_error = _spike_height_loss(
        np.asarray([50.0]), np.asarray([105.0]), sigma_mV=5.0
    )
    assert large_error > moderate_error
    assert large_error > 1.0


def test_precomputed_objective_context_preserves_loss():
    observed = trace("depolarizing_step")
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV.copy())
    uncached = depolarizing_objective([observed], [simulated])
    context = build_objective_context([observed], stage="depolarizing")
    cached = depolarizing_objective([observed], [simulated], context=context)
    assert cached.value == uncached.value
    assert cached.details == uncached.details

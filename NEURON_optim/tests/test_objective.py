import numpy as np

from neuron_optim.data import Trace
from neuron_optim.objective import (
    SimulationOutput,
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


def test_hyper_exponential_loss_is_zero_for_matching_trace():
    observed = trace()
    simulated = SimulationOutput(observed.time_ms, observed.voltage_mV.copy())
    result = hyperpolarizing_objective([observed], [simulated])
    assert result.value == 0.0


def test_hyper_spike_penalty_overrides_voltage_loss():
    observed = trace()
    voltage = observed.voltage_mV.copy()
    voltage[1050:1053] = [-10.0, 30.0, -10.0]
    simulated = SimulationOutput(observed.time_ms, voltage)
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

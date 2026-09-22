import numpy as np
from pathlib import Path

from neuron_optim.config import RunConfig
from neuron_optim.data import Trace, crop_trace
from neuron_optim.objective import SimulationOutput
from neuron_optim.stages import (
    _payload_to_simulations,
    _save_generation_simulations,
    _simulation_payload,
)


def test_simulation_payload_round_trips_and_is_persisted(tmp_path):
    time = np.arange(0.0, 1.0, 0.1)
    outputs = [[SimulationOutput(time, time - 1.0)], None]
    payload = [_simulation_payload(outputs[0]), None]

    restored = _payload_to_simulations(payload[0])
    np.testing.assert_allclose(restored[0].time_ms, time)
    np.testing.assert_allclose(restored[0].voltage_mV, time - 1.0)

    population = np.asarray([[0.1], [0.2]])
    losses = np.asarray([1.0, 2.0])
    _save_generation_simulations(tmp_path, 3, population, losses, payload)

    with np.load(tmp_path / "simulations_generation_0003.npz") as saved:
        np.testing.assert_allclose(saved["population"], population)
        np.testing.assert_allclose(saved["losses"], losses)
        np.testing.assert_array_equal(saved["valid"], [True, False])
        np.testing.assert_allclose(
            saved["candidate_0000_trace_0000_voltage_mV"], time - 1.0
        )


def test_hyper_fitness_crop_preserves_simulation_time_origin():
    time = np.arange(0.0, 1000.1, 0.1)
    trace = Trace("cell", "v75ctrl", "hyperpolarizing_pulse", time,
                  np.zeros(time.size), np.zeros(time.size), 500.0, 800.0)
    cropped = crop_trace(trace, start_ms=trace.epoch_start_ms - 100.0)
    assert trace.time_ms[0] == 0.0
    assert cropped.time_ms[0] == 400.0
    assert cropped.epoch_start_ms == 500.0
    assert cropped.epoch_stop_ms == 800.0
    assert cropped.time_ms[-1] == trace.time_ms[-1]


def test_hyper_fitness_window_config_defaults_to_100_ms():
    config = RunConfig({"data": {"root": ".", "cell": "cell"}},
                       Path("config.yaml"))
    assert config.hyperpolarizing_fitness_pre_ms == 100.0

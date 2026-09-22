import numpy as np

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

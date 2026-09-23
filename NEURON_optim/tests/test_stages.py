import numpy as np
from pathlib import Path

from neuron_optim.config import RunConfig
from neuron_optim.data import Trace, crop_trace
from neuron_optim.objective import SimulationOutput
from neuron_optim.stages import (
    _payload_to_simulations,
    _save_generation_simulations,
    _simulation_payload,
    make_basins,
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


def test_hyperpolarizing_trace_selection_defaults_to_first_trace():
    config = RunConfig({
        "data": {"root": ".", "cell": "cell",
                 "trace_names": ["first", "second", "third", "fourth"]},
    }, Path("config.yaml"))
    assert config.hyperpolarizing_trace_names == ("first",)


def test_hyperpolarizing_trace_selection_uses_configured_ordinal():
    config = RunConfig({
        "data": {"root": ".", "cell": "cell",
                 "trace_names": ["first", "second", "third", "fourth"],
                 "hyperpolarizing_trace_index": 2},
    }, Path("config.yaml"))
    assert config.hyperpolarizing_trace_names == ("third",)


def test_make_basins_scans_all_stage1_generations(tmp_path):
    config = RunConfig({
        "parameters": {"include": ["Epas"], "exclude": []},
        "stage1": {"seeds": [0, 1], "generations": 2,
                   "basin_quota_per_seed": 1},
    }, Path("config.yaml"))
    stage1_run = tmp_path / "stage1_hyper"
    for seed in (0, 1):
        seed_dir = stage1_run / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True)
        for generation in (1, 2):
            population = np.asarray([[
                0.1 * (seed + 1) + 0.01 * generation
            ], [0.8 + 0.01 * seed + 0.001 * generation]])
            losses = np.asarray([float(generation + seed), 10.0 + generation + seed])
            np.savez_compressed(
                seed_dir / f"population_generation_{generation:04d}.npz",
                population=population, losses=losses,
            )

    basins = make_basins(config, stage1_run)

    assert len(basins) == 8
    assert {record["source_generation"] for record in basins} == {1, 2}
    assert {record["source_seed"] for record in basins} == {0, 1}

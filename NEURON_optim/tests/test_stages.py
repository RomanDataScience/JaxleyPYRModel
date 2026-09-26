import numpy as np
from pathlib import Path

from neuron_optim.config import RunConfig
from neuron_optim.parameters import BOUNDS
from neuron_optim.data import Trace, crop_trace
from neuron_optim.objective import SimulationOutput
from neuron_optim.stages import (
    _payload_to_simulations,
    _save_generation_simulations,
    _simulation_payload,
    _stage2_tasks,
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


def test_stage_parameter_space_applies_configured_bounds():
    config = RunConfig({
        "parameters": {"include": ["gna", "gnaaxon", "nav16_O1I1b2"]},
        "stage2": {
            "parameter_names": ["gna", "gnaaxon", "nav16_O1I1b2"],
            "lower_bounds": {"gna": 0.10, "gnaaxon": 0.5},
            "upper_bounds": {"nav16_O1I1b2": 250.0},
        },
    }, Path("config.yaml"))
    space = config.stage_parameter_space("stage2")
    assert space.lower.tolist() == [0.10, 0.5, 1.0]
    assert space.upper.tolist() == [0.2, 2.0, 250.0]


def test_stage2_keeps_all_parameters_variable_in_local_basin_space(tmp_path):
    config = RunConfig({
        "stage2": {"seeds": [0]},
    }, Path("config.yaml"))
    basin = {
        "basin_id": "b000",
        "physical": config.parameters.reference.tolist(),
    }

    tasks = _stage2_tasks(config, tmp_path, [basin])

    assert len(tasks) == 1
    _, _, _, _, _, _, space, fixed_values = tasks[0]
    assert len(space.keys) == 56
    assert "nav16_I1O1b1" in space.keys
    assert fixed_values is None
    index = space.keys.index("nav16_I1O1b1")
    np.testing.assert_allclose(
        [space.lower[index], space.upper[index]], [0.00325, 0.00675]
    )


def test_stage2_global_bound_parameters_ignore_the_local_window(tmp_path):
    config = RunConfig({
        "stage2": {"seeds": [0], "global_bound_parameters": ["soma_km"]},
    }, Path("config.yaml"))
    basin = {
        "basin_id": "b000",
        "physical": config.parameters.reference.tolist(),
    }

    _, _, _, _, _, _, space, _ = _stage2_tasks(config, tmp_path, [basin])[0]
    km = space.keys.index("soma_km")
    assert [space.lower[km], space.upper[km]] == list(BOUNDS["soma_km"])
    # Parameters not listed keep the local window (default gna12 = 0.03).
    na12 = space.keys.index("gna12")
    np.testing.assert_allclose([space.lower[na12], space.upper[na12]], [0.0195, 0.0405])


def test_stage2_local_relative_width_is_configurable(tmp_path):
    config = RunConfig({
        "stage2": {"seeds": [0], "local_relative_width": 0.15},
    }, Path("config.yaml"))
    basin = {
        "basin_id": "b000",
        "physical": config.parameters.reference.tolist(),
    }

    _, _, _, _, _, _, space, _ = _stage2_tasks(config, tmp_path, [basin])[0]
    index = space.keys.index("nav16_I1O1b1")
    np.testing.assert_allclose(
        [space.lower[index], space.upper[index]], [0.00425, 0.00575]
    )


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

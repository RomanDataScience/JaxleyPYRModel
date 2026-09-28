from pathlib import Path

import numpy as np

from neuron_optim import stages
from neuron_optim.config import RunConfig
from neuron_optim.data import Trace
from neuron_optim.objective import SimulationOutput, subthreshold_objective
from neuron_optim.parameters import BOUNDS, SODIUM_CONDUCTANCES, SODIUM_KINETIC


def _step_trace(voltage=None) -> Trace:
    time = np.arange(0.0, 400.0, 0.1)
    if voltage is None:
        voltage = np.full(time.size, -65.0)
        voltage[(time >= 100.0) & (time <= 300.0)] = -50.0
        spike = (time >= 150.0) & (time < 151.0)
        voltage[spike] = 30.0
    current = np.zeros(time.size)
    current[(time >= 100.0) & (time <= 300.0)] = 0.3
    return Trace(cell="c", trace="t", protocol="depolarizing_step", time_ms=time,
                 voltage_mV=voltage, current_nA=current,
                 epoch_start_ms=100.0, epoch_stop_ms=300.0)


def test_subthreshold_objective_ignores_recorded_spikes_and_scores_offsets():
    trace = _step_trace()
    spike_free = trace.voltage_mV.copy()
    spike_free[spike_free > 0.0] = -50.0
    perfect = subthreshold_objective([trace], [SimulationOutput(trace.time_ms, spike_free)])
    assert perfect.value < 1e-9
    assert set(perfect.details["traces"][0]["regions"]) == {"baseline", "onset", "plateau", "recovery"}

    low = subthreshold_objective([trace], [SimulationOutput(trace.time_ms, spike_free - 8.0)])
    plateau = low.details["traces"][0]["regions"]["plateau"]
    assert plateau["offset_mV"] == -8.0
    assert low.value > perfect.value

    spiking = spike_free.copy()
    spiking[(trace.time_ms >= 200.0) & (trace.time_ms < 201.0)] = 30.0
    penalized = subthreshold_objective([trace], [SimulationOutput(trace.time_ms, spiking)],
                                       simulated_spike_penalty=10.0)
    assert penalized.value >= 10.0


def test_two_stage_pipeline_builds_hyper_then_sodium_free_then_full_spaces(tmp_path, monkeypatch):
    config = RunConfig({
        "data": {},
        "depol_hyper": {"seeds": [0], "parameter_names": ["RmSoma", "soma_kap", "gna"]},
        "depol_sub": {"seeds": [0, 1], "local_relative_width": 0.2,
                      "fixed_sodium": {"gna": 0.08, "gnadend": 0.0225}},
        "depol_full": {"seeds": [0], "local_relative_width": 0.35,
                       "start_values": {"gna12": 0.06, "na12_shift": 8.0}},
    }, Path("config.yaml"))
    calls = []

    def fake_run_studies(_config, tasks):
        calls.append(tasks)
        space = tasks[0][6]
        best = {key: float(value) for key, value in zip(space.keys, space.reference)}
        if tasks[0][1] == stages.HYPER_PRE_STAGE:
            best["RmSoma"] = 50_000.0
            best["soma_kap"] = 0.02
            return [{"loss": 1.0, "physical_by_name": best}]
        if tasks[0][1] == stages.SUBTHRESHOLD_STAGE:
            best["RmSoma"] = 100_000.0
            return [{"loss": 2.0, "physical_by_name": best},
                    {"loss": 1.0, "physical_by_name": best}]
        return [{"loss": 0.5, "physical_by_name": {}}]

    monkeypatch.setattr(stages, "_run_studies", fake_run_studies)
    stages.run_depolarizing_two_stage(config, tmp_path)

    hyper_tasks, sub_tasks, full_tasks = calls
    sodium = set(SODIUM_CONDUCTANCES) | set(SODIUM_KINETIC)
    hyper_space, hyper_fixed = hyper_tasks[0][6], hyper_tasks[0][7]
    assert hyper_tasks[0][1] == stages.HYPER_PRE_STAGE
    assert not set(hyper_space.keys) & sodium
    assert hyper_space.keys == ("RmSoma", "soma_kap")  # sodium keys never fitted here
    # Nav1.6 (soma, dendrites) kept at fixed values; axonal Nav and na12 off.
    expected_sodium = {"gna": 0.08, "gna12": 0.0, "gnaaxon": 0.0, "gnadend": 0.0225}
    assert hyper_fixed == expected_sodium
    assert (tmp_path / "stage0_hyper" / "best.json").exists()

    sub_space, sub_fixed = sub_tasks[0][6], sub_tasks[0][7]
    assert not set(sub_space.keys) & sodium
    assert sub_fixed == expected_sodium
    assert (tmp_path / "stageA_subthreshold" / "best.json").exists()
    # Passive keys are held near the hyperpolarizing fit; others stay global.
    rm = sub_space.keys.index("RmSoma")
    np.testing.assert_allclose([sub_space.lower[rm], sub_space.upper[rm]], [40_000.0, 60_000.0])
    kap = sub_space.keys.index("soma_kap")
    assert [sub_space.lower[kap], sub_space.upper[kap]] == list(BOUNDS["soma_kap"])
    # Stage A starts from the stage-0 best (before the per-seed perturbation).
    start = sub_space.physical(sub_tasks[0][4])
    assert abs(start[kap] - 0.02) <= 0.15 * (BOUNDS["soma_kap"][1] - BOUNDS["soma_kap"][0]) + 1e-12

    full_space, full_fixed = full_tasks[0][6], full_tasks[0][7]
    assert full_fixed is None
    assert len(full_space.keys) == len(config.parameters.keys)
    rm = full_space.keys.index("RmSoma")
    np.testing.assert_allclose([full_space.lower[rm], full_space.upper[rm]], [65_000.0, 135_000.0])
    for key in SODIUM_CONDUCTANCES + ("nax_ar2", "na12_shift"):
        j = full_space.keys.index(key)
        assert [full_space.lower[j], full_space.upper[j]] == list(BOUNDS[key])
    # start_values seed stage B (within the ±0.15 normalized seed perturbation).
    full_start = full_space.physical(full_tasks[0][4])
    for key, value in {"gna12": 0.06, "na12_shift": 8.0}.items():
        j = full_space.keys.index(key)
        assert abs(full_start[j] - value) <= 0.15 * (BOUNDS[key][1] - BOUNDS[key][0]) + 1e-12


def test_hyper_pre_stage_uses_first_hyperpolarizing_pulse_and_hyper_objective(tmp_path):
    config = RunConfig({"data": {"trace_names": ["a", "b", "c", "d"]},
                        "depol_hyper": {"sigma_hyper_mV": 0.5}}, tmp_path / "config.yaml")
    assert config.hyperpolarizing_trace_names == ("a",)
    assert stages._objective_function(stages.HYPER_PRE_STAGE) is stages.hyperpolarizing_objective
    assert stages._study_section_name(stages.HYPER_PRE_STAGE) == stages.HYPER_PRE_STAGE
    assert stages._objective_options(config.raw, stages.HYPER_PRE_STAGE)["sigma_mV"] == 0.5


def test_two_stage_config_needs_only_depolarizing_sections(tmp_path):
    section = {"seeds": [0], "generations": 1, "population_size": 4}
    config = RunConfig({
        "data": {"root": str(tmp_path)},
        "runtime": {"pipeline": "depolarizing_two_stage"},
        "stage2": dict(section), "depol_hyper": dict(section),
        "depol_sub": dict(section), "depol_full": dict(section),
    }, tmp_path / "config.yaml")
    assert config.validate() == []
    config.raw["depol_full"]["local_relative_width"] = 0.0
    assert "depol_full.local_relative_width must be > 0" in config.validate()
    del config.raw["depol_hyper"]
    assert "depol_hyper.generations must be >= 1" in config.validate()


def test_kd_patch_makes_midpoints_per_section(tmp_path):
    from neuron_optim.mechanisms import _patch_sources, _repo_root

    _patch_sources(_repo_root() / "channels_converted" / "mod", tmp_path)
    text = (tmp_path / "kd.mod").read_text()
    assert "RANGE gk, gbar, i, deactivation_tau_scale, vhalfm, vhalfh" in text


def test_fixed_sodium_rejects_non_sodium_keys(tmp_path, monkeypatch):
    import pytest
    config = RunConfig({"data": {}, "depol_sub": {"fixed_sodium": {"soma_kap": 0.1}}},
                       Path("config.yaml"))
    monkeypatch.setattr(stages, "_run_studies", lambda *_: pytest.fail("ran a study"))
    with pytest.raises(ValueError, match="fixed_sodium"):
        stages.run_depolarizing_two_stage(config, tmp_path)


def test_parameters_fixed_is_excluded_from_fit_and_validated(tmp_path):
    config = RunConfig({"data": {"root": str(tmp_path)},
                        "parameters": {"fixed": {"soma_h_scale": 0.0}}},
                       tmp_path / "config.yaml")
    assert config.fixed_parameters == {"soma_h_scale": 0.0}
    assert "soma_h_scale" not in config.parameters.keys
    assert "soma_hbar" in config.parameters.keys
    config.raw["parameters"]["fixed"] = {"soma_h_scale": 2.0, "not_a_key": 1.0}
    errors = config.validate()
    assert any("soma_h_scale=2.0 is outside" in e for e in errors)
    assert any("unknown parameter: not_a_key" in e for e in errors)


def test_run_study_applies_config_fixed_values(tmp_path, monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_manifest(_config, _stage, _seed, _space, _run_dir, _basin, _mean, fixed):
        seen.update(fixed)
        raise Stop

    monkeypatch.setattr(stages, "_load_traces", lambda *_: [])
    monkeypatch.setattr(stages, "_fitness_traces", lambda *_: [])
    monkeypatch.setattr(stages, "_study_manifest", fake_manifest)
    config = RunConfig({"data": {}, "parameters": {"fixed": {"soma_h_scale": 0.0}},
                        "stage2": {}}, tmp_path / "config.yaml")
    import pytest
    with pytest.raises(Stop):
        stages.run_study(config, stage="depol_full", seed=0, run_dir=tmp_path / "s",
                         fixed_values={"gna": 0.0})
    assert seen == {"soma_h_scale": 0.0, "gna": 0.0}


def test_hyper_full_pipeline_runs_stage0_then_full_from_stage0_best(tmp_path, monkeypatch):
    config = RunConfig({
        "data": {},
        "runtime": {"pipeline": "hyper_full"},
        "depol_hyper": {"seeds": [0], "parameter_names": ["RmSoma", "soma_kap"],
                        "fixed_sodium": {"gna": 0.08}},
        "depol_full": {"seeds": [0], "local_relative_width": 0.35,
                       "start_values": {"gna12": 0.06}},
    }, Path("config.yaml"))
    calls = []

    def fake_run_studies(_config, tasks):
        calls.append(tasks)
        if tasks[0][1] == stages.HYPER_PRE_STAGE:
            return [{"loss": 1.0, "physical_by_name": {"RmSoma": 100_000.0, "soma_kap": 0.02}}]
        return [{"loss": 0.5, "physical_by_name": {}}]

    monkeypatch.setattr(stages, "_run_studies", fake_run_studies)
    stages.run_pipeline(config, tmp_path)

    assert [tasks[0][1] for tasks in calls] == [stages.HYPER_PRE_STAGE, stages.FULL_DEPOL_STAGE]
    hyper_tasks, full_tasks = calls
    assert hyper_tasks[0][6].keys == ("RmSoma", "soma_kap")
    assert hyper_tasks[0][7] == {"gna": 0.08, "gna12": 0.0, "gnaaxon": 0.0, "gnadend": 0.0}
    assert not (tmp_path / "stageA_subthreshold").exists()

    full_space, full_fixed = full_tasks[0][6], full_tasks[0][7]
    assert full_fixed is None
    # Stage-0 parameters held within ±35% of their stage-0 value ...
    rm = full_space.keys.index("RmSoma")
    np.testing.assert_allclose([full_space.lower[rm], full_space.upper[rm]], [65_000.0, 135_000.0])
    kap = full_space.keys.index("soma_kap")
    np.testing.assert_allclose([full_space.lower[kap], full_space.upper[kap]], [0.013, 0.027])
    # ... everything else, sodium included, on global bounds.
    for key in ("gna", "gna12", "gnaaxon", "soma_hbar", "gkdrsoma"):
        j = full_space.keys.index(key)
        assert [full_space.lower[j], full_space.upper[j]] == list(BOUNDS[key])


def test_fixed_sodium_prefers_depol_hyper_then_depol_sub():
    both = RunConfig({"depol_hyper": {"fixed_sodium": {"gna": 0.05}},
                      "depol_sub": {"fixed_sodium": {"gna": 0.08}}}, Path("c.yaml"))
    assert stages._fixed_sodium(both)["gna"] == 0.05
    sub_only = RunConfig({"depol_sub": {"fixed_sodium": {"gnadend": 0.02}}}, Path("c.yaml"))
    assert stages._fixed_sodium(sub_only) == {"gna": 0.0, "gna12": 0.0, "gnaaxon": 0.0,
                                              "gnadend": 0.02}
    assert set(stages._fixed_sodium(RunConfig({}, Path("c.yaml"))).values()) == {0.0}


def test_hyper_full_config_needs_only_its_sections(tmp_path):
    section = {"seeds": [0], "generations": 1, "population_size": 4}
    config = RunConfig({
        "data": {"root": str(tmp_path)},
        "runtime": {"pipeline": "hyper_full"},
        "stage2": dict(section), "depol_hyper": dict(section), "depol_full": dict(section),
    }, tmp_path / "config.yaml")
    assert config.validate() == []
    del config.raw["depol_hyper"]
    assert "depol_hyper.generations must be >= 1" in config.validate()


def test_hyper_stage_start_values_and_tightened_bounds(tmp_path, monkeypatch):
    config = RunConfig({
        "data": {},
        "depol_hyper": {"seeds": [0], "parameter_names": ["soma_hbar", "RmSoma"],
                        "upper_bounds": {"soma_hbar": 6e-6},
                        "start_values": {"soma_hbar": 3e-6}},
        "depol_full": {"seeds": [0]},
    }, Path("config.yaml"))
    calls = []

    def fake_run_studies(_config, tasks):
        calls.append(tasks)
        return [{"loss": 1.0, "physical_by_name": {"soma_hbar": 3e-6, "RmSoma": 1e5}}]

    monkeypatch.setattr(stages, "_run_studies", fake_run_studies)
    stages.run_hyper_full(config, tmp_path)
    space, mean = calls[0][0][6], calls[0][0][4]
    j = space.keys.index("soma_hbar")
    assert space.upper[j] == 6e-6
    # Seeded start is 3e-6 perturbed by at most 0.15 of the (0, 6e-6) range.
    assert abs(space.physical(mean)[j] - 3e-6) <= 0.15 * 6e-6 + 1e-18

    bad = RunConfig({"data": {}, "depol_hyper": {"parameter_names": ["RmSoma"],
                                                 "start_values": {"soma_hbar": 3e-6}}},
                    Path("config.yaml"))
    import pytest
    with pytest.raises(ValueError, match="start_values"):
        stages.run_hyper_full(bad, tmp_path / "bad")

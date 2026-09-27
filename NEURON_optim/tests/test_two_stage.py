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


def test_two_stage_pipeline_builds_sodium_free_then_full_spaces(tmp_path, monkeypatch):
    config = RunConfig({
        "depol_sub": {"seeds": [0, 1]},
        "depol_full": {"seeds": [0], "local_relative_width": 0.35},
    }, Path("config.yaml"))
    calls = []

    def fake_run_studies(_config, tasks):
        calls.append(tasks)
        space = tasks[0][6]
        if tasks[0][1] == stages.SUBTHRESHOLD_STAGE:
            best = {key: float(value) for key, value in zip(space.keys, space.reference)}
            best["RmSoma"] = 100_000.0
            return [{"loss": 2.0, "physical_by_name": best},
                    {"loss": 1.0, "physical_by_name": best}]
        return [{"loss": 0.5, "physical_by_name": {}}]

    monkeypatch.setattr(stages, "_run_studies", fake_run_studies)
    stages.run_depolarizing_two_stage(config, tmp_path)

    sub_tasks, full_tasks = calls
    sub_space, sub_fixed = sub_tasks[0][6], sub_tasks[0][7]
    assert not set(sub_space.keys) & (set(SODIUM_CONDUCTANCES) | set(SODIUM_KINETIC))
    assert sub_fixed == {key: 0.0 for key in SODIUM_CONDUCTANCES}
    assert (tmp_path / "stageA_subthreshold" / "best.json").exists()

    full_space, full_fixed = full_tasks[0][6], full_tasks[0][7]
    assert full_fixed is None
    assert len(full_space.keys) == len(config.parameters.keys)
    rm = full_space.keys.index("RmSoma")
    np.testing.assert_allclose([full_space.lower[rm], full_space.upper[rm]], [65_000.0, 135_000.0])
    for key in SODIUM_CONDUCTANCES + ("nax_ar2", "na12_shift"):
        j = full_space.keys.index(key)
        assert [full_space.lower[j], full_space.upper[j]] == list(BOUNDS[key])


def test_two_stage_config_needs_only_depolarizing_sections(tmp_path):
    section = {"seeds": [0], "generations": 1, "population_size": 4}
    config = RunConfig({
        "data": {"root": str(tmp_path)},
        "runtime": {"pipeline": "depolarizing_two_stage"},
        "stage2": dict(section), "depol_sub": dict(section), "depol_full": dict(section),
    }, tmp_path / "config.yaml")
    assert config.validate() == []
    config.raw["depol_full"]["local_relative_width"] = 0.0
    assert "depol_full.local_relative_width must be > 0" in config.validate()


def test_kd_patch_makes_midpoints_per_section(tmp_path):
    from neuron_optim.mechanisms import _patch_sources, _repo_root

    _patch_sources(_repo_root() / "channels_converted" / "mod", tmp_path)
    text = (tmp_path / "kd.mod").read_text()
    assert "RANGE gk, gbar, i, deactivation_tau_scale, vhalfm, vhalfh" in text

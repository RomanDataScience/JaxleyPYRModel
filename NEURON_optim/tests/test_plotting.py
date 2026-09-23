import json

import numpy as np

from neuron_optim.data import Trace
from neuron_optim.objective import SimulationOutput
from neuron_optim.parameters import make_parameter_space
from neuron_optim.plotting import (
    plot_depolarizing_step_generation,
    plot_generation,
)


def test_generation_plot_writes_ranked_candidate(tmp_path):
    time = np.arange(0.0, 10.0, 0.1)
    trace = Trace("cell", "v75ctrl", "hyperpolarizing_pulse", time,
                  np.full(time.size, -70.0), np.zeros(time.size), 4.0, 6.0)
    space = make_parameter_space(include=["Epas"])
    population = np.asarray([[0.5], [0.6]])
    losses = np.asarray([0.2, 0.1])
    simulations = [[SimulationOutput(time, np.full(time.size, -70.0))] for _ in population]
    plot_generation(output_dir=tmp_path, generation=1, stage="hyper", traces=[trace],
                    population=population, losses=losses, simulations=simulations,
                    space=space, top_k=2, population_indices=np.asarray([7, 3]))
    assert (tmp_path / "generation_0001" / "candidates.png").exists()
    assert (tmp_path / "generation_0001" / "top_candidates.json").exists()
    metadata = json.loads((tmp_path / "generation_0001" / "top_candidates.json").read_text())
    assert [item["population_index"] for item in metadata] == [3, 7]


def test_depolarizing_step_plot_uses_short_window(tmp_path):
    time = np.arange(0.0, 250.0, 0.1)
    trace = Trace("cell", "v75ctrl", "depolarizing_step", time,
                  np.full(time.size, -70.0), np.zeros(time.size), 100.0, 150.0)
    space = make_parameter_space(include=["Epas"])
    population = np.asarray([[0.5], [0.6]])
    losses = np.asarray([0.2, 0.1])
    simulations = [[SimulationOutput(time, np.full(time.size, -70.0))] for _ in population]

    plot_depolarizing_step_generation(
        output_dir=tmp_path, generation=1, traces=[trace],
        population=population, losses=losses, simulations=simulations,
        space=space, top_k=2, population_indices=np.asarray([7, 3]),
    )

    generation_dir = tmp_path / "generation_0001"
    assert (generation_dir / "depolarizing_step_candidates.png").exists()
    metadata = json.loads(
        (generation_dir / "depolarizing_step_top_candidates.json").read_text()
    )
    assert [item["population_index"] for item in metadata] == [3, 7]

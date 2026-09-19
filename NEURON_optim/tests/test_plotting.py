import numpy as np

from neuron_optim.data import Trace
from neuron_optim.objective import SimulationOutput
from neuron_optim.parameters import make_parameter_space
from neuron_optim.plotting import plot_generation


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
                    space=space, top_k=2)
    assert (tmp_path / "generation_0001" / "rank_01.png").exists()
    assert (tmp_path / "generation_0001" / "top_candidates.json").exists()

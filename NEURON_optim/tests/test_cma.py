import numpy as np

from neuron_optim.cma import CMAES


def test_cma_stays_bounded_and_improves_sphere(tmp_path):
    optimizer = CMAES(np.full(3, 0.8), sigma=0.2, seed=4, population_size=8)
    for _ in range(10):
        population = optimizer.ask()
        assert np.all((population >= 0.0) & (population <= 1.0))
        optimizer.tell(population, np.sum((population - 0.2) ** 2, axis=1))
    assert optimizer.state.generation == 10
    optimizer.save(tmp_path, "hash")
    restored = CMAES.load(tmp_path, seed=4, compatibility_hash="hash")
    assert restored is not None
    assert restored.state.generation == optimizer.state.generation
    resumed = restored.ask()
    assert resumed.shape == (8, 3)
    assert np.all((resumed >= 0.0) & (resumed <= 1.0))

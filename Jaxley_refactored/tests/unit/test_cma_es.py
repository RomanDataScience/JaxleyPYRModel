import json

import numpy as np

from jaxley_refactored.fitting.global_search import CMAES
from jaxley_refactored.fitting.global_search.checkpoints import CMACheckpoint


def test_cma_es_improves_bounded_sphere():
    target = np.full(4, 0.2)
    optimizer = CMAES(
        np.full(4, 0.8), sigma=0.2, seed=7, population_size=10
    )
    initial = np.sum((optimizer.state.mean - target) ** 2)

    for _ in range(35):
        population = optimizer.ask()
        optimizer.tell(population, np.sum((population - target) ** 2, axis=1))

    assert np.sum((optimizer.state.mean - target) ** 2) < initial * 1e-3
    assert np.all((optimizer.state.mean >= 0.0) & (optimizer.state.mean <= 1.0))


def test_cma_checkpoint_reproduces_next_population(tmp_path):
    optimizer = CMAES(
        np.full(3, 0.5), sigma=0.1, seed=11, population_size=8
    )
    population = optimizer.ask()
    optimizer.tell(population, np.sum(population**2, axis=1))
    checkpoint = CMACheckpoint(tmp_path, "compatible")
    checkpoint.save(optimizer)

    expected = optimizer.ask()
    restored = checkpoint.load(seed=11)

    assert restored is not None
    np.testing.assert_array_equal(restored.ask(), expected)


def test_cma_checkpoint_is_non_pickle(tmp_path):
    optimizer = CMAES(np.full(2, 0.5), sigma=0.1, seed=3, population_size=6)
    checkpoint = CMACheckpoint(tmp_path, "compatible")
    checkpoint.save(optimizer)

    with (tmp_path / "latest.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["version"] == 1
    with np.load(tmp_path / "latest.npz", allow_pickle=False) as arrays:
        assert "covariance" in arrays.files

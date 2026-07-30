import json

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("population_size", "parent_fraction", "expected_mu"),
    (
        (40, 0.30, 12),
        (10, 0.30, 3),
        (7, 0.30, 2),
        (4, 0.30, 1),
        (7, 0.01, 1),
        (7, 1.0, 7),
    ),
)
def test_cma_parent_fraction_selects_requested_fraction(
    population_size, parent_fraction, expected_mu
):
    optimizer = CMAES(
        np.full(2, 0.5),
        sigma=0.1,
        seed=3,
        population_size=population_size,
        parent_fraction=parent_fraction,
    )

    assert optimizer.mu == expected_mu
    assert optimizer.weights.shape == (expected_mu,)
    assert np.isclose(optimizer.weights.sum(), 1.0)


@pytest.mark.parametrize(
    "parent_fraction", [0.0, -0.1, 1.1, np.nan, np.inf]
)
def test_cma_parent_fraction_is_validated(parent_fraction):
    with pytest.raises(ValueError, match="parent_fraction"):
        CMAES(
            np.full(2, 0.5),
            sigma=0.1,
            seed=3,
            population_size=10,
            parent_fraction=parent_fraction,
        )


def test_cma_checkpoint_reproduces_next_population(tmp_path):
    optimizer = CMAES(
        np.full(3, 0.5),
        sigma=0.1,
        seed=11,
        population_size=8,
        parent_fraction=0.30,
    )
    population = optimizer.ask()
    optimizer.tell(population, np.sum(population**2, axis=1))
    checkpoint = CMACheckpoint(tmp_path, "compatible")
    checkpoint.save(optimizer)

    expected = optimizer.ask()
    restored = checkpoint.load(seed=11)

    assert restored is not None
    assert restored.parent_fraction == 0.30
    assert restored.mu == 2
    restored_population = restored.ask()
    np.testing.assert_array_equal(restored_population, expected)

    losses = np.sum(expected**2, axis=1)
    optimizer.tell(expected, losses)
    restored.tell(restored_population, losses)
    for name, expected_array in optimizer.arrays().items():
        np.testing.assert_allclose(restored.arrays()[name], expected_array)
    np.testing.assert_array_equal(restored.ask(), optimizer.ask())


def test_cma_checkpoint_is_non_pickle(tmp_path):
    optimizer = CMAES(np.full(2, 0.5), sigma=0.1, seed=3, population_size=6)
    checkpoint = CMACheckpoint(tmp_path, "compatible")
    checkpoint.save(optimizer)

    with (tmp_path / "latest.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["version"] == 2
    assert metadata["parent_fraction"] == 0.5
    assert metadata["mu"] == 3
    with np.load(tmp_path / "latest.npz", allow_pickle=False) as arrays:
        assert "covariance" in arrays.files


def test_legacy_cma_checkpoint_defaults_to_half_parents(tmp_path):
    optimizer = CMAES(np.full(2, 0.5), sigma=0.1, seed=3, population_size=6)
    checkpoint = CMACheckpoint(tmp_path, "compatible")
    checkpoint.save(optimizer)

    metadata_path = tmp_path / "latest.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["version"] = 1
    metadata.pop("parent_fraction")
    metadata.pop("mu")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    restored = checkpoint.load(seed=3, expected_parent_fraction=0.5)

    assert restored is not None
    assert restored.parent_fraction == 0.5
    assert restored.mu == 3


def test_cma_checkpoint_rejects_different_parent_fraction(tmp_path):
    optimizer = CMAES(
        np.full(2, 0.5),
        sigma=0.1,
        seed=3,
        population_size=10,
        parent_fraction=0.5,
    )
    checkpoint = CMACheckpoint(tmp_path, "compatible")
    checkpoint.save(optimizer)

    with pytest.raises(ValueError, match="parent_fraction"):
        checkpoint.load(seed=3, expected_parent_fraction=0.30)

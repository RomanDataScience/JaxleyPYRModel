from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jaxley_refactored.config import load_config
from jaxley_refactored.config.hashing import config_as_dict
from jaxley_refactored.config.schema import InitializationSpec
from jaxley_refactored.fitting.initialization import (
    initial_normalized_values,
    initial_physical_values,
)
from jaxley_refactored.parameters import combe2023_catalog


PROJECT = Path(__file__).resolve().parents[2]


def _inputs(seed: int, initialization: InitializationSpec):
    specs = combe2023_catalog().select(
        include=("soma_hbar", "soma_caLH", "AXNa", "RmSoma")
    )
    model = SimpleNamespace(
        reference_values=tuple(spec.default for spec in specs),
        parameterizer=SimpleNamespace(specs=specs),
    )
    fit = SimpleNamespace(initialization=initialization)
    runtime = SimpleNamespace(seed=seed)
    return model, fit, runtime


def test_jittered_initialization_is_reproducible_and_seed_dependent():
    initialization = InitializationSpec(
        mode="jittered_reference",
        scale=0.2,
        preserve_exact_zero_reference=True,
    )
    first = initial_normalized_values(*_inputs(7, initialization))
    repeated = initial_normalized_values(*_inputs(7, initialization))
    different = initial_normalized_values(*_inputs(8, initialization))

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_jitter_preserves_exact_zero_reference_but_moves_other_parameters():
    initialization = InitializationSpec(
        mode="jittered_reference",
        scale=0.2,
        preserve_exact_zero_reference=True,
    )
    model, fit, runtime = _inputs(12, initialization)
    physical = initial_physical_values(model, fit, runtime)
    references = np.asarray(model.reference_values)

    zero_index = next(
        index for index, spec in enumerate(model.parameterizer.specs)
        if spec.name == "soma_caLH"
    )
    assert physical[zero_index] == 0.0
    assert np.any(physical[references != 0.0] != references[references != 0.0])


def test_reference_initialization_returns_catalog_defaults():
    initialization = InitializationSpec()
    model, fit, runtime = _inputs(99, initialization)

    np.testing.assert_allclose(
        initial_physical_values(model, fit, runtime),
        model.reference_values,
    )


def test_hybrid_config_does_not_change_legacy_fit_default():
    legacy = load_config(PROJECT / "configs/losses/LSU_1_wide_bounds.yaml")
    hybrid = load_config(PROJECT / "configs/search/LSU_1_cma_adam.yaml")

    assert legacy.search.strategy == "fit"
    assert "search" not in config_as_dict(legacy)
    assert hybrid.search.strategy == "hybrid"
    assert config_as_dict(hybrid)["search"]["strategy"] == "hybrid"
    assert hybrid.search.global_search.population_size == 30
    assert hybrid.search.global_search.generations == 50
    assert hybrid.search.local_exploration.backtracking is False
    assert hybrid.search.local_refinement.backtracking is True
    assert hybrid.search.reporting.cma_plot_every_generations == 1
    assert hybrid.search.reporting.adam_plot_every_epochs == 10
    assert hybrid.search.reporting.plot_final_candidates is True

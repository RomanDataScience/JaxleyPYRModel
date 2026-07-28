from types import SimpleNamespace

import numpy as np

from jaxley_refactored.config.schema import InitializationSpec
from jaxley_refactored.fitting.initialization import (
    initial_normalized_values,
    initial_physical_values,
)
from jaxley_refactored.parameters import combe2023_catalog


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

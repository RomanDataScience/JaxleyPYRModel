import numpy as np

from neuron_optim.parameters import (
    ALL_KEYS,
    CONDUCTANCE,
    KINETIC,
    PASSIVE,
    make_parameter_space,
    passive_model_values,
)


def test_default_parameter_space_is_bounded_and_44_dimensional():
    space = make_parameter_space()
    assert len(ALL_KEYS) == 44
    assert len(space.keys) == 44
    assert np.all(space.lower < space.upper)
    normalized = space.normalize(space.reference)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    np.testing.assert_allclose(space.physical(normalized), space.reference)


def test_passive_model_mapping_disables_active_currents():
    space = make_parameter_space(include=PASSIVE)
    values = passive_model_values(dict(zip(space.keys, space.reference, strict=True)))

    assert set(space.keys) == set(PASSIVE)
    assert all(values[key] == 0.0 for key in CONDUCTANCE)
    assert all(values[key] == 1.0 for key in KINETIC)
    assert values["Epas"] == space.reference[PASSIVE.index("Epas")]

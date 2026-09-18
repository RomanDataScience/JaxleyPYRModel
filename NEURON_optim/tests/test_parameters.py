import numpy as np

from neuron_optim.parameters import ALL_KEYS, make_parameter_space


def test_default_parameter_space_is_bounded_and_44_dimensional():
    space = make_parameter_space()
    assert len(ALL_KEYS) == 44
    assert len(space.keys) == 44
    assert np.all(space.lower < space.upper)
    normalized = space.normalize(space.reference)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    np.testing.assert_allclose(space.physical(normalized), space.reference)

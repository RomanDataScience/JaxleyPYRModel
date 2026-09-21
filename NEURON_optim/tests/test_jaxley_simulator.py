import numpy as np
import pytest

from neuron_optim.jaxley_simulator import _linear_play_interval_current


def test_linear_play_interval_current_uses_interval_averages_and_padding():
    current = np.array([0.0, 2.0, 4.0, 8.0])

    np.testing.assert_allclose(
        _linear_play_interval_current(current),
        [1.0, 3.0, 6.0, 8.0],
    )


def test_linear_play_interval_current_requires_two_samples():
    with pytest.raises(ValueError):
        _linear_play_interval_current(np.array([1.0]))

import jax.numpy as jnp
import numpy as np

from jaxley_refactored.config.schema import LineSearchSpec, OptimizerSpec
from jaxley_refactored.fitting.optimizer import Adam, BacktrackingLineSearch
from jaxley_refactored.parameters import ProjectedBoxSpace


def test_backtracking_accepts_first_loss_decreasing_candidate_and_grows_rate():
    spec = LineSearchSpec(
        enabled=True,
        reduction_factor=0.5,
        growth_factor=1.2,
        minimum_learning_rate=1e-4,
        maximum_learning_rate=1.0,
        maximum_trials=6,
    )
    space = ProjectedBoxSpace(jnp.zeros(1), jnp.ones(1))
    adam = Adam(OptimizerSpec(line_search=spec), space)
    search = BacktrackingLineSearch(spec, adam.candidate)

    # x=0.8, target=0.7: the first three rates overshoot the minimum.
    # The fourth candidate (lr=0.125) lowers the quadratic loss.
    result = search.search(
        jnp.asarray([0.8]),
        jnp.asarray([1.0]),
        current_loss=0.01,
        initial_learning_rate=1.0,
        evaluate=lambda values: (jnp.sum((values - 0.7) ** 2), None),
    )

    assert result.accepted
    assert result.trials == 4
    assert result.learning_rate == 0.125
    assert result.next_learning_rate == 0.15
    np.testing.assert_allclose(result.values, [0.675])


def test_backtracking_rejects_step_without_committing_values():
    spec = LineSearchSpec(
        enabled=True,
        minimum_learning_rate=0.1,
        maximum_learning_rate=1.0,
        maximum_trials=3,
    )
    space = ProjectedBoxSpace(jnp.zeros(1), jnp.ones(1))
    adam = Adam(OptimizerSpec(line_search=spec), space)
    search = BacktrackingLineSearch(spec, adam.candidate)
    initial = jnp.asarray([0.4])

    result = search.search(
        initial,
        jnp.asarray([1.0]),
        current_loss=0.0,
        initial_learning_rate=0.8,
        evaluate=lambda values: (jnp.asarray(1.0), None),
    )

    assert not result.accepted
    assert result.trials == 3
    np.testing.assert_allclose(result.values, initial)

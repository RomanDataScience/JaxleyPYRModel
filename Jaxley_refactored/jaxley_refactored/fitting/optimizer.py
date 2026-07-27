"""JAX-native Adam directions with optional loss-decreasing backtracking."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import jax.numpy as jnp

from jaxley_refactored.config.schema import LineSearchSpec, OptimizerSpec
from jaxley_refactored.parameters import ProjectedBoxSpace


@dataclass(frozen=True)
class AdamState:
    step: int
    first_moment: object
    second_moment: object
    current_learning_rate: float | None = None


@dataclass(frozen=True)
class BacktrackingResult:
    """Outcome of testing one Adam direction at progressively smaller scales."""

    values: object
    evaluation: object | None
    learning_rate: float
    next_learning_rate: float
    trials: int
    accepted: bool


class BacktrackingLineSearch:
    """Select the largest tested projected step that strictly lowers the loss."""

    def __init__(self, spec: LineSearchSpec, project_step: Callable):
        self.spec = spec
        self.project_step = project_step

    def search(
        self,
        values,
        direction,
        current_loss: float,
        initial_learning_rate: float,
        evaluate: Callable,
    ) -> BacktrackingResult:
        rate = min(
            self.spec.maximum_learning_rate,
            max(self.spec.minimum_learning_rate, initial_learning_rate),
        )
        next_rate = rate
        for trial in range(1, self.spec.maximum_trials + 1):
            candidate = self.project_step(values, direction, rate)
            evaluation = evaluate(candidate)
            candidate_loss = float(evaluation[0])
            if math.isfinite(candidate_loss) and candidate_loss < current_loss:
                return BacktrackingResult(
                    values=candidate,
                    evaluation=evaluation,
                    learning_rate=rate,
                    next_learning_rate=min(
                        self.spec.maximum_learning_rate,
                        rate * self.spec.growth_factor,
                    ),
                    trials=trial,
                    accepted=True,
                )
            next_rate = max(
                self.spec.minimum_learning_rate,
                rate * self.spec.reduction_factor,
            )
            if next_rate == rate:
                break
            rate = next_rate
        return BacktrackingResult(
            values=values,
            evaluation=None,
            learning_rate=rate,
            next_learning_rate=next_rate,
            trials=trial,
            accepted=False,
        )


class Adam:
    def __init__(self, spec: OptimizerSpec, space: ProjectedBoxSpace):
        self.spec = spec
        self.space = space

    def initialize(self, values):
        zeros = jnp.zeros_like(values)
        return AdamState(
            step=0,
            first_moment=zeros,
            second_moment=zeros,
            current_learning_rate=self.spec.learning_rate,
        )

    def direction(self, gradient, state: AdamState):
        """Return a clipped Adam direction and the state to commit on acceptance."""
        norm = jnp.linalg.norm(gradient)
        scale = jnp.minimum(
            1.0, self.spec.gradient_clip_norm / (norm + self.spec.epsilon)
        )
        gradient = gradient * scale
        step = state.step + 1
        first = (
            self.spec.beta1 * state.first_moment
            + (1.0 - self.spec.beta1) * gradient
        )
        second = (
            self.spec.beta2 * state.second_moment
            + (1.0 - self.spec.beta2) * gradient**2
        )
        first_hat = first / (1.0 - self.spec.beta1**step)
        second_hat = second / (1.0 - self.spec.beta2**step)
        direction = first_hat / (
            jnp.sqrt(second_hat) + self.spec.epsilon
        )
        learning_rate = (
            self.spec.learning_rate
            if state.current_learning_rate is None
            else state.current_learning_rate
        )
        return (
            direction,
            AdamState(
                step=step,
                first_moment=first,
                second_moment=second,
                current_learning_rate=learning_rate,
            ),
            norm,
        )

    def candidate(self, values, direction, learning_rate: float):
        """Project one proposed step into normalized parameter bounds."""
        return self.space.project(values - learning_rate * direction)

    def accept(self, proposed_state: AdamState, learning_rate: float) -> AdamState:
        """Commit moments and grow the next trial step after an accepted move."""
        search = self.spec.line_search
        next_rate = min(
            search.maximum_learning_rate,
            learning_rate * search.growth_factor,
        )
        return replace(proposed_state, current_learning_rate=next_rate)

    def reject(self, state: AdamState, learning_rate: float) -> AdamState:
        """Keep moments unchanged and retain a smaller trial rate for next epoch."""
        return replace(
            state,
            current_learning_rate=max(
                self.spec.line_search.minimum_learning_rate, learning_rate
            ),
        )

    def update(self, values, gradient, state: AdamState):
        """Apply ordinary fixed-learning-rate Adam."""
        direction, proposed_state, norm = self.direction(gradient, state)
        updated = self.candidate(values, direction, self.spec.learning_rate)
        return updated, proposed_state, norm

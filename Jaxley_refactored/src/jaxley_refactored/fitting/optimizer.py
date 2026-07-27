"""Minimal JAX-native Adam with global-norm clipping and box projection."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jaxley_refactored.config.schema import OptimizerSpec
from jaxley_refactored.parameters import ProjectedBoxSpace


@dataclass(frozen=True)
class AdamState:
    step: int
    first_moment: object
    second_moment: object


class Adam:
    def __init__(self, spec: OptimizerSpec, space: ProjectedBoxSpace):
        self.spec = spec
        self.space = space

    def initialize(self, values):
        zeros = jnp.zeros_like(values)
        return AdamState(step=0, first_moment=zeros, second_moment=zeros)

    def update(self, values, gradient, state: AdamState):
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
        updated = values - self.spec.learning_rate * first_hat / (
            jnp.sqrt(second_hat) + self.spec.epsilon
        )
        return (
            self.space.project(updated),
            AdamState(step=step, first_moment=first, second_moment=second),
            norm,
        )


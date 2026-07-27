"""Boundary-safe optimization coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp

from .catalog import ParameterSpec


@dataclass(frozen=True)
class ProjectedBoxSpace:
    """Map normalized optimizer coordinates to physical parameter values.

    Coordinates live in ``[0, 1]`` and are projected after every optimizer
    update. Unlike a sigmoid transform, this represents exact boundary values,
    so a conductance whose reference is exactly zero remains exactly zero.
    """

    lower: jnp.ndarray
    upper: jnp.ndarray

    @classmethod
    def from_specs(cls, specs: Iterable[ParameterSpec]) -> "ProjectedBoxSpace":
        specs = tuple(specs)
        return cls(
            lower=jnp.asarray([spec.bounds[0] for spec in specs]),
            upper=jnp.asarray([spec.bounds[1] for spec in specs]),
        )

    def normalize(self, physical):
        physical = jnp.asarray(physical)
        return jnp.clip((physical - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def physical(self, normalized):
        normalized = self.project(normalized)
        return self.lower + normalized * (self.upper - self.lower)

    @staticmethod
    def project(normalized):
        return jnp.clip(jnp.asarray(normalized), 0.0, 1.0)


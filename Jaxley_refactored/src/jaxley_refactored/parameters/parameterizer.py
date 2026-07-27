"""Selective conversion from a parameter vector to Jaxley's ``param_state``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import jax.numpy as jnp

from .catalog import ParameterSpec


class ParameterStateBackend(Protocol):
    """Narrow interface implemented by a model-specific compatibility backend."""

    def parameter_state(
        self,
        cell: Any,
        keys: Sequence[str],
        values: Any,
        state: Any = None,
    ) -> Any: ...


@dataclass(frozen=True)
class Parameterizer:
    """Immutable, fit-specific parameter-state factory.

    The key tuple is static and the values are dynamic JAX arrays. This is the
    structure JAX needs to compile one executable and reuse it for every epoch.
    """

    cell: Any
    specs: tuple[ParameterSpec, ...]
    backend: ParameterStateBackend
    fixed_specs: tuple[ParameterSpec, ...] = ()
    fixed_values: tuple[float, ...] = ()

    def __post_init__(self):
        if len(self.fixed_specs) != len(self.fixed_values):
            raise ValueError("Each fixed parameter requires exactly one value.")
        overlap = set(self.keys) & {spec.name for spec in self.fixed_specs}
        if overlap:
            raise ValueError(f"Parameters cannot be both fitted and fixed: {overlap}")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def state(self, values, base_state=None):
        state_keys = self.keys + tuple(spec.name for spec in self.fixed_specs)
        state_values = values
        if self.fixed_values:
            state_values = jnp.concatenate(
                (jnp.asarray(values), jnp.asarray(self.fixed_values))
            )
        return self.backend.parameter_state(
            self.cell,
            state_keys,
            state_values,
            base_state,
        )

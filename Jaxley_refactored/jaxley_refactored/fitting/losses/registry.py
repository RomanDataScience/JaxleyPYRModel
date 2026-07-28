"""Loss registry resolved before JIT compilation."""

from __future__ import annotations

from collections.abc import Callable

from . import primitives


class LossRegistry:
    """Map stable configuration names to small differentiable callables."""

    def __init__(self):
        self._terms: dict[str, Callable] = {}

    def register(self, name: str, term: Callable) -> None:
        if name in self._terms:
            raise ValueError(f"Loss already registered: {name}")
        self._terms[name] = term

    def get(self, name: str) -> Callable:
        try:
            return self._terms[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown loss {name!r}; available: {sorted(self._terms)}"
            ) from error


def default_loss_registry() -> LossRegistry:
    registry = LossRegistry()
    registry.register("masked_voltage_mse", primitives.masked_mse)
    registry.register("voltage_mse", primitives.masked_mse)
    registry.register("voltage_mae", primitives.masked_mae)
    registry.register("pseudo_huber", primitives.pseudo_huber)
    registry.register("normalized_voltage_mse", primitives.normalized_mse)
    registry.register("derivative_mse", primitives.derivative_mse)
    registry.register("correlation_loss", primitives.correlation_loss)
    registry.register("resting_voltage_error", primitives.mean_voltage_error)
    registry.register("steady_state_error", primitives.mean_voltage_error)
    registry.register("soft_firing_rate_error", primitives.soft_firing_rate_error)
    registry.register("subthreshold_mean_error", primitives.subthreshold_mean_error)
    registry.register(
        "soft_minimum_voltage_error", primitives.soft_minimum_voltage_error
    )
    registry.register(
        "soft_maximum_voltage_error", primitives.soft_maximum_voltage_error
    )
    return registry

"""Composable, registered, JAX-safe loss functions."""

from .composite import (
    BucketObjective,
    apply_multiplicative_penalties,
    component_denominators,
)
from .primitives import (
    masked_mse,
    soft_upward_crossing_count,
    weighted_bucket_loss,
)
from .registry import LossRegistry, default_loss_registry

__all__ = [
    "BucketObjective",
    "LossRegistry",
    "apply_multiplicative_penalties",
    "component_denominators",
    "default_loss_registry",
    "masked_mse",
    "soft_upward_crossing_count",
    "weighted_bucket_loss",
]

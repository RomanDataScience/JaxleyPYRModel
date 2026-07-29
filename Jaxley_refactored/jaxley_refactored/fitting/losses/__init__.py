"""Composable, registered, JAX-safe loss functions."""

from .composite import (
    BucketObjective,
    apply_multiplicative_penalties,
    component_denominators,
    observed_interspike_masks,
)
from .primitives import (
    masked_mse,
    soft_dblo_error,
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
    "observed_interspike_masks",
    "soft_dblo_error",
    "soft_upward_crossing_count",
    "weighted_bucket_loss",
]

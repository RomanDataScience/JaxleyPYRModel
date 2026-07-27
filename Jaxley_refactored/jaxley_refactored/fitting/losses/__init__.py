"""Composable, registered, JAX-safe loss functions."""

from .composite import BucketObjective, component_denominators
from .primitives import masked_mse, weighted_bucket_loss
from .registry import LossRegistry, default_loss_registry

__all__ = [
    "BucketObjective",
    "LossRegistry",
    "component_denominators",
    "default_loss_registry",
    "masked_mse",
    "weighted_bucket_loss",
]

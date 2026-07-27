"""Small, transformation-safe objective primitives."""

from __future__ import annotations

import jax.numpy as jnp


def masked_mse(predicted, observed, mask):
    mask = jnp.asarray(mask, dtype=predicted.dtype)
    denominator = jnp.sum(mask, axis=-1)
    if mask.shape[-1] == 0:
        raise ValueError("A score mask cannot have zero time samples.")
    return jnp.sum(mask * (predicted - observed) ** 2, axis=-1) / denominator


def weighted_bucket_loss(predicted, observed, mask, weights):
    return jnp.sum(jnp.asarray(weights) * masked_mse(predicted, observed, mask))


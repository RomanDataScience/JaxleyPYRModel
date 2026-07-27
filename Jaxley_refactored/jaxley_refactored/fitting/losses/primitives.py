"""Differentiable per-trace loss primitives."""

from __future__ import annotations

import jax.numpy as jnp


def _masked_mean(values, mask):
    mask = jnp.asarray(mask, dtype=values.dtype)
    denominator = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    return jnp.sum(mask * values, axis=-1) / denominator


def masked_mse(predicted, observed, mask, *, scale=1.0, **_):
    return _masked_mean(((predicted - observed) / scale) ** 2, mask)


def masked_mae(predicted, observed, mask, *, scale=1.0, **_):
    return _masked_mean(jnp.abs(predicted - observed) / scale, mask)


def pseudo_huber(predicted, observed, mask, *, scale=1.0, delta=1.0, **_):
    residual = (predicted - observed) / scale
    values = delta**2 * (jnp.sqrt(1.0 + (residual / delta) ** 2) - 1.0)
    return _masked_mean(values, mask)


def normalized_mse(predicted, observed, mask, *, scale=1.0, **_):
    mask_float = jnp.asarray(mask, dtype=observed.dtype)
    high = jnp.max(jnp.where(mask, observed, -jnp.inf), axis=-1)
    low = jnp.min(jnp.where(mask, observed, jnp.inf), axis=-1)
    dynamic_scale = jnp.maximum(high - low, scale)
    return _masked_mean(
        ((predicted - observed) / dynamic_scale[:, None]) ** 2, mask_float
    )


def derivative_mse(predicted, observed, mask, *, dt_ms, scale=1.0, **_):
    predicted_derivative = jnp.diff(predicted, axis=-1) / dt_ms
    observed_derivative = jnp.diff(observed, axis=-1) / dt_ms
    derivative_mask = mask[..., 1:] & mask[..., :-1]
    return _masked_mean(
        ((predicted_derivative - observed_derivative) / scale) ** 2,
        derivative_mask,
    )


def correlation_loss(predicted, observed, mask, **_):
    mask_float = jnp.asarray(mask, dtype=predicted.dtype)
    count = jnp.maximum(jnp.sum(mask_float, axis=-1), 1.0)
    predicted_mean = jnp.sum(mask_float * predicted, axis=-1) / count
    observed_mean = jnp.sum(mask_float * observed, axis=-1) / count
    predicted_centered = mask_float * (predicted - predicted_mean[:, None])
    observed_centered = mask_float * (observed - observed_mean[:, None])
    numerator = jnp.sum(predicted_centered * observed_centered, axis=-1)
    denominator = jnp.sqrt(
        jnp.sum(predicted_centered**2, axis=-1)
        * jnp.sum(observed_centered**2, axis=-1)
        + 1e-12
    )
    return 1.0 - numerator / denominator


def mean_voltage_error(predicted, observed, mask, *, scale=1.0, **_):
    difference = _masked_mean(predicted, mask) - _masked_mean(observed, mask)
    return (difference / scale) ** 2


def weighted_bucket_loss(predicted, observed, mask, weights):
    """Backward-compatible weighted masked voltage MSE."""
    return jnp.sum(jnp.asarray(weights) * masked_mse(predicted, observed, mask))

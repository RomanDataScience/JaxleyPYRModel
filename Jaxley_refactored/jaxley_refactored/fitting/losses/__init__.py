"""Composable, registered, JAX-safe loss functions."""

from .composite import (
    BucketObjective,
    apply_multiplicative_penalties,
    component_denominators,
    observed_interspike_masks,
    observed_spike_peak_masks,
)
from .primitives import (
    masked_mse,
    soft_dblo_error,
    soft_interspike_minimum_voltage_error,
    soft_interspike_trough_shape_error,
    soft_mean_spike_peak_voltage_error,
    soft_spike_width_slope_error,
    soft_spike_train_mse,
    soft_trough_depth_error,
    soft_ahp_depth_error,
    soft_ahp_deficit_error,
    soft_ahp_timing_moment_error,
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
    "observed_spike_peak_masks",
    "soft_dblo_error",
    "soft_interspike_minimum_voltage_error",
    "soft_interspike_trough_shape_error",
    "soft_mean_spike_peak_voltage_error",
    "soft_spike_width_slope_error",
    "soft_spike_train_mse",
    "soft_trough_depth_error",
    "soft_ahp_depth_error",
    "soft_ahp_deficit_error",
    "soft_ahp_timing_moment_error",
    "soft_upward_crossing_count",
    "weighted_bucket_loss",
]

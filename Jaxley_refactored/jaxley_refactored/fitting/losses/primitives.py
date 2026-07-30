"""Differentiable per-trace loss primitives."""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp


def _masked_mean(values, mask):
    mask = jnp.asarray(mask, dtype=values.dtype)
    denominator = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    return jnp.sum(mask * values, axis=-1) / denominator


def _soft_masked_minimum(values, mask, temperature_mV):
    """Numerically stable smooth minimum over the final masked axis."""

    mask = jnp.asarray(mask, dtype=bool)
    logits = -values / temperature_mV
    masked_logits = jnp.where(mask, logits, -jnp.inf)
    reference = jnp.max(masked_logits, axis=-1, keepdims=True)
    valid = jnp.any(mask, axis=-1)
    reference = jnp.where(valid[..., None], reference, 0.0)
    count = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    shifted_logits = jnp.where(mask, logits - reference, -jnp.inf)
    mean_exponential = jnp.sum(jnp.exp(shifted_logits), axis=-1) / count
    mean_exponential = jnp.where(valid, mean_exponential, 1.0)
    result = -temperature_mV * (
        jnp.squeeze(reference, axis=-1) + jnp.log(mean_exponential)
    )
    return jnp.where(valid, result, 0.0)


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


def mean_window_difference_error(
    predicted,
    observed,
    mask,
    *,
    comparison_mask,
    scale=1.0,
    **_,
):
    """Compare the voltage difference between two configured time windows."""

    predicted_difference = _masked_mean(predicted, mask) - _masked_mean(
        predicted, comparison_mask
    )
    observed_difference = _masked_mean(observed, mask) - _masked_mean(
        observed, comparison_mask
    )
    return ((predicted_difference - observed_difference) / scale) ** 2


def _soft_upward_crossings(
    voltage,
    *,
    threshold_mV=-20.0,
    temperature_mV=2.0,
):
    """Count positive rises in smooth threshold occupancy.

    Summing positive occupancy increments gives approximately one event for a
    complete below-to-above threshold excursion. Unlike multiplying adjacent
    occupancies, a stationary near-threshold plateau contributes zero instead
    of accumulating a fractional event at every sample.
    """

    above = jnn.sigmoid((voltage - threshold_mV) / temperature_mV)
    return jnn.relu(above[..., 1:] - above[..., :-1])


def soft_firing_rate_error(
    predicted,
    observed,
    mask,
    *,
    dt_ms,
    scale=1.0,
    threshold_mV=-20.0,
    temperature_mV=2.0,
    **_,
):
    """Squared error between continuous upward-crossing rate surrogates."""

    def soft_rate(voltage):
        pair_mask = mask[..., 1:] & mask[..., :-1]
        soft_crossings = _soft_upward_crossings(
            voltage,
            threshold_mV=threshold_mV,
            temperature_mV=temperature_mV,
        )
        duration_s = jnp.maximum(jnp.sum(pair_mask, axis=-1) * dt_ms * 1e-3, 1e-9)
        return jnp.sum(pair_mask * soft_crossings, axis=-1) / duration_s

    difference_hz = soft_rate(predicted) - soft_rate(observed)
    return (difference_hz / scale) ** 2


def soft_upward_crossing_count(
    voltage,
    event_destination_mask,
    *,
    threshold_mV=-20.0,
    temperature_mV=2.0,
):
    """Continuously approximate upward threshold crossings at selected samples.

    The destination-sample mask assigns a crossing at the stimulus boundary to
    the interval containing the new sample. This counts a crossing at the first
    recovery sample as outside the stimulus, while a crossing at stimulus onset
    remains inside.
    """

    soft_crossings = _soft_upward_crossings(
        voltage,
        threshold_mV=threshold_mV,
        temperature_mV=temperature_mV,
    )
    destinations = jnp.asarray(
        event_destination_mask[..., 1:], dtype=voltage.dtype
    )
    return jnp.sum(destinations * soft_crossings, axis=-1)


def soft_forbidden_spike_count_error(
    predicted,
    observed,
    mask,
    *,
    scale=1.0,
    threshold_mV=-20.0,
    temperature_mV=2.0,
    **_,
):
    """Penalize every smooth upward crossing in a forbidden window."""

    del observed
    count = soft_upward_crossing_count(
        predicted,
        mask,
        threshold_mV=threshold_mV,
        temperature_mV=temperature_mV,
    )
    return (count / scale) ** 2


def subthreshold_mean_error(
    predicted,
    observed,
    mask,
    *,
    scale=1.0,
    threshold_mV=-20.0,
    **_,
):
    """Compare inter-spike plateau means using a fixed experimental mask."""

    plateau_mask = mask & (observed < threshold_mV)
    difference = _masked_mean(predicted, plateau_mask) - _masked_mean(
        observed, plateau_mask
    )
    return (difference / scale) ** 2


def soft_minimum_voltage_error(
    predicted,
    observed,
    mask,
    *,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare smooth minimum voltages, e.g. recovery AHP depth."""

    difference = _soft_masked_minimum(
        predicted, mask, temperature_mV
    ) - _soft_masked_minimum(observed, mask, temperature_mV)
    return (difference / scale) ** 2


def soft_dblo_error(
    predicted,
    observed,
    mask,
    *,
    baseline_mask,
    interspike_masks,
    interspike_valid,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare smooth depolarization baseline offsets (DBLO).

    For each trace, DBL is the mean smooth minimum over fixed experimental
    interspike intervals. DBLO subtracts the mean pre-step resting voltage.
    The interval topology is derived from the observation before tracing, so
    this primitive remains continuous in the simulated voltage.
    """

    interval_masks = (
        jnp.asarray(interspike_masks, dtype=bool)
        & jnp.asarray(mask, dtype=bool)[:, None, :]
    )
    valid_intervals = (
        jnp.asarray(interspike_valid, dtype=bool)
        & jnp.any(interval_masks, axis=-1)
    )
    valid_float = valid_intervals.astype(predicted.dtype)
    interval_count = jnp.maximum(jnp.sum(valid_float, axis=-1), 1.0)

    def dblo(voltage):
        troughs = _soft_masked_minimum(
            voltage[:, None, :], interval_masks, temperature_mV
        )
        depolarization_baseline = (
            jnp.sum(valid_float * troughs, axis=-1) / interval_count
        )
        resting_voltage = _masked_mean(voltage, baseline_mask)
        return depolarization_baseline - resting_voltage

    valid_trace = (
        jnp.any(valid_intervals, axis=-1)
        & jnp.any(jnp.asarray(baseline_mask, dtype=bool), axis=-1)
    )
    difference = dblo(predicted) - dblo(observed)
    loss = (difference / scale) ** 2
    return jnp.where(valid_trace, loss, 0.0)


def soft_interspike_minimum_voltage_error(
    predicted,
    observed,
    mask,
    *,
    interspike_masks,
    interspike_valid,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare absolute mean interspike minimum voltage.

    Fixed experimental intervals cover the region after each spike peak and
    before the next spike's upward threshold crossing. Smooth minima keep the
    metric differentiable in simulated voltage. Unlike DBLO, no resting
    voltage is subtracted.
    """

    interval_masks = (
        jnp.asarray(interspike_masks, dtype=bool)
        & jnp.asarray(mask, dtype=bool)[:, None, :]
    )
    valid_intervals = (
        jnp.asarray(interspike_valid, dtype=bool)
        & jnp.any(interval_masks, axis=-1)
    )
    valid_float = valid_intervals.astype(predicted.dtype)
    interval_count = jnp.maximum(jnp.sum(valid_float, axis=-1), 1.0)

    def interspike_minima(voltage):
        minima = _soft_masked_minimum(
            voltage[:, None, :], interval_masks, temperature_mV
        )
        return minima

    difference = (
        interspike_minima(predicted) - interspike_minima(observed)
    ) / scale
    loss = jnp.sum(valid_float * difference**2, axis=-1) / interval_count
    return jnp.where(jnp.any(valid_intervals, axis=-1), loss, 0.0)


def soft_interspike_trough_shape_error(
    predicted,
    observed,
    mask,
    *,
    interspike_masks,
    interspike_valid,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare trough position, rounded width, and asymmetry within each ISI."""

    interval_masks = (
        jnp.asarray(interspike_masks, dtype=bool)
        & jnp.asarray(mask, dtype=bool)[:, None, :]
    )
    valid_intervals = (
        jnp.asarray(interspike_valid, dtype=bool)
        & jnp.any(interval_masks, axis=-1)
    )
    valid_float = valid_intervals.astype(predicted.dtype)
    interval_count = jnp.maximum(jnp.sum(valid_float, axis=-1), 1.0)

    sample_index = jnp.arange(predicted.shape[-1], dtype=predicted.dtype)
    first = jnp.min(
        jnp.where(interval_masks, sample_index, jnp.inf), axis=-1
    )
    last = jnp.max(
        jnp.where(interval_masks, sample_index, -jnp.inf), axis=-1
    )
    first = jnp.where(valid_intervals, first, 0.0)
    last = jnp.where(valid_intervals, last, 1.0)
    span = jnp.maximum(last - first, 1.0)
    phase = (sample_index[None, None, :] - first[..., None]) / span[..., None]

    def features(voltage):
        logits = -voltage[:, None, :] / temperature_mV
        logits = jnp.where(interval_masks, logits, -jnp.inf)
        reference = jnp.max(logits, axis=-1, keepdims=True)
        reference = jnp.where(valid_intervals[..., None], reference, 0.0)
        unnormalized = jnp.where(
            interval_masks, jnp.exp(logits - reference), 0.0
        )
        probabilities = unnormalized / jnp.maximum(
            jnp.sum(unnormalized, axis=-1, keepdims=True), 1e-12
        )
        center = jnp.sum(probabilities * phase, axis=-1)
        centered = phase - center[..., None]
        variance = jnp.sum(probabilities * centered**2, axis=-1)
        width = jnp.sqrt(jnp.maximum(variance, 1e-12))
        asymmetry = jnp.sum(probabilities * centered**3, axis=-1)
        return center, width, asymmetry

    predicted_features = features(predicted)
    observed_features = features(observed)
    per_interval = sum(
        ((predicted_item - observed_item) / scale) ** 2
        for predicted_item, observed_item in zip(
            predicted_features, observed_features, strict=True
        )
    ) / 3.0
    loss = jnp.sum(valid_float * per_interval, axis=-1) / interval_count
    return jnp.where(jnp.any(valid_intervals, axis=-1), loss, 0.0)


def soft_mean_spike_peak_voltage_error(
    predicted,
    observed,
    mask,
    *,
    spike_masks,
    spike_valid,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Mean per-spike squared peak-voltage error in fixed observed windows."""

    masks = (
        jnp.asarray(spike_masks, dtype=bool)
        & jnp.asarray(mask, dtype=bool)[:, None, :]
    )
    valid = jnp.asarray(spike_valid, dtype=bool) & jnp.any(masks, axis=-1)
    valid_float = valid.astype(predicted.dtype)
    count = jnp.maximum(jnp.sum(valid_float, axis=-1), 1.0)

    def peaks(voltage):
        logits = voltage[:, None, :] / temperature_mV
        masked_logits = jnp.where(masks, logits, -jnp.inf)
        reference = jnp.max(masked_logits, axis=-1, keepdims=True)
        reference = jnp.where(valid[..., None], reference, 0.0)
        sample_count = jnp.maximum(jnp.sum(masks, axis=-1), 1.0)
        mean_exponential = (
            jnp.sum(
                jnp.where(
                    masks, jnp.exp(masked_logits - reference), 0.0
                ),
                axis=-1,
            )
            / sample_count
        )
        mean_exponential = jnp.where(valid, mean_exponential, 1.0)
        return temperature_mV * (
            jnp.squeeze(reference, axis=-1) + jnp.log(mean_exponential)
        )

    difference = (peaks(predicted) - peaks(observed)) / scale
    loss = jnp.sum(valid_float * difference**2, axis=-1) / count
    return jnp.where(jnp.any(valid, axis=-1), loss, 0.0)


def soft_ahp_depth_error(
    predicted,
    observed,
    mask,
    *,
    baseline_mask,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare recovery minimum depth relative to each trace's own baseline."""

    def depth(voltage):
        baseline = _masked_mean(voltage, baseline_mask)
        minimum = _soft_masked_minimum(voltage, mask, temperature_mV)
        return baseline - minimum

    difference = (depth(predicted) - depth(observed)) / scale
    valid = jnp.any(mask, axis=-1) & jnp.any(baseline_mask, axis=-1)
    return jnp.where(valid, difference**2, 0.0)


def soft_ahp_deficit_error(
    predicted,
    observed,
    mask,
    *,
    baseline_mask,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare mean voltage deficit below baseline across recovery."""

    def mean_deficit(voltage):
        baseline = _masked_mean(voltage, baseline_mask)
        deficit = temperature_mV * jnn.softplus(
            (baseline[:, None] - voltage) / temperature_mV
        )
        return _masked_mean(deficit, mask)

    difference = (mean_deficit(predicted) - mean_deficit(observed)) / scale
    valid = jnp.any(mask, axis=-1) & jnp.any(baseline_mask, axis=-1)
    return jnp.where(valid, difference**2, 0.0)


def soft_maximum_voltage_error(
    predicted,
    observed,
    mask,
    *,
    scale=1.0,
    temperature_mV=1.0,
    **_,
):
    """Compare smooth maximum voltages, e.g. depolarizing spike peaks."""

    def soft_maximum(voltage):
        logits = voltage / temperature_mV
        masked_logits = jnp.where(mask, logits, -jnp.inf)
        reference = jnp.max(masked_logits, axis=-1, keepdims=True)
        valid = jnp.any(mask, axis=-1)
        reference = jnp.where(valid[:, None], reference, 0.0)
        count = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
        mean_exponential = (
            jnp.sum(
                jnp.where(mask, jnp.exp(masked_logits - reference), 0.0),
                axis=-1,
            )
            / count
        )
        mean_exponential = jnp.where(valid, mean_exponential, 1.0)
        result = temperature_mV * (
            jnp.squeeze(reference, axis=-1) + jnp.log(mean_exponential)
        )
        return jnp.where(valid, result, 0.0)

    difference = soft_maximum(predicted) - soft_maximum(observed)
    return (difference / scale) ** 2


def weighted_bucket_loss(predicted, observed, mask, weights):
    """Backward-compatible weighted masked voltage MSE."""
    return jnp.sum(jnp.asarray(weights) * masked_mse(predicted, observed, mask))

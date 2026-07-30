"""Bind configured loss terms to one static trace bucket."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import jax.numpy as jnp
import numpy as np

from jaxley_refactored.config.schema import LossComponentSpec, LossPenaltySpec
from jaxley_refactored.data import TraceBucket

from . import primitives
from .registry import LossRegistry


def observed_interspike_masks(
    observed_mV: np.ndarray,
    window_mask: np.ndarray,
    *,
    threshold_mV: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed peak-to-next-threshold intervals from observed spikes.

    Upward crossings are detected only when their destination sample is in the
    configured window. For each consecutive pair, the interval begins at the
    first spike's observed peak and ends immediately before the next spike's
    upward threshold crossing. A dummy all-false interval keeps shapes static
    when a bucket contains fewer than two observed spikes.
    """

    observed = np.asarray(observed_mV, dtype=float)
    window = np.asarray(window_mask, dtype=bool)
    if observed.ndim != 2 or window.shape != observed.shape:
        raise ValueError(
            "observed_mV and window_mask must be same-shape [trace, time] arrays."
        )

    crossings_by_trace: list[np.ndarray] = []
    for voltage, selected in zip(observed, window, strict=True):
        crossings = np.flatnonzero(
            (voltage[:-1] < threshold_mV)
            & (voltage[1:] >= threshold_mV)
            & selected[1:]
        ) + 1
        crossings_by_trace.append(crossings)

    interval_count = max(
        1,
        max(
            (max(0, crossings.size - 1) for crossings in crossings_by_trace),
            default=0,
        ),
    )
    masks = np.zeros(
        (observed.shape[0], interval_count, observed.shape[1]), dtype=bool
    )
    valid = np.zeros((observed.shape[0], interval_count), dtype=bool)

    for trace_index, crossings in enumerate(crossings_by_trace):
        for interval_index, (crossing, next_crossing) in enumerate(
            zip(crossings[:-1], crossings[1:], strict=True)
        ):
            peak = crossing + int(
                np.argmax(observed[trace_index, crossing:next_crossing])
            )
            masks[trace_index, interval_index, peak:next_crossing] = (
                window[trace_index, peak:next_crossing]
            )
            valid[trace_index, interval_index] = bool(
                np.any(masks[trace_index, interval_index])
            )
    return masks, valid


def observed_spike_peak_masks(
    observed_mV: np.ndarray,
    window_mask: np.ndarray,
    *,
    threshold_mV: float,
    dt_ms: float,
    half_width_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed windows around every observed in-window spike peak."""

    observed = np.asarray(observed_mV, dtype=float)
    window = np.asarray(window_mask, dtype=bool)
    if observed.ndim != 2 or window.shape != observed.shape:
        raise ValueError(
            "observed_mV and window_mask must be same-shape [trace, time] arrays."
        )
    peaks_by_trace: list[list[int]] = []
    for voltage, selected in zip(observed, window, strict=True):
        crossings = np.flatnonzero(
            (voltage[:-1] < threshold_mV)
            & (voltage[1:] >= threshold_mV)
            & selected[1:]
        ) + 1
        peaks = []
        for index, crossing in enumerate(crossings):
            stop = (
                int(crossings[index + 1])
                if index + 1 < len(crossings)
                else len(voltage)
            )
            selected_indices = np.flatnonzero(selected[crossing:stop])
            if selected_indices.size:
                local_stop = crossing + int(selected_indices[-1]) + 1
                peaks.append(
                    crossing
                    + int(np.argmax(voltage[crossing:local_stop]))
                )
        peaks_by_trace.append(peaks)

    peak_count = max(1, max((len(peaks) for peaks in peaks_by_trace), default=0))
    masks = np.zeros(
        (observed.shape[0], peak_count, observed.shape[1]), dtype=bool
    )
    valid = np.zeros((observed.shape[0], peak_count), dtype=bool)
    half_steps = max(1, int(round(half_width_ms / dt_ms)))
    for trace_index, peaks in enumerate(peaks_by_trace):
        for peak_index, peak in enumerate(peaks):
            start = max(0, peak - half_steps)
            stop = min(observed.shape[1], peak + half_steps + 1)
            masks[trace_index, peak_index, start:stop] = window[
                trace_index, start:stop
            ]
            valid[trace_index, peak_index] = bool(
                np.any(masks[trace_index, peak_index])
            )
    return masks, valid


def component_denominators(
    components: Iterable[LossComponentSpec],
    buckets: Iterable[TraceBucket],
    *,
    renormalize_protocol_filtered: bool = True,
) -> dict[str, float]:
    """Return cross-bucket component normalizers.

    The backward-compatible conditional mode divides a protocol-filtered term
    by its selected trace weight. Disabling that mode preserves the configured
    global protocol weights in the final scalar objective.
    """
    buckets = tuple(buckets)
    result = {}
    for component in components:
        selected = sum(
            record.weight
            for bucket in buckets
            for record in bucket.records
            if not component.protocols or record.protocol in component.protocols
        )
        if selected <= 0.0:
            raise ValueError(
                f"Loss component {component.label!r} selects no weighted traces."
            )
        result[component.label] = (
            selected
            if renormalize_protocol_filtered and component.protocols
            else 1.0
        )
    return result


@dataclass(frozen=True)
class BoundTerm:
    spec: LossComponentSpec
    function: object
    mask: object
    weights: object
    denominator: float
    baseline_mask: object | None
    interspike_masks: object | None
    interspike_valid: object | None
    comparison_mask: object | None
    spike_masks: object | None
    spike_valid: object | None


@dataclass(frozen=True)
class BoundPenalty:
    spec: LossPenaltySpec
    event_mask: object
    protocol_selector: object


def apply_multiplicative_penalties(base_loss, counts, penalties):
    """Apply soft-count multipliers and return their count derivatives.

    Each configured multiplier is capped only as a numerical guard for
    pathological near-threshold traces. Below that ceiling, this is exactly
    ``base_loss * product(factor ** soft_count)``.
    """

    log_multiplier = jnp.asarray(0.0, dtype=jnp.asarray(base_loss).dtype)
    count_log_slopes = {}
    for penalty in penalties:
        log_factor = jnp.log(
            jnp.asarray(penalty.factor_per_spike, dtype=log_multiplier.dtype)
        )
        raw_log_term = log_factor * counts[penalty.label]
        maximum_log_term = jnp.log(
            jnp.asarray(penalty.maximum_multiplier, dtype=log_multiplier.dtype)
        )
        log_multiplier = log_multiplier + jnp.minimum(
            raw_log_term, maximum_log_term
        )
        count_log_slopes[penalty.label] = jnp.where(
            raw_log_term < maximum_log_term,
            log_factor,
            jnp.asarray(0.0, dtype=log_multiplier.dtype),
        )
    numeric_limit = jnp.log(jnp.asarray(jnp.finfo(log_multiplier.dtype).max)) - 2.0
    numerically_active = log_multiplier < numeric_limit
    multiplier = jnp.exp(jnp.minimum(log_multiplier, numeric_limit))
    count_log_slopes = {
        label: jnp.where(
            numerically_active,
            slope,
            jnp.asarray(0.0, dtype=log_multiplier.dtype),
        )
        for label, slope in count_log_slopes.items()
    }
    return base_loss * multiplier, multiplier, count_log_slopes


class BucketObjective:
    """Composite scalar loss for one batch, with named component outputs."""

    def __init__(
        self,
        components: Iterable[LossComponentSpec],
        bucket: TraceBucket,
        denominators: Mapping[str, float],
        registry: LossRegistry,
        penalties: Iterable[LossPenaltySpec] = (),
    ):
        terms = []
        for component in components:
            protocol_selector = np.asarray(
                [
                    not component.protocols or record.protocol in component.protocols
                    for record in bucket.records
                ],
                dtype=float,
            )
            component_mask = np.asarray(
                bucket.window_masks[component.window], dtype=bool
            )
            if component.kind == "experimental_voltage_band_mse":
                component_mask = (
                    component_mask
                    & (bucket.observed_mV >= component.voltage_band_lower_mV)
                    & (bucket.observed_mV <= component.voltage_band_upper_mV)
                )
                selected = protocol_selector.astype(bool)
                if np.any(selected & ~np.any(component_mask, axis=-1)):
                    raise ValueError(
                        f"Loss component {component.label!r} selects no "
                        "experimental samples in its voltage band."
                    )
            baseline_mask = None
            interspike_masks = None
            interspike_valid = None
            comparison_mask = None
            spike_masks = None
            spike_valid = None
            if component.kind in {
                "soft_ahp_depth_error",
                "soft_ahp_deficit_error",
            }:
                baseline_mask = jnp.asarray(bucket.window_masks["baseline"])
            if component.kind == "soft_trough_depth_error":
                # Restrict the reference baseline to the scored pre-stimulus
                # interval, excluding any startup transient before scoring.
                baseline_mask = jnp.asarray(
                    bucket.window_masks["baseline"]
                    & bucket.window_masks["score"]
                )
            if component.kind == "mean_window_difference_error":
                component_mask = np.zeros_like(bucket.observed_mV, dtype=bool)
                second_mask = np.zeros_like(bucket.observed_mV, dtype=bool)
                for row, record in enumerate(bucket.records):
                    size = len(record.time_ms)
                    component_mask[row, :size] = (
                        (record.time_ms >= component.first_window_start_ms)
                        & (record.time_ms <= component.first_window_end_ms)
                    )
                    second_mask[row, :size] = (
                        (record.time_ms >= component.second_window_start_ms)
                        & (record.time_ms <= component.second_window_end_ms)
                    )
                selected = protocol_selector.astype(bool)
                if np.any(
                    selected
                    & (
                        ~np.any(component_mask, axis=-1)
                        | ~np.any(second_mask, axis=-1)
                    )
                ):
                    raise ValueError(
                        f"Loss component {component.label!r} has an empty "
                        "configured time window for a selected trace."
                    )
                comparison_mask = jnp.asarray(second_mask)
            if component.kind in {
                "soft_dblo_error",
                "soft_interspike_minimum_voltage_error",
                "soft_interspike_trough_shape_error",
            }:
                if component.kind == "soft_dblo_error":
                    baseline_mask = jnp.asarray(bucket.window_masks["baseline"])
                interval_masks, interval_valid = observed_interspike_masks(
                    bucket.observed_mV,
                    component_mask,
                    threshold_mV=component.threshold_mV,
                )
                if component.kind in {
                    "soft_interspike_minimum_voltage_error",
                    "soft_interspike_trough_shape_error",
                }:
                    # This metric is explicitly after the preceding spike:
                    # exclude the observed peak that starts each interval.
                    starts = np.argmax(interval_masks, axis=-1)
                    for trace_index, interval_index in np.argwhere(interval_valid):
                        interval_masks[
                            trace_index,
                            interval_index,
                            starts[trace_index, interval_index],
                        ] = False
                    interval_valid = np.any(interval_masks, axis=-1)
                interspike_masks = jnp.asarray(interval_masks)
                interspike_valid = jnp.asarray(interval_valid)
            if component.kind == "soft_mean_spike_peak_voltage_error":
                peak_masks, peak_valid = observed_spike_peak_masks(
                    bucket.observed_mV,
                    component_mask,
                    threshold_mV=component.threshold_mV,
                    dt_ms=bucket.dt_ms,
                    half_width_ms=component.spike_window_half_width_ms,
                )
                spike_masks = jnp.asarray(peak_masks)
                spike_valid = jnp.asarray(peak_valid)
            terms.append(
                BoundTerm(
                    spec=component,
                    function=registry.get(component.kind),
                    mask=jnp.asarray(component_mask),
                    weights=jnp.asarray(bucket.weights * protocol_selector),
                    denominator=denominators[component.label],
                    baseline_mask=baseline_mask,
                    interspike_masks=interspike_masks,
                    interspike_valid=interspike_valid,
                    comparison_mask=comparison_mask,
                    spike_masks=spike_masks,
                    spike_valid=spike_valid,
                )
            )
        self.terms = tuple(terms)
        bound_penalties = []
        for penalty in penalties:
            protocol_selector = np.asarray(
                [
                    not penalty.protocols or record.protocol in penalty.protocols
                    for record in bucket.records
                ],
                dtype=float,
            )
            bound_penalties.append(
                BoundPenalty(
                    spec=penalty,
                    event_mask=jnp.asarray(bucket.window_masks[penalty.window]),
                    protocol_selector=jnp.asarray(protocol_selector),
                )
            )
        self.penalties = tuple(bound_penalties)
        self.dt_ms = bucket.dt_ms

    def __call__(self, predicted, observed):
        contributions = {}
        total = jnp.asarray(0.0)
        for term in self.terms:
            interspike_context = {}
            if term.spec.kind in {
                "soft_dblo_error",
                "soft_interspike_minimum_voltage_error",
                "soft_interspike_trough_shape_error",
            }:
                interspike_context = {
                    "interspike_masks": term.interspike_masks,
                    "interspike_valid": term.interspike_valid,
                }
                if term.spec.kind == "soft_dblo_error":
                    interspike_context["baseline_mask"] = term.baseline_mask
            if term.spec.kind == "mean_window_difference_error":
                interspike_context["comparison_mask"] = term.comparison_mask
            if term.spec.kind == "soft_mean_spike_peak_voltage_error":
                interspike_context.update(
                    {
                        "spike_masks": term.spike_masks,
                        "spike_valid": term.spike_valid,
                    }
                )
            if term.spec.kind in {
                "soft_trough_depth_error",
                "soft_ahp_depth_error",
                "soft_ahp_deficit_error",
            }:
                interspike_context["baseline_mask"] = jnp.asarray(
                    term.baseline_mask
                )
            per_trace = term.function(
                predicted,
                observed,
                term.mask,
                dt_ms=self.dt_ms,
                scale=term.spec.scale,
                delta=term.spec.delta,
                threshold_mV=term.spec.threshold_mV,
                temperature_mV=term.spec.temperature_mV,
                **interspike_context,
            )
            contribution = (
                term.spec.weight
                * jnp.sum(term.weights * per_trace)
                / term.denominator
            )
            contributions[term.spec.label] = contribution
            total = total + contribution
        return total, contributions

    def penalty_counts(self, predicted):
        """Return raw soft spike counts for this bucket, without trace weights."""

        counts = {}
        for penalty in self.penalties:
            per_trace = primitives.soft_upward_crossing_count(
                predicted,
                penalty.event_mask,
                threshold_mV=penalty.spec.threshold_mV,
                temperature_mV=penalty.spec.temperature_mV,
            )
            counts[penalty.spec.label] = jnp.sum(
                penalty.protocol_selector * per_trace
            )
        return counts

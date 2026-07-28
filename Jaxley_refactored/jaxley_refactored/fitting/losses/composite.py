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


@dataclass(frozen=True)
class BoundPenalty:
    spec: LossPenaltySpec
    outside_stimulus_mask: object
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
            terms.append(
                BoundTerm(
                    spec=component,
                    function=registry.get(component.kind),
                    mask=jnp.asarray(bucket.window_masks[component.window]),
                    weights=jnp.asarray(bucket.weights * protocol_selector),
                    denominator=denominators[component.label],
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
                    outside_stimulus_mask=jnp.asarray(
                        bucket.window_masks["outside_stimulus"]
                    ),
                    protocol_selector=jnp.asarray(protocol_selector),
                )
            )
        self.penalties = tuple(bound_penalties)
        self.dt_ms = bucket.dt_ms

    def __call__(self, predicted, observed):
        contributions = {}
        total = jnp.asarray(0.0)
        for term in self.terms:
            per_trace = term.function(
                predicted,
                observed,
                term.mask,
                dt_ms=self.dt_ms,
                scale=term.spec.scale,
                delta=term.spec.delta,
                threshold_mV=term.spec.threshold_mV,
                temperature_mV=term.spec.temperature_mV,
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
                penalty.outside_stimulus_mask,
                threshold_mV=penalty.spec.threshold_mV,
                temperature_mV=penalty.spec.temperature_mV,
            )
            counts[penalty.spec.label] = jnp.sum(
                penalty.protocol_selector * per_trace
            )
        return counts

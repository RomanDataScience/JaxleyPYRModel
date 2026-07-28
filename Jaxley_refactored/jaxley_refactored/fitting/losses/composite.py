"""Bind configured loss terms to one static trace bucket."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import jax.numpy as jnp
import numpy as np

from jaxley_refactored.config.schema import LossComponentSpec
from jaxley_refactored.data import TraceBucket

from .registry import LossRegistry


def component_denominators(
    components: Iterable[LossComponentSpec], buckets: Iterable[TraceBucket]
) -> dict[str, float]:
    """Normalize protocol-filtered terms across all static buckets."""
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
        result[component.label] = selected
    return result


@dataclass(frozen=True)
class BoundTerm:
    spec: LossComponentSpec
    function: object
    mask: object
    weights: object
    denominator: float


class BucketObjective:
    """Composite scalar loss for one batch, with named component outputs."""

    def __init__(
        self,
        components: Iterable[LossComponentSpec],
        bucket: TraceBucket,
        denominators: Mapping[str, float],
        registry: LossRegistry,
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

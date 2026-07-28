"""Deterministic fitted-parameter initialization policies."""

from __future__ import annotations

import numpy as np


def initial_normalized_values(model, fit, runtime) -> np.ndarray:
    """Return the configured starting point in projected-box coordinates."""
    references = np.asarray(model.reference_values, dtype=float)
    lower = np.asarray([spec.bounds[0] for spec in model.parameterizer.specs])
    upper = np.asarray([spec.bounds[1] for spec in model.parameterizer.specs])
    normalized = np.clip((references - lower) / (upper - lower), 0.0, 1.0)

    initialization = fit.initialization
    if initialization.mode == "reference":
        return normalized

    rng = np.random.default_rng(runtime.seed)
    perturbation = rng.uniform(
        -initialization.scale,
        initialization.scale,
        size=normalized.shape,
    )
    jittered = np.clip(normalized + perturbation, 0.0, 1.0)
    if initialization.preserve_exact_zero_reference:
        jittered = np.where(references == 0.0, normalized, jittered)
    return jittered


def initial_physical_values(model, fit, runtime) -> np.ndarray:
    """Return the configured starting point in physical parameter units."""
    normalized = initial_normalized_values(model, fit, runtime)
    lower = np.asarray([spec.bounds[0] for spec in model.parameterizer.specs])
    upper = np.asarray([spec.bounds[1] for spec in model.parameterizer.specs])
    return lower + normalized * (upper - lower)

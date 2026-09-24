"""Combe parameter catalog and normalized-coordinate transforms.

The values mirror the catalog in ``JaxleyModel/model/model_Combe.py``.  The
optimizer keeps this small dependency-free module usable from validation jobs
that do not import JAX/Jaxley.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULTS: dict[str, float] = {
    "RmSoma": 149999.0,
    "RaSoma": 42.562,
    "RmTuft": 45373.4,
    "RaTuft": 35.0,
    "DistHalfRm": 151.741,
    "DistHalfRa": 90.8296,
    "SlopeRm": 13.8656,
    "SlopeRa": 7.76766,
    "soma_hbar": 0.00003,
    "KirGbar": 0.00020307 * 5.0,
    "Epas": -71.9879,
    "CmSoma": 1.0,
    "SpineFactorBasal": 3.5,
    "SpineFactorTuft": 3.5,
    "soma_caL": 0.00006,
    "soma_car": 0.00003,
    "gsomacar": 0.00008,
    "soma_caLH": 0.0,
    "soma_caT": 0.0003,
    "soma_km": 0.0,
    "mykca_init": 0.0,
    "soma_kca": 0.0,
    "persist": 0.00225,
    "nav16_persist_vhalf": -56.0,
    "nav16_persist_k": -0.6,
    "nav16_C1O1v2": -42.0,
    "nav16_C1O1k2": -3.5,
    "nav16_C1I1b2": 0.175,
    "nav16_C1I1v2": -51.5,
    "nav16_C1I1k2": -9.0,
    "nav16_O1I1b2": 9.0,
    "nav16_O1I1v2": 2.5,
    "nav16_O1I1k2": -9.0,
    "AXNa": 3.5,
    "gkdrsoma": 0.0,
    "gkdrdend": 0.0,
    "psoma": 0.00075,
    "slowsoma": 0.15,
    "slownotsoma": 0.1,
    "sinfsoma": 1.35,
    "soma_kap": 7.0 * 0.0005 * 4.0 * 2.75,
    "axon_kap": 7.0 * 0.0005 * 4.0 * 4.0,
    "basal_kap": 0.0025036,
    "soma_kad": 7.0 * 0.0005 * 4.0 * 2.75,
    "gna": 0.08,
    "axongkdr": 0.011,
    "gnadend": 0.015 * 1.5,
    "gkdrapical": 0.01 * 0.05,
    "gkv2soma": 0.00264 * 5.0,
    "gkv2": 0.00198 * 10.0,
    "gkv2axon": 0.00198 * 10.0,
    "gkv2scale": 0.3,
    # Kept as a fixed compatibility constant; it is not an optimization key.
    "scale_Na_conduct": 1.0,
    "distalv": 0.0,
    "proximalv": -35.0,
    "icangbar": 0.06 * 0.75,
    "icand_can": 0.0,
    "gip3": 1.85,
    "kd_deactivation_tau_scale": 1.0,
    "nat_fast_inactivation_tau_scale": 0.7,
    "nat_slow_recovery_tau_scale": 1.0,
    "h_tau_scale": 1.0,
}

BOUNDS: dict[str, tuple[float, float]] = {
    "RmSoma": (50_000.0, 300_000.0),
    "RaSoma": (20.0, 150.0),
    "RmTuft": (10_000.0, 150_000.0),
    "RaTuft": (20.0, 150.0),
    "DistHalfRm": (20.0, 500.0),
    "DistHalfRa": (20.0, 300.0),
    "SlopeRm": (1.0, 80.0),
    "SlopeRa": (1.0, 80.0),
    "Epas": (-90.0, -50.0),
    "CmSoma": (0.3, 5.0),
    "SpineFactorBasal": (1.0, 6.0),
    "SpineFactorTuft": (1.0, 6.0),
    "soma_hbar": (0.0, 0.0003),
    "KirGbar": (0.0, 0.005),
    "soma_caL": (0.0, 0.0006),
    "soma_car": (0.0, 0.0003),
    "gsomacar": (0.0, 0.0008),
    "soma_caLH": (0.0, 0.001),
    "soma_caT": (0.0, 0.003),
    "soma_km": (0.0, 0.01),
    "mykca_init": (0.0, 0.01),
    "soma_kca": (0.0, 0.01),
    # Nav1.6 persistent recovery is a strong nonlinear lever on sustained Na+
    # current; keep its calibration range below the unstable high-persist
    # regime observed in the depolarizing smoke fits.
    "persist": (0.0015, 0.0030),
    # Smooth voltage dependence of the I1 -> O1 persistent-recovery
    # transition. The hyperpolarized suppression is an emergent steady-state
    # property, not an imposed voltage cutoff.
    "nav16_persist_vhalf": (-56.5, -55.0),
    "nav16_persist_k": (-0.8, -0.4),
    "nav16_C1O1v2": (-43.0, -41.0),
    "nav16_C1O1k2": (-3.75, -3.25),
    "nav16_C1I1b2": (0.15, 0.2),
    "nav16_C1I1v2": (-53.0, -50.0),
    "nav16_C1I1k2": (-10.0, -8.0),
    "nav16_O1I1b2": (8.0, 10.0),
    "nav16_O1I1v2": (0.0, 5.0),
    "nav16_O1I1k2": (-10.0, -8.0),
    "AXNa": (0.1, 10.0),
    "gkdrsoma": (0.0, 0.02),
    "gkdrdend": (0.0, 0.02),
    "soma_kap": (0.0, 0.2),
    "axon_kap": (0.0, 0.2),
    "basal_kap": (0.0, 0.05),
    "soma_kad": (0.0, 0.2),
    "gna": (0.075, 0.085),
    "axongkdr": (0.0, 0.05),
    "gnadend": (0.0, 0.1),
    "gkdrapical": (0.0, 0.01),
    "gkv2soma": (0.0, 0.1),
    "gkv2": (0.0, 0.1),
    "gkv2axon": (0.0, 0.1),
    "gkv2scale": (0.0, 2.0),
    "icangbar": (0.0, 0.2),
    "kd_deactivation_tau_scale": (0.25, 4.0),
    "nat_fast_inactivation_tau_scale": (0.6, 0.8),
    "nat_slow_recovery_tau_scale": (0.5, 2.0),
    "h_tau_scale": (0.5, 2.0),
}

CONDUCTANCE = (
    "soma_hbar", "KirGbar", "soma_caL", "soma_car", "gsomacar",
    "soma_caLH", "soma_caT", "soma_km", "mykca_init", "soma_kca",
    "persist", "AXNa", "gkdrsoma", "gkdrdend", "soma_kap", "axon_kap", "basal_kap",
    "soma_kad", "gna", "axongkdr", "gnadend", "gkdrapical", "gkv2soma",
    "gkv2", "gkv2axon", "gkv2scale", "icangbar",
)
PASSIVE = (
    "RmSoma", "RaSoma", "RmTuft", "RaTuft", "DistHalfRm", "DistHalfRa",
    "SlopeRm", "SlopeRa", "Epas", "CmSoma", "SpineFactorBasal",
    "SpineFactorTuft",
)
KINETIC = (
    "kd_deactivation_tau_scale", "nat_fast_inactivation_tau_scale",
    "nat_slow_recovery_tau_scale", "h_tau_scale", "nav16_C1O1v2",
    "nav16_C1O1k2", "nav16_C1I1b2", "nav16_C1I1v2", "nav16_C1I1k2",
    "nav16_O1I1b2", "nav16_O1I1v2", "nav16_O1I1k2",
    "nav16_persist_vhalf", "nav16_persist_k",
)
ALL_KEYS = PASSIVE + CONDUCTANCE + KINETIC


@dataclass(frozen=True)
class ParameterSpace:
    keys: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    reference: np.ndarray

    def normalize(self, values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.clip((values - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def physical(self, normalized: Sequence[float]) -> np.ndarray:
        normalized = np.asarray(normalized, dtype=float)
        return self.lower + np.clip(normalized, 0.0, 1.0) * (self.upper - self.lower)

    def mapping(self, normalized: Sequence[float]) -> dict[str, float]:
        return dict(zip(self.keys, self.physical(normalized), strict=True))


def make_parameter_space(include: Sequence[str] | None = None,
                         exclude: Sequence[str] = (),
                         lower_overrides: Mapping[str, float] | None = None,
                         upper_overrides: Mapping[str, float] | None = None) -> ParameterSpace:
    keys = tuple(include) if include else ALL_KEYS
    excluded = set(exclude)
    keys = tuple(key for key in keys if key not in excluded)
    unknown = sorted(set(keys) - set(BOUNDS))
    if unknown:
        raise ValueError(f"Unknown Combe parameter(s): {unknown}")
    if not keys:
        raise ValueError("Parameter space cannot be empty.")
    lower_overrides = dict(lower_overrides or {})
    upper_overrides = dict(upper_overrides or {})
    unknown_overrides = sorted(
        (set(lower_overrides) | set(upper_overrides)) - set(keys)
    )
    if unknown_overrides:
        raise ValueError(
            f"Bounds override parameter(s) are not in the selected space: {unknown_overrides}"
        )
    lower = np.asarray(
        [lower_overrides.get(key, BOUNDS[key][0]) for key in keys], dtype=float
    )
    upper = np.asarray(
        [upper_overrides.get(key, BOUNDS[key][1]) for key in keys], dtype=float
    )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Parameter-space bounds must be finite.")
    if not np.all(lower < upper):
        invalid = [key for key, lo, hi in zip(keys, lower, upper, strict=True) if lo >= hi]
        raise ValueError(f"Parameter-space bounds must satisfy lower < upper: {invalid}")
    reference = np.asarray([DEFAULTS[key] for key in keys], dtype=float)
    return ParameterSpace(keys, lower, upper, reference)


def local_normalized_bounds(
    centers: Mapping[str, float],
    keys: Sequence[str],
    half_width: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return physical bounds around centers using normalized coordinates."""
    if not np.isfinite(half_width) or half_width < 0.0:
        raise ValueError("Normalized local-bound half_width must be non-negative.")
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for key in keys:
        if key not in centers:
            raise KeyError(f"Missing center value for local bound: {key}")
        global_lower, global_upper = BOUNDS[key]
        center = float(centers[key])
        normalized = np.clip(
            (center - global_lower) / (global_upper - global_lower), 0.0, 1.0
        )
        low_normalized = max(0.0, normalized - half_width)
        high_normalized = min(1.0, normalized + half_width)
        if np.isclose(low_normalized, high_normalized):
            raise ValueError(f"Local bounds collapsed for parameter: {key}")
        lower[key] = global_lower + low_normalized * (global_upper - global_lower)
        upper[key] = global_lower + high_normalized * (global_upper - global_lower)
    return lower, upper


def complete_parameter_mapping(
    space: ParameterSpace,
    normalized: Sequence[float],
    fixed_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Merge a stage candidate into a complete simulator parameter mapping."""
    mapping = dict(DEFAULTS)
    mapping.update({key: float(value) for key, value in (fixed_values or {}).items()})
    mapping.update(space.mapping(normalized))
    return mapping


def passive_model_values(values: Mapping[str, float]) -> dict[str, float]:
    """Complete a passive candidate with all active currents disabled.

    ``apply_parameters`` needs the complete Combe parameter mapping because it
    replays every mechanism on every segment.  A passive-only optimization
    therefore supplies the fitted passive values here and receives a complete
    mapping with all conductance-bearing parameters set to zero.  Kinetic
    scales are left at their neutral value because they have no effect when
    the corresponding conductances are zero.
    """
    completed = dict(DEFAULTS)
    completed.update({key: float(value) for key, value in values.items()})
    for key in CONDUCTANCE:
        completed[key] = 0.0
    for key in KINETIC:
        completed[key] = 1.0
    return completed

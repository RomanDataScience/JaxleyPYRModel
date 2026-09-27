"""Steady-state Nav1.6 (``na16a``) window conductance, computed without simulating.

The Nav1.6 Markov scheme (C1, O1, I1, I2) has a steady-state open probability
that is far from zero between about -65 and -40 mV. Multiplied by ``gna`` (soma)
or ``gnadend`` (apical dendrites) it is a persistent sodium conductance. Some
kinetic combinations inside the global bounds make it tens to hundreds of times
larger than at the default parameters, which holds the cell in depolarization
block before the step starts. ``window_ratio`` measures that against defaults
so a candidate can be rejected before it is simulated.

Rates follow ``channels_converted/mod/Nav16_a.mod``. The fitted keys come from
the parameter mapping; the rest keep their MOD values. Slow inactivation (I2)
is left out: it only lowers the open probability, and the apical copy runs
with ``dist = 0``, so leaving it out gives the conservative (larger) value.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .parameters import DEFAULTS

# Rates not exposed as fit parameters (Nav16_a.mod PARAMETER block).
_FIXED = {
    "C1O1b2": 14.0,
    "O1C1b1": 4.0, "O1C1v1": -48.0, "O1C1k1": 9.0,
    "O1I1b1": 1.0, "O1I1v1": -42.0, "O1I1k1": 12.0,
    "I1O1v1": -40.0, "I1O1k1": -10.0,
    "I1C1b1": 0.2, "I1C1v1": -65.0, "I1C1k1": 10.0,
}
_FITTED = ("C1O1v2", "C1O1k2", "I1O1b1", "C1I1b2", "C1I1v2", "C1I1k2",
           "O1I1b2", "O1I1v2", "O1I1k2")


def _rate(v: float, b: float, vv: float, k: float) -> float:
    """``rates2`` from the MOD file: ``b / (1 + exp((v - vv) / k))``."""
    arg = (v - vv) / k
    if arg < -50.0:
        return b
    if arg > 50.0:
        return 0.0
    return b / (1.0 + np.exp(arg))


def open_probability(v_mV: float, values: Mapping[str, float]) -> float:
    """Steady-state O1 occupancy of the C1/O1/I1 scheme at ``v_mV``."""
    p = dict(_FIXED)
    p.update({key: float(values[f"nav16_{key}"]) for key in _FITTED})
    c1o1 = _rate(v_mV, p["C1O1b2"], p["C1O1v2"], p["C1O1k2"])
    o1c1 = _rate(v_mV, p["O1C1b1"], p["O1C1v1"], p["O1C1k1"])
    # Q10 multiplies every rate and cancels at steady state; 0.5 is in the MOD.
    o1i1 = 0.5 * (_rate(v_mV, p["O1I1b1"], p["O1I1v1"], p["O1I1k1"])
                  + _rate(v_mV, p["O1I1b2"], p["O1I1v2"], p["O1I1k2"]))
    i1o1 = _rate(v_mV, p["I1O1b1"], p["I1O1v1"], p["I1O1k1"])
    i1c1 = _rate(v_mV, p["I1C1b1"], p["I1C1v1"], p["I1C1k1"])
    c1i1 = _rate(v_mV, p["C1I1b2"], p["C1I1v2"], p["C1I1k2"])
    # Generator over (C1, O1, I1); solve Q^T x = 0 with sum(x) = 1.
    q = np.zeros((3, 3))
    for i, j, rate in ((0, 1, c1o1), (1, 0, o1c1), (1, 2, o1i1),
                       (2, 1, i1o1), (2, 0, i1c1), (0, 2, c1i1)):
        q[i, j] += rate
        q[i, i] -= rate
    system = np.vstack([q.T, np.ones(3)])
    rhs = np.zeros(4)
    rhs[-1] = 1.0
    occupancy = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return float(occupancy[1])


def window_ratio(values: Mapping[str, float],
                 voltages_mV: Sequence[float] = (-60.0, -50.0)) -> float:
    """Largest persistent Nav1.6 conductance relative to the default model.

    Soma (``gna``) and apical (``gnadend``) conductances are compared separately
    at each voltage, each against the same compartment with every parameter at
    its default; the maximum ratio is returned.
    """
    ratio = 0.0
    for v in voltages_mV:
        candidate = open_probability(v, values)
        reference = open_probability(v, DEFAULTS)
        for key in ("gna", "gnadend"):
            default_g = DEFAULTS[key] * reference
            if default_g > 0.0:
                ratio = max(ratio, float(values[key]) * candidate / default_g)
    return ratio

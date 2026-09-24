"""Jaxley translation of ``Nav16_a_vgated_persist.mod``."""

from __future__ import annotations

import jax.numpy as jnp

from .common import channel_prefix, safe_exp
from .nav16_a import Nav16A


class Nav16AVGP(Nav16A):
    """Nav1.6 with voltage-gated persistent recovery.

    This keeps the original four-state Nav1.6 Markov scheme and kinetic-scale
    handling from :class:`Nav16A`. The only changed transition is ``I1 -> O1``:

    ``I1O1 = persist * persist_gate(v) * O1I1``.

    The gate matches ``Nav16_a_vgated_persist.mod`` exactly, including its
    default rising and falling half-activation voltages and slopes.
    """

    def __init__(self, name=None):
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params.update({
            f"{prefix}_p_vhalf": -50.0,
            f"{prefix}_p_k": 5.0,
            f"{prefix}_p_vhalf2": 0.0,
            f"{prefix}_p_k2": 15.0,
        })

    def rates2(self, v, b, vv, k):
        argument = (v - vv) / k
        logistic = b / (1.0 + safe_exp(argument))
        return jnp.where(argument < -50.0, b,
                         jnp.where(argument > 50.0, 0.0, logistic))

    def persist_gate(self, v, params):
        prefix = channel_prefix(self)
        rising = self.rates2(
            v, 1.0, params[f"{prefix}_p_vhalf"], -params[f"{prefix}_p_k"]
        )
        falling = self.rates2(
            v, 1.0, params[f"{prefix}_p_vhalf2"], params[f"{prefix}_p_k2"]
        )
        return rising * falling

    def rates(self, v, params):
        rates = list(super().rates(v, params))
        rates[3] = rates[3] * self.persist_gate(v, params)
        return tuple(rates)

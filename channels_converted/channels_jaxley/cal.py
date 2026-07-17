import jax.numpy as jnp

from .common import Channel, channel_prefix, gate_update, ghk, safe_exp, state_or_param


class Cal(Channel):
    """Jaxley translation of `cal.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gcalbar": 0.0,
            f"{prefix}_ki": 0.001,
            f"{prefix}_tfa": 5.0,
            "celsius": 34.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0}
        self.current_name = "i_Ca"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        minf, taum = self.rates(v, params)
        return {f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, minf, taum)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        cai = state_or_param(states, params, "CaCon_i", 100e-6)
        cao = state_or_param(states, params, "CaCon_e", 2.0)
        h2 = params[f"{prefix}_ki"] / (params[f"{prefix}_ki"] + cai)
        g = params[f"{prefix}_gcalbar"] * states[f"{prefix}_m"] * h2
        return g * ghk(v, cai, cao, params["celsius"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        minf, _ = self.rates(v, params)
        return {f"{prefix}_m": minf}

    def rates(self, v, params):
        prefix = channel_prefix(self)
        denom = safe_exp((-27.01 - v) / 3.8) - 1.0
        alpha = 0.055 * (-27.01 - v) / jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
        beta = 0.94 * safe_exp((-63.01 - v) / 17.0)
        taum = 1.0 / (params[f"{prefix}_tfa"] * (alpha + beta))
        minf = alpha / (alpha + beta)
        return minf, taum


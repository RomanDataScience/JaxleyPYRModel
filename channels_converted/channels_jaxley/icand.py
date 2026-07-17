import jax.numpy as jnp

from .common import Channel, channel_prefix, gate_update, safe_exp


class Icand(Channel):
    """Jaxley translation of `icand.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 1e-4,
            f"{prefix}_erev": 0.0,
            f"{prefix}_can": 0.0,
            f"{prefix}_taumin": 0.1,
            f"{prefix}_Kd": 87e-3,
        }
        self.channel_states = {f"{prefix}_Po": 0.0}
        self.current_name = "i_icand"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        can = params[f"{prefix}_can"]
        po_inf, tau = self.rates(can, params)
        return {f"{prefix}_Po": gate_update(states[f"{prefix}_Po"], dt, po_inf, tau)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_gbar"] * states[f"{prefix}_Po"] * (v - params[f"{prefix}_erev"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        po_inf, _ = self.rates(params[f"{prefix}_can"], params)
        return {f"{prefix}_Po": po_inf}

    def rates(self, cai, params):
        prefix = channel_prefix(self)
        alpha = 0.0057 * safe_exp(0.0060 * -60.0)
        beta = 0.033 * safe_exp(-0.019 * -60.0)
        alpha2 = alpha / (1.0 + params[f"{prefix}_Kd"] / jnp.maximum(cai, 1e-12))
        tau = 1.0 / (alpha2 + beta)
        return alpha2 / (alpha2 + beta), jnp.maximum(tau, params[f"{prefix}_taumin"])

import jax.numpy as jnp

from .common import Channel, channel_prefix, gate_update, state_or_param


class Kca(Channel):
    """Jaxley translation of `kca.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 0.01,
            f"{prefix}_beta": 0.03,
            f"{prefix}_cac": 0.025,
            f"{prefix}_taumin": 5.0,
            f"{prefix}_cainit": 100e-6,
            "eK": -80.0,
            "celsius": 36.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0}
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        cai = state_or_param(states, params, "CaCon_i", params[f"{prefix}_cainit"])
        m_inf, tau_m = self.rates(cai, params)
        return {f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, m_inf, tau_m)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_gbar"] * states[f"{prefix}_m"] ** 3 * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        m_inf, _ = self.rates(params[f"{prefix}_cainit"], params)
        return {f"{prefix}_m": m_inf}

    def rates(self, cai, params):
        prefix = channel_prefix(self)
        car = (cai / params[f"{prefix}_cac"]) ** 2
        m_inf = car / (1.0 + car)
        tadj = 3.0 ** ((params["celsius"] - 22.0) / 10.0)
        tau_m = 1.0 / params[f"{prefix}_beta"] / (1.0 + car) / tadj
        return m_inf, jnp.maximum(tau_m, params[f"{prefix}_taumin"])


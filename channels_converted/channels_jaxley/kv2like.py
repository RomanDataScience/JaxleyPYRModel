import jax.numpy as jnp

from .common import Channel, channel_prefix, gate_update, safe_exp, vtrap


class Kv2like(Channel):
    """Jaxley translation of `Kv2like.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 1e-5,
            "eK": -85.0,
            "celsius": 34.0,
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h1": 1.0,
            f"{prefix}_h2": 1.0,
        }
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        m_inf, m_tau, h_inf, h1_tau, h2_tau = self.rates(v, params)
        return {
            f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, m_inf, m_tau),
            f"{prefix}_h1": gate_update(states[f"{prefix}_h1"], dt, h_inf, h1_tau),
            f"{prefix}_h2": gate_update(states[f"{prefix}_h2"], dt, h_inf, h2_tau),
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        m = states[f"{prefix}_m"]
        h1 = states[f"{prefix}_h1"]
        h2 = states[f"{prefix}_h2"]
        g = params[f"{prefix}_gbar"] * m * m * (0.5 * h1 + 0.5 * h2)
        return g * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        m_inf, _, h_inf, _, _ = self.rates(v, params)
        return {
            f"{prefix}_m": m_inf,
            f"{prefix}_h1": h_inf,
            f"{prefix}_h2": h_inf,
        }

    def rates(self, v, params):
        qt = 2.3 ** ((params["celsius"] - 21.0) / 10.0)
        m_alpha = 0.12 * vtrap(-(v - 43.0), 11.0)
        m_beta = 0.02 * safe_exp(-(v + 1.27) / 120.0)
        m_inf = m_alpha / (m_alpha + m_beta)
        m_tau = 2.5 / (qt * (m_alpha + m_beta))
        h_inf = 1.0 / (1.0 + safe_exp((v + 58.0) / 11.0))
        h1_tau = (360.0 + (1010.0 + 23.7 * (v + 54.0)) * safe_exp(-((v + 75.0) / 48.0) ** 2)) / qt
        h2_tau = (2350.0 + 1380.0 * safe_exp(-0.011 * v) - 210.0 * safe_exp(-0.03 * v)) / qt
        return m_inf, m_tau, h_inf, h1_tau, h2_tau


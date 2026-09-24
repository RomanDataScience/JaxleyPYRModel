import jax.numpy as jnp

from .common import Channel, channel_prefix, gate_update, ghk, neuron_table, safe_exp, state_or_param


class Cat(Channel):
    """Jaxley translation of `cat.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gcatbar": 0.0,
            f"{prefix}_ki": 0.001,
            f"{prefix}_tfa": 1.0,
            f"{prefix}_tfi": 0.68,
            "celsius": 22.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0, f"{prefix}_h": 1.0}
        self.current_name = "i_Ca"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        minf, taum, hinf, tauh = self.rates(v, params)
        return {
            f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, minf, taum),
            f"{prefix}_h": gate_update(states[f"{prefix}_h"], dt, hinf, tauh),
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        cai = state_or_param(states, params, "CaCon_i", 100e-6)
        cao = state_or_param(states, params, "CaCon_e", 2.0)
        h2 = params[f"{prefix}_ki"] / (params[f"{prefix}_ki"] + cai)
        g = params[f"{prefix}_gcatbar"] * states[f"{prefix}_m"] ** 2 * states[f"{prefix}_h"] * h2
        return g * ghk(v, cai, cao, params["celsius"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        minf, _, hinf, _ = self.rates(v, params)
        return {f"{prefix}_m": minf, f"{prefix}_h": hinf}

    def rates(self, v, params):
        prefix = channel_prefix(self)
        def alpha_h_rate(voltage):
            return 1.6e-4 * safe_exp(-(voltage + 57.0) / 19.0)

        def beta_h_rate(voltage):
            return 1.0 / (safe_exp((-voltage + 15.0) / 10.0) + 1.0)

        def alpha_m_rate(voltage):
            denom = safe_exp((-voltage + 19.88) / 10.0) - 1.0
            return 0.1967 * (-voltage + 19.88) / jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)

        def beta_m_rate(voltage):
            return 0.046 * safe_exp(-voltage / 22.73)

        alph = neuron_table(v, alpha_h_rate)
        beth = neuron_table(v, beta_h_rate)
        alpm = neuron_table(v, alpha_m_rate)
        betm = neuron_table(v, beta_m_rate)
        taum = 1.0 / (params[f"{prefix}_tfa"] * (alpm + betm))
        minf = alpm / (alpm + betm)
        tauh = 1.0 / (params[f"{prefix}_tfi"] * (alph + beth))
        hinf = alph / (alph + beth)
        return minf, taum, hinf, tauh

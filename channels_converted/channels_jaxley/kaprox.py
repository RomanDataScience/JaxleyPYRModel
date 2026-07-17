import jax.numpy as jnp

from .common import Channel, FARADAY, R, channel_prefix, gate_update, safe_exp


class Kap(Channel):
    """Jaxley translation of `kaprox.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gkabar": 0.008,
            f"{prefix}_vhalfn": 11.0,
            f"{prefix}_vhalfl": -56.0,
            f"{prefix}_a0n": 0.05,
            f"{prefix}_zetan": -1.5,
            f"{prefix}_zetal": 3.0,
            f"{prefix}_gmn": 0.55,
            f"{prefix}_gml": 1.0,
            f"{prefix}_lmin": 2.0,
            f"{prefix}_nmin": 0.1,
            f"{prefix}_pw": -1.0,
            f"{prefix}_tq": -40.0,
            f"{prefix}_qq": 5.0,
            f"{prefix}_q10": 5.0,
            f"{prefix}_qtl": 1.0,
            "eK": -85.0,
            "celsius": 34.0,
        }
        self.channel_states = {f"{prefix}_n": 0.0, f"{prefix}_l": 0.0}
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        ninf, taun, linf, taul = self.rates(v, params)
        return {
            f"{prefix}_n": gate_update(states[f"{prefix}_n"], dt, ninf, taun),
            f"{prefix}_l": gate_update(states[f"{prefix}_l"], dt, linf, taul),
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        g = params[f"{prefix}_gkabar"] * states[f"{prefix}_n"] * states[f"{prefix}_l"]
        return g * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        ninf, _, linf, _ = self.rates(v, params)
        return {f"{prefix}_n": ninf, f"{prefix}_l": linf}

    def _zeta_n(self, v, params):
        prefix = channel_prefix(self)
        arg = (v - params[f"{prefix}_tq"]) / params[f"{prefix}_qq"]
        return params[f"{prefix}_zetan"] + params[f"{prefix}_pw"] / (1.0 + safe_exp(arg))

    def rates(self, v, params):
        prefix = channel_prefix(self)
        celsius = params["celsius"]
        zeta_n = self._zeta_n(v, params)
        alpn = safe_exp(1e-3 * zeta_n * (v - params[f"{prefix}_vhalfn"]) * FARADAY / (R * (273.16 + celsius)))
        betn = safe_exp(1e-3 * zeta_n * params[f"{prefix}_gmn"] * (v - params[f"{prefix}_vhalfn"]) * FARADAY / (R * (273.16 + celsius)))
        alpl = safe_exp(1e-3 * params[f"{prefix}_zetal"] * (v - params[f"{prefix}_vhalfl"]) * FARADAY / (R * (273.16 + celsius)))
        qt = params[f"{prefix}_q10"] ** ((celsius - 24.0) / 10.0)
        ninf = 1.0 / (1.0 + alpn)
        taun = betn / (qt * params[f"{prefix}_a0n"] * (1.0 + alpn))
        taun = jnp.maximum(taun, params[f"{prefix}_nmin"])
        linf = 1.0 / (1.0 + alpl)
        taul = 0.26 * (v + 50.0) / params[f"{prefix}_qtl"]
        taul = jnp.maximum(taul, params[f"{prefix}_lmin"] / params[f"{prefix}_qtl"])
        return ninf, taun, linf, taul

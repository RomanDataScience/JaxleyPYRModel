import jax.numpy as jnp

from .common import FARADAY, R, channel_prefix, gate_update, safe_exp, sigmoid_arg
from .nax import Nax


class Na3Dend(Nax):
    """Jaxley translation of `na3dend.mod`."""

    def __init__(self, name=None):
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params.update(
            {
                f"{prefix}_tha": -23.0,
                f"{prefix}_qq": 10.0,
                f"{prefix}_tq": -55.0,
                f"{prefix}_vhalfs": -60.0,
                f"{prefix}_a0s": 0.0003,
                f"{prefix}_zetas": 12.0,
                f"{prefix}_gms": 0.2,
                f"{prefix}_smax": 10.0,
                f"{prefix}_vvh": -58.0,
                f"{prefix}_vvs": 2.0,
                f"{prefix}_ar2": 1.0,
            }
        )
        self.channel_states[f"{prefix}_s"] = 1.0

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        minf, mtau, hinf, htau, sinf, taus = self.rates(v, params)
        return {
            f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, minf, mtau),
            f"{prefix}_h": gate_update(states[f"{prefix}_h"], dt, hinf, htau),
            f"{prefix}_s": gate_update(states[f"{prefix}_s"], dt, sinf, taus),
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        g = (
            params[f"{prefix}_gbar"]
            * states[f"{prefix}_m"] ** 3
            * states[f"{prefix}_h"]
            * states[f"{prefix}_s"]
        )
        return g * (v - params["eNa"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        minf, _, hinf, _, sinf, _ = self.rates(v, params)
        return {f"{prefix}_m": minf, f"{prefix}_h": hinf, f"{prefix}_s": sinf}

    def rates(self, v, params):
        prefix = channel_prefix(self)
        minf, mtau, hinf, htau = super().rates(v, params)
        c = sigmoid_arg((v - params[f"{prefix}_vvh"]) / params[f"{prefix}_vvs"])
        arg_s = 1e-3 * params[f"{prefix}_zetas"] * (v - params[f"{prefix}_vhalfs"]) * FARADAY / (
            R * (273.16 + params["celsius"])
        )
        arg_bs = (
            1e-3
            * params[f"{prefix}_zetas"]
            * params[f"{prefix}_gms"]
            * (v - params[f"{prefix}_vhalfs"])
            * FARADAY
            / (R * (273.16 + params["celsius"]))
        )
        alps = safe_exp(arg_s)
        bets = safe_exp(arg_bs)
        sinf = c + params[f"{prefix}_ar2"] * (1.0 - c)
        taus = jnp.maximum(bets / (params[f"{prefix}_a0s"] * (1.0 + alps)), params[f"{prefix}_smax"])
        return minf, mtau, hinf, htau, sinf, taus


import jax.numpy as jnp

from .common import (
    MOD_FARADAY, MOD_R, Channel, channel_prefix, gate_update, safe_exp, sigmoid_arg, trap0,
)


class Nax(Channel):
    """Jaxley translation of `nax.mod`.

    Includes the optional slow inactivation gate ``s`` (from na3dend.mod);
    ``ar2 = 1`` keeps ``s = 1``, i.e. the original channel.
    """

    slow_inactivation = True

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 0.010,
            f"{prefix}_tha": -30.0,
            f"{prefix}_qa": 7.2,
            f"{prefix}_Ra": 0.4,
            f"{prefix}_Rb": 0.124,
            f"{prefix}_thi1": -45.0,
            f"{prefix}_thi2": -45.0,
            f"{prefix}_qd": 1.5,
            f"{prefix}_qg": 1.5,
            f"{prefix}_mmin": 0.02,
            f"{prefix}_hmin": 0.5,
            f"{prefix}_q10": 2.0,
            f"{prefix}_Rg": 0.01,
            f"{prefix}_Rd": 0.03,
            f"{prefix}_thinf": -50.0,
            f"{prefix}_qinf": 1.0,
            f"{prefix}_fast_inactivation_tau_scale": 1.0,
            "eNa": 50.0,
            "celsius": 34.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0, f"{prefix}_h": 1.0}
        if self.slow_inactivation:
            self.channel_params.update({
                f"{prefix}_vhalfs": -60.0,
                f"{prefix}_a0s": 0.0003,
                f"{prefix}_zetas": 12.0,
                f"{prefix}_gms": 0.2,
                f"{prefix}_smax": 10.0,
                f"{prefix}_vvh": -58.0,
                f"{prefix}_vvs": 2.0,
                f"{prefix}_ar2": 1.0,
            })
            self.channel_states[f"{prefix}_s"] = 1.0
        self.current_name = "i_Na"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        minf, mtau, hinf, htau = self.rates(v, params)
        updated = {
            f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, minf, mtau),
            f"{prefix}_h": gate_update(states[f"{prefix}_h"], dt, hinf, htau),
        }
        if self.slow_inactivation:
            sinf, taus = self.slow_rates(v, params)
            updated[f"{prefix}_s"] = gate_update(states[f"{prefix}_s"], dt, sinf, taus)
        return updated

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        g = params[f"{prefix}_gbar"] * states[f"{prefix}_m"] ** 3 * states[f"{prefix}_h"]
        if self.slow_inactivation:
            g = g * states[f"{prefix}_s"]
        return g * (v - params["eNa"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        minf, _, hinf, _ = self.rates(v, params)
        initial = {f"{prefix}_m": minf, f"{prefix}_h": hinf}
        if self.slow_inactivation:
            initial[f"{prefix}_s"] = self.slow_rates(v, params)[0]
        return initial

    def slow_rates(self, v, params):
        """Slow inactivation steady state and time constant (na3dend.mod)."""
        prefix = channel_prefix(self)
        c = sigmoid_arg((v - params[f"{prefix}_vvh"]) / params[f"{prefix}_vvs"])
        scale = 1e-3 * params[f"{prefix}_zetas"] * (v - params[f"{prefix}_vhalfs"]) * MOD_FARADAY / (
            MOD_R * (273.16 + params["celsius"])
        )
        alps = safe_exp(scale)
        bets = safe_exp(scale * params[f"{prefix}_gms"])
        sinf = c + params[f"{prefix}_ar2"] * (1.0 - c)
        taus = jnp.maximum(bets / (params[f"{prefix}_a0s"] * (1.0 + alps)), params[f"{prefix}_smax"])
        return sinf, taus

    def rates(self, v, params):
        prefix = channel_prefix(self)
        qt = params[f"{prefix}_q10"] ** ((params["celsius"] - 24.0) / 10.0)
        a = trap0(v, params[f"{prefix}_tha"], params[f"{prefix}_Ra"], params[f"{prefix}_qa"])
        b = trap0(-v, -params[f"{prefix}_tha"], params[f"{prefix}_Rb"], params[f"{prefix}_qa"])
        mtau = jnp.maximum(1.0 / (a + b) / qt, params[f"{prefix}_mmin"])
        minf = a / (a + b)
        a = trap0(v, params[f"{prefix}_thi1"], params[f"{prefix}_Rd"], params[f"{prefix}_qd"])
        b = trap0(-v, -params[f"{prefix}_thi2"], params[f"{prefix}_Rg"], params[f"{prefix}_qg"])
        # Scale before the hmin floor, as the patched NEURON nax/na3dend do.
        htau = jnp.maximum(
            1.0 / (a + b) / qt * params[f"{prefix}_fast_inactivation_tau_scale"],
            params[f"{prefix}_hmin"],
        )
        hinf = sigmoid_arg((v - params[f"{prefix}_thinf"]) / params[f"{prefix}_qinf"])
        return minf, mtau, hinf, htau

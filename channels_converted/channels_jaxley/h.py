import jax.numpy as jnp

from .common import Channel, channel_prefix, gate_update, safe_exp


class H(Channel):
    """Jaxley translation of `h.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 0.0,
            f"{prefix}_vhalf": -81.0,
            f"{prefix}_K": 8.5,
            f"{prefix}_eh": -10.0,
            f"{prefix}_tau_scale": 1.0,
        }
        self.channel_states = {f"{prefix}_n": 0.0}
        self.current_name = "i_H"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        ninf, taun = self.rates(v, params)
        return {f"{prefix}_n": gate_update(states[f"{prefix}_n"], dt, ninf, taun)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_gbar"] * states[f"{prefix}_n"] * (v - params[f"{prefix}_eh"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        ninf, _ = self.rates(v, params)
        return {f"{prefix}_n": ninf}

    def rates(self, v, params):
        prefix = channel_prefix(self)
        taun = 2.0 / (safe_exp((v + 186.32) / -29.91) + safe_exp((v + 21.84) / 13.77))
        taun = jnp.maximum(taun, 5.0)
        taun = taun * params[f"{prefix}_tau_scale"]
        ninf = 1.0 - 1.0 / (1.0 + safe_exp((params[f"{prefix}_vhalf"] - v) / params[f"{prefix}_K"]))
        return ninf, taun

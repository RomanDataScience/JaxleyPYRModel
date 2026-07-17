from .common import Channel, channel_prefix, gate_update, safe_exp, sigmoid_arg


class Km(Channel):
    """Jaxley translation of `km.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 1e-4,
            f"{prefix}_vhalfl": -42.0,
            f"{prefix}_kl": -4.0,
            f"{prefix}_vhalft": -42.0,
            f"{prefix}_a0t": 0.04,
            f"{prefix}_zetat": 4.0,
            f"{prefix}_gmt": 0.7,
            f"{prefix}_q10": 5.0,
            f"{prefix}_b0": 60.0,
            f"{prefix}_st": 1.0,
            "eK": -85.0,
            "celsius": 34.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0}
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        inf, tau = self.rates(v, params)
        return {f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, inf, tau)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        m = states[f"{prefix}_m"]
        g = params[f"{prefix}_gbar"] * (m ** params[f"{prefix}_st"])
        return g * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        inf, _ = self.rates(v, params)
        return {f"{prefix}_m": inf}

    def rates(self, v, params):
        prefix = channel_prefix(self)
        inf = sigmoid_arg((v - params[f"{prefix}_vhalfl"]) / params[f"{prefix}_kl"])
        alpha = safe_exp(0.0378 * params[f"{prefix}_zetat"] * (v - params[f"{prefix}_vhalft"]))
        beta = safe_exp(0.0378 * params[f"{prefix}_zetat"] * params[f"{prefix}_gmt"] * (v - params[f"{prefix}_vhalft"]))
        tau = params[f"{prefix}_b0"] + beta / (params[f"{prefix}_a0t"] * (1.0 + alpha))
        return inf, tau


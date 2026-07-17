from .common import Channel, channel_prefix, gate_update, inv_sigmoid_arg, sigmoid_arg


class Kd(Channel):
    """Jaxley translation of `kd.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 0.1,
            f"{prefix}_vhalfm": -43.0,
            f"{prefix}_km": 8.0,
            f"{prefix}_vhalfh": -67.0,
            f"{prefix}_kh": 7.3,
            f"{prefix}_q10": 2.3,
            "eK": -100.0,
            "celsius": 34.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0, f"{prefix}_h": 1.0}
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        minf, mtau, hinf, htau = self.rates(v, params)
        return {
            f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, minf, mtau),
            f"{prefix}_h": gate_update(states[f"{prefix}_h"], dt, hinf, htau),
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_gbar"] * states[f"{prefix}_m"] * states[f"{prefix}_h"] * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        minf, _, hinf, _ = self.rates(v, params)
        return {f"{prefix}_m": minf, f"{prefix}_h": hinf}

    def rates(self, v, params):
        prefix = channel_prefix(self)
        arg_m = (v - params[f"{prefix}_vhalfm"]) / params[f"{prefix}_km"]
        arg_h = (v - params[f"{prefix}_vhalfh"]) / params[f"{prefix}_kh"]
        return inv_sigmoid_arg(arg_m), 0.6, sigmoid_arg(arg_h), 1500.0


from .common import Channel, channel_prefix, gate_update, sigmoid_arg


class Nap(Channel):
    """Jaxley translation of `nap.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gnabar": 0.0,
            f"{prefix}_vhalf": -60.4,
            f"{prefix}_K": 2.0,
            "eNa": 50.0,
        }
        self.channel_states = {f"{prefix}_n": 0.0}
        self.current_name = "i_Na"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        n_inf = self.n_inf(v, params)
        return {f"{prefix}_n": gate_update(states[f"{prefix}_n"], dt, n_inf, 10.0)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        n = states[f"{prefix}_n"]
        return params[f"{prefix}_gnabar"] * n * n * n * (v - params["eNa"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        return {f"{prefix}_n": self.n_inf(v, params)}

    def n_inf(self, v, params):
        prefix = channel_prefix(self)
        return sigmoid_arg((params[f"{prefix}_vhalf"] - v) / params[f"{prefix}_K"])


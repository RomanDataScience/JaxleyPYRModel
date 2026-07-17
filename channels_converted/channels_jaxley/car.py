from .common import Channel, channel_prefix, gate_update, sigmoid_arg


class Car(Channel):
    """Jaxley translation of `car.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gcabar": 0.0,
            "eCa": 140.0,
        }
        self.channel_states = {f"{prefix}_m": 0.0, f"{prefix}_h": 1.0}
        self.current_name = "i_Ca"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        minf, mtau, hinf, htau = self.rates(v)
        return {
            f"{prefix}_m": gate_update(states[f"{prefix}_m"], dt, minf, mtau),
            f"{prefix}_h": gate_update(states[f"{prefix}_h"], dt, hinf, htau),
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        g = params[f"{prefix}_gcabar"] * states[f"{prefix}_m"] ** 3 * states[f"{prefix}_h"]
        return g * (v - params["eCa"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        minf, _, hinf, _ = self.rates(v)
        return {f"{prefix}_m": minf, f"{prefix}_h": hinf}

    @staticmethod
    def rates(v):
        return sigmoid_arg((v + 48.5) / -3.0), 5.0, sigmoid_arg((v + 53.0) / 1.0), 50.0


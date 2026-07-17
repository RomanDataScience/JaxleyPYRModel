from .common import Channel, channel_prefix, safe_exp


class Kir(Channel):
    """Jaxley translation of `PotassiumInwardRectifier.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 0.0,
            f"{prefix}_ek": -95.0,
            f"{prefix}_Offset": 15.0,
            f"{prefix}_Slope": -10.0,
        }
        self.channel_states = {}
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        return {}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        ek = params.get(f"{prefix}_ek", params.get("eK", -95.0))
        arg = -(v - ek + params[f"{prefix}_Offset"]) / params[f"{prefix}_Slope"]
        g = params[f"{prefix}_gbar"] / (1.0 + safe_exp(arg))
        return (v - ek) * g

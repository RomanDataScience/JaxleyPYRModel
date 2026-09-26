from .common import channel_prefix
from .nax import Nax


class Na12(Nax):
    """Jaxley translation of `na12.mod`: `nax` kinetics shifted by `sh` mV."""

    def __init__(self, name=None):
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params[f"{prefix}_sh"] = 0.0

    def rates(self, v, params):
        prefix = channel_prefix(self)
        return super().rates(v - params[f"{prefix}_sh"], params)

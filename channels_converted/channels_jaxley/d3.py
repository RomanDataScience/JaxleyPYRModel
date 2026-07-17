import jax.numpy as jnp

from .common import Channel, channel_prefix


class D3(Channel):
    """No-current Jaxley placeholder for geometry metadata from `d3.mod`."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_x": 0.0,
            f"{prefix}_y": 0.0,
            f"{prefix}_z": 0.0,
        }
        self.channel_states = {}
        self.current_name = "i_d3"

    def update_states(self, states, dt, v, params):
        return {}

    def compute_current(self, states, v, params):
        return jnp.zeros_like(v)


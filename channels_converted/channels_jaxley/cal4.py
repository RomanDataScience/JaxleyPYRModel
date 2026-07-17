import jax.numpy as jnp
from jaxley.pumps import Pump

from .common import FARADAY, channel_prefix


class Cal4(Pump):
    """Reduced Jaxley translation of the `cal4.mod` calcium kinetic state.

    The original MOD file uses NEURON KINETIC radial diffusion and buffer
    constructs. Jaxley represents mechanisms that modify ion concentrations as
    pumps, so this class owns the `CaCon_i` update and can be combined with
    `cell.diffuse("CaCon_i")` for longitudinal calcium diffusion.
    """

    def __init__(self, name=None):
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_ip3i": 10e-3,
            f"{prefix}_cai0": 50e-6,
            f"{prefix}_cath": 0.2e-3,
            f"{prefix}_gamma": 8.0,
            f"{prefix}_jmax": 3.5e-3,
            f"{prefix}_caer": 0.400,
            f"{prefix}_Kip3": 0.8e-3,
            f"{prefix}_Kact": 0.7e-3,
            f"{prefix}_kon": 2.7,
            f"{prefix}_Kinh": 0.6e-3,
            f"{prefix}_sites": 3.0,
            f"{prefix}_alpha": 1.0,
            f"{prefix}_beta": 1.0,
            f"{prefix}_vmax": 1e-4,
            f"{prefix}_Kp": 0.27e-3,
            f"{prefix}_current_fraction": 1.0,
            f"{prefix}_shell_depth": 0.1,
        }
        self.channel_states = {
            "i_Ca": 1e-8,
            "CaCon_i": 50e-6,
            f"{prefix}_ho": 0.0,
        }
        self.ion_name = "CaCon_i"
        self.current_name = "i_Cal4"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        ca = states["CaCon_i"]
        ho = states[f"{prefix}_ho"]
        ca_safe = jnp.maximum(ca, 1e-12)
        dho = (
            params[f"{prefix}_kon"] * params[f"{prefix}_Kinh"] * (1.0 - ho)
            - params[f"{prefix}_kon"] * ca_safe * ho
        )
        new_ho = jnp.clip(ho + dt * dho, 0.0, 1.0)

        return {
            "i_Ca": states["i_Ca"],
            "CaCon_i": states["CaCon_i"],
            f"{prefix}_ho": new_ho,
        }

    def compute_current(self, states, modified_state, params):
        prefix = channel_prefix(self)
        ca = jnp.maximum(modified_state, 1e-12)
        ho = states[f"{prefix}_ho"]

        current_drive = self.calcium_current_drive(states["i_Ca"], params)
        ip3_release = self.ip3_release(ca, ho, params)
        leak = self.leak_balance(params) * params[f"{prefix}_beta"] * (
            1.0 - ca / params[f"{prefix}_caer"]
        )
        serca = self.serca_uptake(ca, params)
        membrane_pump = self.membrane_pump(ca, params)

        dca_dt = current_drive + ip3_release + leak - serca - membrane_pump
        return -dca_dt

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        ca = params[f"{prefix}_cai0"]
        ho = params[f"{prefix}_Kinh"] / (ca + params[f"{prefix}_Kinh"])
        return {"CaCon_i": ca, f"{prefix}_ho": ho}

    def calcium_current_drive(self, i_ca, params):
        prefix = channel_prefix(self)
        depth = jnp.maximum(params[f"{prefix}_shell_depth"], 1e-12)
        fraction = params[f"{prefix}_current_fraction"]
        return -10_000.0 * i_ca * fraction / (2.0 * FARADAY * depth)

    def ip3_release(self, ca, ho, params):
        prefix = channel_prefix(self)
        ip3 = params[f"{prefix}_ip3i"] / (
            params[f"{prefix}_ip3i"] + params[f"{prefix}_Kip3"]
        )
        act = ca / (ca + params[f"{prefix}_Kact"])
        return (
            params[f"{prefix}_alpha"]
            * params[f"{prefix}_jmax"]
            * (1.0 - ca / params[f"{prefix}_caer"])
            * (ip3 * act * ho) ** params[f"{prefix}_sites"]
        )

    def serca_uptake(self, ca, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_beta"] * params[f"{prefix}_vmax"] * ca**2 / (
            ca**2 + params[f"{prefix}_Kp"] ** 2
        )

    def membrane_pump(self, ca, params):
        prefix = channel_prefix(self)
        return (
            1e-3
            * params[f"{prefix}_gamma"]
            * jnp.maximum(ca - params[f"{prefix}_cath"], 0.0)
        )

    def leak_balance(self, params):
        prefix = channel_prefix(self)
        ca = params[f"{prefix}_cai0"]
        ho = params[f"{prefix}_Kinh"] / (ca + params[f"{prefix}_Kinh"])
        ip3 = params[f"{prefix}_ip3i"] / (
            params[f"{prefix}_ip3i"] + params[f"{prefix}_Kip3"]
        )
        act = ca / (ca + params[f"{prefix}_Kact"])
        serca = params[f"{prefix}_vmax"] * ca**2 / (
            ca**2 + params[f"{prefix}_Kp"] ** 2
        )
        channel = (
            params[f"{prefix}_jmax"]
            * (1.0 - ca / params[f"{prefix}_caer"])
            * (ip3 * act * ho) ** params[f"{prefix}_sites"]
        )
        return (serca - channel) / jnp.maximum(
            1.0 - ca / params[f"{prefix}_caer"], 1e-12
        )


def enable_cal4_diffusion(cell, axial_diffusion=0.22):
    """Enable whole-cell diffusion for the `CaCon_i` state used by `Cal4`."""
    if axial_diffusion <= 0.0:
        raise ValueError("axial_diffusion must be strictly positive.")

    diffusion_states = getattr(cell, "diffusion_states", None)
    if diffusion_states is None and hasattr(cell, "base"):
        diffusion_states = getattr(cell.base, "diffusion_states", [])

    if "CaCon_i" not in (diffusion_states or []):
        cell.diffuse("CaCon_i")
    cell.set("axial_diffusion_CaCon_i", axial_diffusion)
    return cell

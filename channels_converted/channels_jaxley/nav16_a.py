import jax.numpy as jnp

from .common import Channel, channel_prefix, safe_exp


class Nav16A(Channel):
    """Jaxley translation of the Markov channel in `Nav16_a.mod`.

    The NEURON mechanism solves a sparse kinetic system. This translation uses
    an implicit Euler solve for the linear Markov system at each fixed-voltage
    step, matching NEURON's `KINETIC ... METHOD sparse` behavior, and an
    algebraic steady-state solve for initialization.
    """

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gbar": 0.1,
            f"{prefix}_I2init": 0.0,
            f"{prefix}_C1O1b2": 14.0,
            f"{prefix}_C1O1v2": 0.0,
            f"{prefix}_C1O1k2": -6.0,
            f"{prefix}_O1C1b1": 4.0,
            f"{prefix}_O1C1v1": -48.0,
            f"{prefix}_O1C1k1": 9.0,
            f"{prefix}_O1C1b2": 0.0,
            f"{prefix}_O1C1v2": 0.0,
            f"{prefix}_O1C1k2": -5.1,
            f"{prefix}_O1I1b1": 1.0,
            f"{prefix}_O1I1v1": -42.0,
            f"{prefix}_O1I1k1": 12.0,
            f"{prefix}_O1I1b2": 5.0,
            f"{prefix}_O1I1v2": 10.0,
            f"{prefix}_O1I1k2": -12.0,
            f"{prefix}_I1C1b1": 0.2,
            f"{prefix}_I1C1v1": -65.0,
            f"{prefix}_I1C1k1": 10.0,
            f"{prefix}_C1I1b2": 0.2,
            f"{prefix}_C1I1v2": -65.0,
            f"{prefix}_C1I1k2": -11.0,
            f"{prefix}_I1I2b2": 0.022,
            f"{prefix}_I1I2v2": -25.0,
            f"{prefix}_I1I2k2": -5.0,
            f"{prefix}_I2I1b1": 0.0018,
            f"{prefix}_I2I1v1": -50.0,
            f"{prefix}_I2I1k1": 12.0,
            f"{prefix}_dist": 0.0,
            f"{prefix}_slowdown": 0.2,
            f"{prefix}_persist": 0.0,
            "eNa": 50.0,
            "celsius": 34.0,
        }
        self.channel_states = {
            f"{prefix}_C1": 1.0,
            f"{prefix}_O1": 0.0,
            f"{prefix}_I1": 0.0,
            f"{prefix}_I2": 0.0,
        }
        self.current_name = "i_Na"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        C1 = states[f"{prefix}_C1"]
        O1 = states[f"{prefix}_O1"]
        I1 = states[f"{prefix}_I1"]
        I2 = states[f"{prefix}_I2"]
        old = jnp.stack([C1, O1, I1, I2], axis=-1)
        system = jnp.eye(4, dtype=old.dtype) - dt * self.rate_matrix(v, params)
        new = jnp.linalg.solve(system, old[..., None]).squeeze(-1)
        new = jnp.maximum(new, 0.0)
        new = new / jnp.maximum(jnp.sum(new, axis=-1, keepdims=True), 1e-12)
        return {
            f"{prefix}_C1": new[..., 0],
            f"{prefix}_O1": new[..., 1],
            f"{prefix}_I1": new[..., 2],
            f"{prefix}_I2": new[..., 3],
        }

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_gbar"] * states[f"{prefix}_O1"] * (v - params["eNa"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        C1O1, O1C1, O1I1, I1O1, I1C1, C1I1, I1I2, I2I1 = self.rates(v, params)
        z = jnp.zeros_like(v)
        o = jnp.ones_like(v)
        rows = [
            jnp.stack([-(C1O1 + C1I1), O1C1, I1C1, z], axis=-1),
            jnp.stack([C1O1, -(O1C1 + O1I1), I1O1, z], axis=-1),
            jnp.stack([C1I1, O1I1, -(I1O1 + I1C1 + I1I2), I2I1], axis=-1),
            jnp.stack([o, o, o, o], axis=-1),
        ]
        mat = jnp.stack(rows, axis=-2)
        rhs = jnp.stack([z, z, z, o], axis=-1)
        ss = jnp.linalg.solve(mat, rhs[..., None]).squeeze(-1)
        ss = jnp.maximum(ss, 0.0)
        ss = ss / jnp.maximum(jnp.sum(ss, axis=-1, keepdims=True), 1e-12)
        return {
            f"{prefix}_C1": ss[..., 0],
            f"{prefix}_O1": ss[..., 1],
            f"{prefix}_I1": ss[..., 2],
            f"{prefix}_I2": ss[..., 3],
        }

    def rates2(self, v, b, vv, k):
        return b / (1.0 + safe_exp((v - vv) / k))

    def rate_matrix(self, v, params):
        C1O1, O1C1, O1I1, I1O1, I1C1, C1I1, I1I2, I2I1 = self.rates(v, params)
        z = jnp.zeros_like(C1O1)
        rows = [
            jnp.stack([-(C1O1 + C1I1), O1C1, I1C1, z], axis=-1),
            jnp.stack([C1O1, -(O1C1 + O1I1), I1O1, z], axis=-1),
            jnp.stack([C1I1, O1I1, -(I1O1 + I1C1 + I1I2), I2I1], axis=-1),
            jnp.stack([z, z, I1I2, -I2I1], axis=-1),
        ]
        return jnp.stack(rows, axis=-2)

    def rates(self, v, params):
        prefix = channel_prefix(self)
        q10 = 3.0 ** ((params["celsius"] - 20.0) / 10.0)
        C1O1 = q10 * self.rates2(v, params[f"{prefix}_C1O1b2"], params[f"{prefix}_C1O1v2"], params[f"{prefix}_C1O1k2"])
        O1C1 = q10 * (
            self.rates2(v, params[f"{prefix}_O1C1b1"], params[f"{prefix}_O1C1v1"], params[f"{prefix}_O1C1k1"])
            + self.rates2(v, params[f"{prefix}_O1C1b2"], params[f"{prefix}_O1C1v2"], params[f"{prefix}_O1C1k2"])
        )
        O1I1 = 0.5 * q10 * (
            self.rates2(v, params[f"{prefix}_O1I1b1"], params[f"{prefix}_O1I1v1"], params[f"{prefix}_O1I1k1"])
            + self.rates2(v, params[f"{prefix}_O1I1b2"], params[f"{prefix}_O1I1v2"], params[f"{prefix}_O1I1k2"])
        )
        I1O1 = params[f"{prefix}_persist"] * O1I1
        I1C1 = q10 * self.rates2(v, params[f"{prefix}_I1C1b1"], params[f"{prefix}_I1C1v1"], params[f"{prefix}_I1C1k1"])
        C1I1 = q10 * self.rates2(v, params[f"{prefix}_C1I1b2"], params[f"{prefix}_C1I1v2"], params[f"{prefix}_C1I1k2"])
        I1I2 = params[f"{prefix}_slowdown"] * params[f"{prefix}_dist"] * q10 * self.rates2(
            v, params[f"{prefix}_I1I2b2"], params[f"{prefix}_I1I2v2"], params[f"{prefix}_I1I2k2"]
        )
        I2I1 = params[f"{prefix}_slowdown"] * q10 * self.rates2(
            v, params[f"{prefix}_I2I1b1"], params[f"{prefix}_I2I1v1"], params[f"{prefix}_I2I1k1"]
        )
        return C1O1, O1C1, O1I1, I1O1, I1C1, C1I1, I1I2, I2I1

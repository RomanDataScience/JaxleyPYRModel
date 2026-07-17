from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jaxley.pumps import Pump

from channels_converted.validation.channel_registry import ChannelSpec
from channels_converted.validation.protocols import VoltageProtocol


COMMON_DEFAULTS = {
    "eNa": 55.0,
    "eK": -85.0,
    "eCa": 140.0,
    "CaCon_i": 1e-4,
    "CaCon_e": 2.0,
    "i_Ca": 0.0,
    "celsius": 34.0,
}


def scalar(value) -> float:
    return float(np.asarray(value))


def build_jaxley_params(
    spec: ChannelSpec,
    param_overrides: Mapping[str, float] | None = None,
) -> dict[str, jnp.ndarray]:
    channel = spec.jaxley_class()
    params = {key: jnp.asarray(value, dtype=float) for key, value in channel.channel_params.items()}

    for key, value in COMMON_DEFAULTS.items():
        params.setdefault(key, jnp.asarray(value, dtype=float))

    if param_overrides:
        prefix = channel.name
        for key, value in param_overrides.items():
            full_key = key if key in params else f"{prefix}_{key}"
            params[full_key] = jnp.asarray(value, dtype=float)

    return params


def _initial_states(channel, params: Mapping[str, jnp.ndarray], voltage: float, dt: float):
    states = {
        key: jnp.asarray(value, dtype=float)
        for key, value in channel.channel_states.items()
    }
    for key, value in COMMON_DEFAULTS.items():
        states.setdefault(key, jnp.asarray(value, dtype=float))

    initialized = channel.init_state(
        states,
        jnp.asarray(voltage, dtype=float),
        params,
        dt,
    )
    states.update(initialized)
    return states


def _record(spec: ChannelSpec, channel, states, params, voltage: float, output, index: int):
    prefix = channel.name
    for state in spec.states:
        key = state.jaxley_var(prefix)
        if key not in states:
            raise KeyError(f"Jaxley state '{key}' not found for channel '{spec.key}'.")
        output[state.label][index] = scalar(states[key])

    if spec.current is not None:
        current = channel.compute_current(states, jnp.asarray(voltage, dtype=float), params)
        output[spec.current.label][index] = scalar(current)


def run_jaxley_channel(
    spec: ChannelSpec,
    protocol: VoltageProtocol,
    *,
    param_overrides: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    channel = spec.jaxley_class()
    params = build_jaxley_params(spec, param_overrides)
    states = _initial_states(channel, params, protocol.voltage[0], protocol.dt)

    output = {state.label: np.zeros_like(protocol.time) for state in spec.states}
    if spec.current is not None:
        output[spec.current.label] = np.zeros_like(protocol.time)

    _record(spec, channel, states, params, protocol.voltage[0], output, 0)

    for index in range(1, protocol.time.size):
        voltage = jnp.asarray(protocol.voltage[index], dtype=float)
        updated = channel.update_states(states, protocol.dt, voltage, params)
        states.update(updated)

        if isinstance(channel, Pump):
            ion_name = channel.ion_name
            d_ion_dt = -channel.compute_current(states, states[ion_name], params)
            states[ion_name] = jnp.maximum(states[ion_name] + protocol.dt * d_ion_dt, 1e-12)

        _record(spec, channel, states, params, protocol.voltage[index], output, index)

    return output

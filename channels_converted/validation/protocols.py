from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VoltageProtocol:
    time: np.ndarray
    voltage: np.ndarray
    dt: float


def voltage_step_protocol(
    *,
    dt: float = 0.025,
    hold_mv: float = -80.0,
    hold_ms: float = 100.0,
    step_ms: float = 100.0,
    tail_ms: float = 50.0,
    step_mvs: Iterable[float] = (-90.0, -70.0, -50.0, -30.0, -10.0, 10.0, 30.0),
) -> VoltageProtocol:
    """Build a shared clamp protocol for channel validation."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    segments: list[tuple[float, float]] = [(hold_mv, hold_ms)]
    segments.extend((float(step), step_ms) for step in step_mvs)
    segments.append((hold_mv, tail_ms))

    voltage_values: list[float] = []
    for value, duration in segments:
        steps = max(1, int(round(duration / dt)))
        voltage_values.extend([float(value)] * steps)
    voltage_values.append(float(segments[-1][0]))

    voltage = np.asarray(voltage_values, dtype=float)
    time = np.arange(voltage.size, dtype=float) * dt
    return VoltageProtocol(time=time, voltage=voltage, dt=float(dt))

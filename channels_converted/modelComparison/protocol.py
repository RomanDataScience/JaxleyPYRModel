from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepProtocol:
    dt: float = 0.025
    tstop: float = 500.0
    delay: float = 100.0
    duration: float = 300.0
    amplitude: float = 0.5
    v_init: float = -72.0

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.tstop <= 0:
            raise ValueError("tstop must be positive.")
        if self.delay < 0:
            raise ValueError("delay must be non-negative.")
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        if self.delay + self.duration > self.tstop:
            raise ValueError("delay + duration must fit inside tstop.")


def step_current(protocol: StepProtocol) -> tuple[np.ndarray, np.ndarray]:
    """Return time points and the per-integration-step stimulus in nA."""
    protocol.validate()

    n_steps = int(round(protocol.tstop / protocol.dt))
    time = np.arange(n_steps + 1, dtype=float) * protocol.dt
    current = np.zeros(n_steps, dtype=float)

    start = int(round(protocol.delay / protocol.dt))
    stop = int(round((protocol.delay + protocol.duration) / protocol.dt))
    current[start:stop] = protocol.amplitude

    return time, current


def current_at_time_points(time: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Pad a step-wise current vector so it can be plotted against voltage time."""
    plotted = np.zeros_like(time, dtype=float)
    if current.size:
        plotted[: min(current.size, plotted.size)] = current[: plotted.size]
        if plotted.size > current.size:
            plotted[current.size :] = current[-1]
    return plotted

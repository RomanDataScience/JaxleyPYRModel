"""Prepare trace-specific initial-state pytrees outside compiled simulation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class InitialStateFactory:
    """Initialize gates independently for every trace's initial voltage.

    Jaxley's ``init_states`` mutates host-side node tables and is intentionally
    called before JIT. The resulting state dictionaries are stacked into a
    pytree that can safely be mapped along the trace axis.
    """

    cell: object
    dt_ms: float

    def build(self, voltages_mV):
        original_voltage = self.cell.nodes["v"].to_numpy(dtype=float).copy()
        states = []
        for voltage in np.asarray(voltages_mV, dtype=float):
            self.cell.set("v", float(voltage))
            self.cell.init_states(delta_t=self.dt_ms)
            self.cell.to_jax()
            baseline_parameters = self.cell.get_all_parameters([])
            state = self.cell.get_all_states([])
            # Supplying ``all_states`` to ``jx.integrate`` makes the caller
            # responsible for including channel-current state entries too.
            # Without this step Jaxley receives ``None`` for current-dependent
            # channel states on the first solver iteration.
            states.append(
                self.cell.append_channel_currents_to_states(
                    state, baseline_parameters, self.dt_ms
                )
            )
        self.cell.set("v", original_voltage)
        self.cell.init_states(delta_t=self.dt_ms)
        self.cell.to_jax()
        return jax.tree.map(lambda *values: jnp.stack(values), *states)

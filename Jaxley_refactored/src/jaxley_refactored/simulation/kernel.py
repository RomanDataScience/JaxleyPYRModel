"""Serial and ``jit(vmap(...))`` simulation backbone."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import jaxley as jx

from jaxley_refactored.config.schema import ProtocolSpec, RuntimeSpec
from jaxley_refactored.parameters import Parameterizer

from .sites import resolve_site


@dataclass
class SimulationKernel:
    """Pure numerical interface around one statically assembled Jaxley cell."""

    cell: object
    parameterizer: Parameterizer
    protocol: ProtocolSpec
    runtime: RuntimeSpec
    dt_ms: float
    n_steps: int

    def __post_init__(self):
        self.cell.delete_stimuli()
        self.cell.delete_recordings()
        self._injection_site = resolve_site(self.cell, self.protocol.injection_site)
        recording = resolve_site(self.cell, self.protocol.recording_site)
        recording.record(self.protocol.recording_site.state, verbose=False)
        checkpoint = max(
            1,
            int(
                math.ceil(
                    self.n_steps
                    ** (1.0 / max(1, self.runtime.checkpoint_levels))
                )
            ),
        )
        self._checkpoint_lengths = [checkpoint] * max(
            1, self.runtime.checkpoint_levels
        )

    def _align(self, voltage):
        if self.protocol.alignment == "drop_initial":
            voltage = voltage[..., 1:]
        if voltage.shape[-1] < self.n_steps:
            raise ValueError(
                f"Simulation returned {voltage.shape[-1]} samples, "
                f"expected at least {self.n_steps}."
            )
        return voltage[..., : self.n_steps]

    def simulate_one(self, physical_parameters, current_nA, initial_states):
        parameter_state = self.parameterizer.state(physical_parameters)
        data_stimuli = self._injection_site.data_stimulate(current_nA, None)
        voltage = jx.integrate(
            self.cell,
            param_state=parameter_state,
            data_stimuli=data_stimuli,
            all_states=initial_states,
            delta_t=self.dt_ms,
            solver=self.runtime.solver,
            voltage_solver=self.runtime.voltage_solver,
            checkpoint_lengths=self._checkpoint_lengths,
        )[0]
        return self._align(voltage)

    def simulate_batch(self, physical_parameters, currents_nA, initial_states):
        if self.runtime.jit:
            return self.compiled_batch(physical_parameters, currents_nA, initial_states)
        if self.runtime.backend == "cpu" and currents_nA.shape[0] == 1:
            return self.simulate_one(
                physical_parameters,
                currents_nA[0],
                jax.tree.map(lambda value: value[0], initial_states),
            )[None, :]
        return self._mapped(physical_parameters, currents_nA, initial_states)

    def simulate_serial(self, physical_parameters, currents_nA, initial_states):
        """Map sequentially for a lower-memory correctness/reference path."""
        mapped = getattr(self, "_compiled_serial", None)

        def serial(parameters, currents, states):
            return jax.lax.map(
                lambda inputs: self.simulate_one(parameters, inputs[0], inputs[1]),
                (currents, states),
            )

        if self.runtime.jit:
            if mapped is None:
                mapped = jax.jit(serial)
                self._compiled_serial = mapped
            return mapped(physical_parameters, currents_nA, initial_states)
        return serial(physical_parameters, currents_nA, initial_states)

    def _mapped(self, physical_parameters, currents_nA, initial_states):
        if self.runtime.jit or currents_nA.shape[0] > 1:
            return jax.vmap(self.simulate_one, in_axes=(None, 0, 0))(
                physical_parameters,
                currents_nA,
                initial_states,
            )
        outputs = [
            self.simulate_one(
                physical_parameters,
                current,
                jax.tree.map(lambda value: value[index], initial_states),
            )
            for index, current in enumerate(currents_nA)
        ]
        return jnp.stack(outputs)

    @property
    def compiled_batch(self):
        """Compile lazily so model inspection does not pay compilation cost."""
        compiled = getattr(self, "_compiled_batch", None)
        if compiled is None:
            compiled = jax.jit(
                jax.vmap(self.simulate_one, in_axes=(None, 0, 0))
            )
            self._compiled_batch = compiled
        return compiled

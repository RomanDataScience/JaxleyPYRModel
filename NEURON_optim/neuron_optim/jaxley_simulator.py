"""Jaxley-backed Combe simulation for the optimization pipeline."""

from __future__ import annotations

import numpy as np

from .data import Trace
from .objective import SimulationOutput


class JaxleySimulator:
    """Simulate Combe traces with the repository's Jaxley model.

    The SWC morphology is used deliberately: the exact HOC-topology builder
    bootstraps through NEURON, which would make a purported Jaxley backend
    depend on NEURON during model construction.
    """

    def __init__(self, *, d_lambda: float = 0.3, quiet: bool = True):
        del quiet  # Jaxley has no equivalent simulator verbosity switch here.
        try:
            import jax.numpy as jnp
            import jaxley as jx
            from JaxleyModel.model.model_Combe import (
                SUPPORTED_FIT_PARAMETER_KEYS,
                Combe2023,
                set_fitted_parameters,
            )
        except Exception as exc:  # pragma: no cover - depends on Jaxley environment
            raise RuntimeError(
                "Jaxley is required for the Jaxley backend. Activate the Jaxley "
                "environment and install the repository model dependencies."
            ) from exc
        self.d_lambda = d_lambda
        self.jnp = jnp
        self.jx = jx
        self._build_model = Combe2023
        self._set_fitted_parameters = set_fitted_parameters
        self._supported_keys = frozenset(SUPPORTED_FIT_PARAMETER_KEYS)

    def _model(self, values: dict[str, float]):
        cell = self._build_model(
            d_lambda=self.d_lambda,
            morphology_source="swc",
        )
        keys = tuple(key for key in values if key in self._supported_keys)
        fitted = self.jnp.asarray([values[key] for key in keys], dtype=self.jnp.float64)
        # Jaxley parameter updates are functional: data_set() accumulates
        # updates in a parameter-state object instead of mutating the cell.
        # Keep that state and pass it to integrate() below.
        param_state = self._set_fitted_parameters(cell, keys, fitted)
        return cell, param_state

    def simulate_many(self, traces: list[Trace], values: dict[str, float], *,
                      v_init_mode: str = "observed_first_sample") -> list[SimulationOutput]:
        outputs = []
        cell, param_state = self._model(values)
        for trace in traces:
            if trace.time_ms.size < 2:
                raise ValueError("Trace has fewer than two samples")
            dt = float(np.median(np.diff(trace.time_ms)))
            if dt <= 0.0 or not np.isfinite(dt):
                raise ValueError("Trace time step must be finite and positive")
            time = np.asarray(trace.time_ms - trace.time_ms[0], dtype=float)
            current = np.asarray(trace.current_nA, dtype=float)
            cell.delete_stimuli()
            cell.delete_recordings()
            cell.soma.branch(0).loc(0.5).stimulate(
                self.jnp.asarray(current, dtype=self.jnp.float64)
            )
            cell.soma.branch(0).loc(0.5).record()
            initial = (float(trace.voltage_mV[0])
                       if v_init_mode == "observed_first_sample"
                       else float(values["Epas"]))
            cell.set("v", initial)
            cell.init_states()
            voltage = np.asarray(
                self.jx.integrate(cell, param_state=param_state, delta_t=dt)[0],
                dtype=float,
            )
            n = min(time.size, voltage.size)
            if n < 2 or not np.isfinite(voltage[:n]).all():
                raise RuntimeError("Jaxley returned an invalid voltage trace")
            outputs.append(SimulationOutput(time[:n], voltage[:n]))
        return outputs

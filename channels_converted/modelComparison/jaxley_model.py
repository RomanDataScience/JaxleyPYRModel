from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".8")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jaxley_mpl")

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import jaxley as jx  # noqa: E402
import pandas as pd  # noqa: E402

from channels_converted.modelComparison.protocol import (  # noqa: E402
    StepProtocol,
    current_at_time_points,
    step_current,
)


warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
JAXLEY_MODEL_DIR = REPO_ROOT / "JaxleyModel" / "model"

if str(JAXLEY_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(JAXLEY_MODEL_DIR))

from model_Combe import COMBE_PARAMS, Combe2023  # noqa: E402


def combe_kinetic_scales() -> dict[tuple[str, str], float]:
    """Kinetic scales of the Jaxley Combe port, as (mechanism, attribute) pairs.

    The attribute names match the patched NEURON mechanisms
    (neuron_optim.mechanisms), so both backends can be given the same values.
    """
    p = COMBE_PARAMS
    return {
        ("kd", "deactivation_tau_scale"): p.kd_deactivation_tau_scale,
        ("na16a", "fast_inactivation_tau_scale"): p.nat_fast_inactivation_tau_scale,
        ("na16a", "slow_recovery_tau_scale"): p.nat_slow_recovery_tau_scale,
        ("nax", "fast_inactivation_tau_scale"): p.nat_fast_inactivation_tau_scale,
        ("na3dend", "fast_inactivation_tau_scale"): p.nat_fast_inactivation_tau_scale,
        ("na12", "fast_inactivation_tau_scale"): p.nat_fast_inactivation_tau_scale,
        ("h", "tau_scale"): p.h_tau_scale,
    }


def build_combe_jaxley_model(*, d_lambda: float = 0.1):
    """Return the one-to-one Combe channel-placement port in Jaxley."""
    return Combe2023(d_lambda=d_lambda)


def run_jaxley_step(
    protocol: StepProtocol,
    *,
    d_lambda: float = 0.1,
    na12: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    time, current = step_current(protocol)
    stimulus = jnp.asarray(current, dtype=jnp.float64)

    cell = build_combe_jaxley_model(d_lambda=d_lambda)
    if na12 is not None:
        # Same values as the NEURON side (run_neuron_step's na12 argument).
        for attr, value in na12.items():
            cell.soma.set(f"na12_{attr}", float(value))
    # Re-assert the port's kinetic scales: when the patched NEURON mechanisms
    # are loaded, the HOC import reads their default of 1 and would override them.
    for (mechanism, attr), value in combe_kinetic_scales().items():
        key = f"{mechanism}_{attr}"
        if key in cell.nodes.columns:
            cell.set(key, float(value))
    cell.delete_stimuli()
    cell.delete_recordings()
    cell.soma.branch(0).loc(0.5).stimulate(stimulus)
    cell.soma.branch(0).loc(0.5).record()
    cell.set("v", protocol.v_init)
    cell.init_states()

    voltage = np.asarray(jx.integrate(cell, delta_t=protocol.dt)[0], dtype=float)
    plotted_current = current_at_time_points(time, current)

    n = min(time.size, voltage.size, plotted_current.size)
    return {
        "time": time[:n],
        "voltage": voltage[:n],
        "current": plotted_current[:n],
    }

"""Cross-backend parity checks for the first configured current-clamp pulses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig
from .data import Trace, load_protocol_traces
from .parameters import DEFAULTS
from .simulator import make_simulator


def _error_metrics(trace: Trace, neuron_voltage: np.ndarray,
                   jaxley_voltage: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    error = jaxley_voltage[mask] - neuron_voltage[mask]
    if error.size == 0:
        raise ValueError(f"Comparison window is empty for {trace.trace} {trace.protocol}")
    return {
        "rmse_mV": float(np.sqrt(np.mean(error ** 2))),
        "mae_mV": float(np.mean(np.abs(error))),
        "max_abs_mV": float(np.max(np.abs(error))),
    }


def _compare_trace(trace: Trace, neuron_simulation, jaxley_simulation) -> dict[str, Any]:
    target_time = np.asarray(trace.time_ms, dtype=float)
    neuron_voltage = np.interp(
        target_time, neuron_simulation.time_ms, neuron_simulation.voltage_mV
    )
    jaxley_voltage = np.interp(
        target_time, jaxley_simulation.time_ms, jaxley_simulation.voltage_mV
    )
    pulse = ((target_time >= trace.epoch_start_ms) &
             (target_time <= trace.epoch_stop_ms))
    finite = np.isfinite(neuron_voltage) & np.isfinite(jaxley_voltage)
    if not finite.all():
        raise RuntimeError(f"Non-finite backend output for {trace.trace} {trace.protocol}")
    return {
        "trace": trace.trace,
        "protocol": trace.protocol,
        "samples": int(target_time.size),
        "epoch_start_ms": float(trace.epoch_start_ms),
        "epoch_stop_ms": float(trace.epoch_stop_ms),
        "initial_voltage_mV": float(trace.voltage_mV[0]),
        "current_pre_median_nA": float(np.median(
            trace.current_nA[target_time <= trace.epoch_start_ms]
        )),
        "current_pulse_median_nA": float(np.median(trace.current_nA[pulse])),
        "input_dt_ms": float(np.median(np.diff(target_time))),
        "neuron_output_dt_ms": float(np.median(np.diff(neuron_simulation.time_ms))),
        "jaxley_output_dt_ms": float(np.median(np.diff(jaxley_simulation.time_ms))),
        "full_trace": _error_metrics(
            trace, neuron_voltage, jaxley_voltage, np.ones(target_time.size, dtype=bool)
        ),
        "pulse_window": _error_metrics(trace, neuron_voltage, jaxley_voltage, pulse),
    }


def compare_first_pulses(config: RunConfig, output_path: Path | None = None) -> dict[str, Any]:
    """Compare both backends with one shared parameter mapping.

    The first configured trace is used for both protocols.  The comparison is
    deliberately independent of the configured optimization backend: it is a
    diagnostic of backend parity, not an objective evaluation.
    """
    root = config.data_root
    hyper_trace = load_protocol_traces(
        root, cell=config.cell, protocol="hyperpolarizing_pulse",
        trace_names=(config.hyperpolarizing_trace_names[0],),
        pre_ms=config.simulation_pre_ms, post_ms=config.simulation_post_ms,
        full_trial=True, center_current=False,
    )[0]
    depolarizing_trace = load_protocol_traces(
        root, cell=config.cell, protocol="depolarizing_step",
        trace_names=(config.trace_names[0],),
        pre_ms=config.simulation_pre_ms, post_ms=config.simulation_post_ms,
        full_trial=True, center_current=False,
    )[0]

    traces = [hyper_trace, depolarizing_trace]
    values = dict(DEFAULTS)
    # Run all NEURON traces before constructing Jaxley's HOC-derived model;
    # building that model clears the process-global NEURON section tree.
    neuron_simulator = make_simulator(
        "neuron", d_lambda=config.d_lambda, quiet=True, force_recompile=True
    )
    neuron_outputs = neuron_simulator.simulate_many(traces, values)
    jaxley_simulator = make_simulator(
        "jaxley", d_lambda=config.d_lambda, quiet=True, morphology_source="hoc"
    )
    jaxley_outputs = jaxley_simulator.simulate_many(traces, values)

    comparisons = [
        _compare_trace(trace, neuron_output, jaxley_output)
        for trace, neuron_output, jaxley_output in zip(
            traces, neuron_outputs, jaxley_outputs, strict=True
        )
    ]
    report: dict[str, Any] = {
        "parameters": values,
        "backend_pair": ["neuron", "jaxley"],
        "morphology_source": "hoc",
        "d_lambda": float(config.d_lambda),
        "numerics": {
            "neuron": {
                "cvode_active": False,
                "secondorder": 0,
                "method": "fixed-step backward Euler",
            },
            "jaxley": {
                "solver": "bwd_euler",
                "voltage_solver": "jaxley.dhs.cpu",
                "stimulus": "linear-play interval averages",
            },
        },
        "comparisons": comparisons,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

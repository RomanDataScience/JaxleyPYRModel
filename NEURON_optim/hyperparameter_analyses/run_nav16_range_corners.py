"""Run all-lower/all-upper optimizer corners on the first two protocols.

This is a diagnostic, not an optimization run.  It evaluates the complete
NEURON_optim parameter space at its two opposite corners on v75ctrl's first
hyperpolarizing and depolarizing traces, then records voltage extrema and
threshold-crossing spike counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from neuron_optim.data import Trace, crop_trace, load_protocol_traces
from neuron_optim.parameters import DEFAULTS, make_parameter_space
from neuron_optim.simulator import make_simulator


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "JaxleyModel" / "Experimental_currentClamp_Analysis" / "Segmented_Traces"
PERSISTENT_KEYS = ("persist", "nav16_persist_vhalf", "nav16_persist_k")


def _corner_values(space, upper: bool) -> dict[str, float]:
    values = dict(DEFAULTS)
    edge = space.upper if upper else space.lower
    values.update({key: float(value) for key, value in zip(space.keys, edge, strict=True)})
    return values


def _summary(trace, simulation, label: str) -> dict:
    time = np.asarray(simulation.time_ms, dtype=float)
    voltage = np.asarray(simulation.voltage_mV, dtype=float)
    peaks, properties = find_peaks(voltage, height=-20.0, prominence=5.0, distance=40)
    epoch = (time >= trace.epoch_start_ms) & (time <= trace.epoch_stop_ms)
    return {
        "corner": label,
        "trace": trace.trace,
        "protocol": trace.protocol,
        "epoch_start_ms": float(trace.epoch_start_ms),
        "epoch_stop_ms": float(trace.epoch_stop_ms),
        "min_mV": float(np.min(voltage)),
        "max_mV": float(np.max(voltage)),
        "epoch_min_mV": float(np.min(voltage[epoch])),
        "epoch_max_mV": float(np.max(voltage[epoch])),
        "spike_count_total": int(peaks.size),
        "spike_count_epoch": int(np.count_nonzero(epoch[peaks])),
        "spike_peak_mV": voltage[peaks].tolist(),
        "spike_time_ms": time[peaks].tolist(),
    }


def _coarsen(trace: Trace, stride: int) -> Trace:
    if stride <= 1:
        return trace
    indices = np.arange(0, trace.time_ms.size, stride, dtype=int)
    if indices[-1] != trace.time_ms.size - 1:
        indices = np.append(indices, trace.time_ms.size - 1)
    return Trace(
        cell=trace.cell,
        trace=trace.trace,
        protocol=trace.protocol,
        time_ms=trace.time_ms[indices],
        voltage_mV=trace.voltage_mV[indices],
        current_nA=trace.current_nA[indices],
        epoch_start_ms=trace.epoch_start_ms,
        epoch_stop_ms=trace.epoch_stop_ms,
    )


def run(backend: str, output: Path, stride: int = 1, scope: str = "full") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    space = make_parameter_space(
        include=PERSISTENT_KEYS if scope == "persistent" else None
    )
    raw_traces = {
        "hyperpolarizing_pulse": load_protocol_traces(
            DATA_ROOT, cell="m20240527cd", protocol="hyperpolarizing_pulse",
            trace_names=("v75ctrl",), full_trial=True,
        )[0],
        "depolarizing_step": load_protocol_traces(
            DATA_ROOT, cell="m20240527cd", protocol="depolarizing_step",
            trace_names=("v75ctrl",), full_trial=True,
        )[0],
    }
    # The requested diagnostic concerns the first pulse. Keep the complete
    # pre-pulse interval, the pulse, and 100 ms of recovery; this preserves the
    # relevant state preparation while avoiding long late-trial tails.
    traces = {
        protocol: crop_trace(
            trace, start_ms=0.0,
            stop_ms=min(float(trace.time_ms[-1]), float(trace.epoch_stop_ms) + 100.0),
        )
        for protocol, trace in raw_traces.items()
    }
    traces = {protocol: _coarsen(trace, stride) for protocol, trace in traces.items()}
    simulator = make_simulator(backend, d_lambda=0.3, quiet=True)
    results = []
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=True)
    for row, (protocol, trace) in enumerate(traces.items()):
        for col, upper in enumerate((False, True)):
            label = "upper" if upper else "lower"
            values = _corner_values(space, upper)
            simulation = simulator.simulate_many([trace], values)[0]
            results.append(_summary(trace, simulation, f"{protocol}_{label}"))
            ax = axes[row, col]
            ax.plot(trace.time_ms, trace.voltage_mV, color="black", lw=0.8, alpha=0.7, label="data")
            ax.plot(simulation.time_ms, simulation.voltage_mV, color="tab:red", lw=0.8, label="corner")
            ax.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="tab:blue", alpha=0.08)
            ax.set_title(f"{protocol}: {label} bounds")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("V (mV)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    figure.tight_layout()
    figure.savefig(output / f"nav16_range_corners_{scope}_{backend}.png", dpi=140)
    plt.close(figure)
    payload = {
        "backend": backend,
        "scope": scope,
        "parameter_keys": list(space.keys),
        "lower_corner": {key: float(value) for key, value in zip(space.keys, space.lower, strict=True)},
        "upper_corner": {key: float(value) for key, value in zip(space.keys, space.upper, strict=True)},
        "results": results,
    }
    (output / f"nav16_range_corners_{scope}_{backend}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("neuron", "jaxley"), default="neuron")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth input sample for diagnostic runs.")
    parser.add_argument("--scope", choices=("full", "persistent"), default="full",
                        help="Vary all optimizer parameters or only persistent-Nav16 parameters.")
    parser.add_argument("--output", type=Path, default=ROOT / "NEURON_optim" / "runs" / "nav16_range_corners")
    args = parser.parse_args()
    payload = run(args.backend, args.output, stride=max(1, args.stride), scope=args.scope)
    for result in payload["results"]:
        print(
            result["corner"],
            f"epoch=[{result['epoch_min_mV']:.2f}, {result['epoch_max_mV']:.2f}] mV",
            f"spikes={result['spike_count_epoch']}",
        )


if __name__ == "__main__":
    main()

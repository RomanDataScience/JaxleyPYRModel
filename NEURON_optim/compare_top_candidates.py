"""Compare archived candidates with the Jaxley and NEURON backends.

The input candidate file is the ``top_candidates.json`` written by the
generation plotter.  Candidates are re-evaluated with the same passive-only
parameter completion, traces, initial condition, and objective settings used
by the calibration run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neuron_optim.config import load_config
from neuron_optim.data import load_protocol_traces
from neuron_optim.objective import SimulationOutput, hyperpolarizing_objective
from neuron_optim.parameters import PASSIVE, passive_model_values
from neuron_optim.simulator import make_simulator
from neuron_optim.stages import _objective_options


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n",
                    encoding="utf-8")


def _json_default(value: object):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _interpolated(simulation: SimulationOutput, time_ms: np.ndarray) -> np.ndarray:
    return np.interp(time_ms, simulation.time_ms, simulation.voltage_mV)


def _plot_candidate(output: Path, rank: int, original_loss: float,
                    neuron_loss: float, traces, neuron_simulations) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(traces), 1, figsize=(11, 2.7 * len(traces)),
                                sharex=False, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, trace, neuron_sim in zip(axes, traces, neuron_simulations, strict=True):
        axis.plot(trace.time_ms, trace.voltage_mV, color="black", linewidth=0.8,
                  label="v_exp")
        axis.plot(trace.time_ms, _interpolated(neuron_sim, trace.time_ms),
                  color="#d62728", linewidth=0.8, label="v_neuron")
        axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms,
                     color="#9ecae1", alpha=0.25)
        axis.set_ylabel("mV")
        axis.set_title(f"{trace.trace} ({trace.protocol})")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time from simulation window start (ms)")
    figure.suptitle(
        f"passive NEURON replay: archived Jaxley rank {rank}; "
        f"Jaxley loss {original_loss:.6g}; NEURON loss {neuron_loss:.6g}"
    )
    figure.savefig(output, dpi=120)
    plt.close(figure)


def compare(candidates_path: Path, config_path: Path, output_dir: Path) -> None:
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if len(candidates) != 10:
        raise ValueError(f"Expected 10 candidates, found {len(candidates)}")
    if [item["rank"] for item in candidates] != list(range(1, 11)):
        raise ValueError("Candidate ranks must be exactly 1 through 10")

    config = load_config(config_path)
    traces = load_protocol_traces(
        config.data_root,
        cell=config.cell,
        protocol="hyperpolarizing_pulse",
        trace_names=config.trace_names,
        pre_ms=config.simulation_pre_ms,
        post_ms=config.simulation_post_ms,
        full_trial=True,
        center_current=False,
    )
    objective_options = _objective_options(config.raw, "passive")
    neuron_simulator = make_simulator("neuron", d_lambda=config.d_lambda, quiet=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detailed = []

    for candidate in candidates:
        rank = int(candidate["rank"])
        values = dict(zip(PASSIVE, candidate["physical"], strict=True))
        values = passive_model_values(values)

        print(f"evaluating archived rank {rank}/10", flush=True)
        neuron_simulations = neuron_simulator.simulate_many(traces, values)
        neuron_result = hyperpolarizing_objective(
            traces, neuron_simulations, **objective_options
        )

        trace_metrics = []
        for trace, neuron_sim in zip(traces, neuron_simulations, strict=True):
            neuron_values = _interpolated(neuron_sim, trace.time_ms)
            trace_metrics.append({
                "trace": trace.trace,
                "neuron_vs_experiment_mae_mV": float(
                    np.mean(np.abs(neuron_values - trace.voltage_mV))
                ),
                "neuron_vs_experiment_rmse_mV": float(
                    np.sqrt(np.mean((neuron_values - trace.voltage_mV) ** 2))
                ),
                "neuron_vs_experiment_max_abs_mV": float(
                    np.max(np.abs(neuron_values - trace.voltage_mV))
                ),
            })

        _plot_candidate(
            plot_dir / f"rank_{rank:02d}.png", rank,
            float(candidate["loss"]), neuron_result.value, traces,
            neuron_simulations,
        )
        summary_rows.append({
            "archived_rank": rank,
            "population_index": int(candidate["population_index"]),
            "archived_jaxley_loss": float(candidate["loss"]),
            "neuron_loss": float(neuron_result.value),
            "neuron_minus_archived_jaxley": float(neuron_result.value - candidate["loss"]),
            "mean_neuron_vs_experiment_mae_mV": float(np.mean([
                item["neuron_vs_experiment_mae_mV"] for item in trace_metrics
            ])),
            "max_neuron_vs_experiment_abs_mV": float(np.max([
                item["neuron_vs_experiment_max_abs_mV"] for item in trace_metrics
            ])),
        })
        detailed.append({
            "archived_candidate": candidate,
            "neuron_loss": neuron_result.value,
            "neuron_details": neuron_result.details,
            "trace_backend_metrics": trace_metrics,
        })

    summary_rows.sort(key=lambda row: row["neuron_loss"])
    for neuron_rank, row in enumerate(summary_rows, start=1):
        row["neuron_rank"] = neuron_rank
    summary_rows.sort(key=lambda row: row["archived_rank"])

    fieldnames = list(summary_rows[0])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_json(output_dir / "summary.json", summary_rows)
    _write_json(output_dir / "details.json", detailed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.candidates.resolve(), args.config.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

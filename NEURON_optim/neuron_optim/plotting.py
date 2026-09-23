"""Generation-level voltage comparison plots for the best candidates.

The input traces are the same prepared windows used by the objective. Plotting
clips simulator outputs to those trace bounds so figures never show samples
outside the fitness interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import Trace
from .objective import SimulationOutput
from .parameters import ParameterSpace


def plot_generation(*, output_dir: Path, generation: int, stage: str,
                    traces: list[Trace], population: np.ndarray,
                    losses: np.ndarray, simulations: list[list[SimulationOutput]],
                    space: ParameterSpace, top_k: int = 10, dpi: int = 120,
                    population_indices: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir / f"generation_{generation:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if population_indices is None:
        population_indices = np.arange(len(population), dtype=int)
    else:
        population_indices = np.asarray(population_indices, dtype=int)
        if population_indices.shape != (len(population),):
            raise ValueError("population_indices must match population length")
    order = np.argsort(losses, kind="stable")[:min(top_k, len(losses))]
    if not len(order):
        return
    # One figure per generation: rows are candidates and columns are traces.
    # This avoids opening/rendering one figure for every candidate while still
    # keeping each candidate's four protocol traces visually grouped.
    figure, axes = plt.subplots(
        len(order), len(traces),
        figsize=(4.2 * len(traces), 2.5 * len(order)),
        sharex=False, constrained_layout=True, squeeze=False,
    )
    metadata = []
    for rank, index in enumerate(order, start=1):
        for trace_index, (trace, simulation) in enumerate(
            zip(traces, simulations[index], strict=True)
        ):
            axis = axes[rank - 1, trace_index]
            # Trace objects are already cropped to the configured fitness
            # window. Clip both series explicitly so a backend that returns
            # extra trial samples cannot expand the plotted interval.
            fitness_start = float(trace.time_ms[0])
            fitness_stop = float(trace.time_ms[-1])
            experimental_mask = ((trace.time_ms >= fitness_start) &
                                 (trace.time_ms <= fitness_stop))
            simulated_mask = ((simulation.time_ms >= fitness_start) &
                              (simulation.time_ms <= fitness_stop))
            plot_trace_time = trace.time_ms[experimental_mask]
            plot_trace_voltage = trace.voltage_mV[experimental_mask]
            plot_sim_time = simulation.time_ms[simulated_mask]
            plot_sim_voltage = simulation.voltage_mV[simulated_mask]
            # Show absolute voltage so the separately scored baseline offset
            # remains visible alongside the Delta_V waveform shape.
            exp_voltage = plot_trace_voltage
            sim_voltage = plot_sim_voltage
            ylabel = "mV"
            axis.plot(plot_trace_time, exp_voltage, color="black", linewidth=0.8, label="v_exp")
            axis.plot(plot_sim_time, sim_voltage, color="#d62728", linewidth=0.8, label="v_sim")
            axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="#9ecae1", alpha=0.25)
            axis.set_ylabel(ylabel)
            axis.set_title(f"rank {rank} · {trace.trace}")
            axis.legend(loc="upper right", fontsize=7)
            axis.grid(alpha=0.2)
        axes[rank - 1, 0].set_ylabel(f"rank {rank}\nmV")
        for axis in axes[rank - 1, :]:
            axis.set_xlabel("Time from simulation window start (ms)")
        metadata.append({"rank": rank, "population_index": int(population_indices[index]),
                         "loss": float(losses[index]),
                         "normalized": population[index].tolist(),
                         "physical": space.physical(population[index]).tolist(),
                         "plot": "candidates.png"})
    figure.suptitle(f"{stage}: generation {generation} · top {len(order)} candidates")
    figure.savefig(output_dir / "candidates.png", dpi=dpi)
    plt.close(figure)
    (output_dir / "top_candidates.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

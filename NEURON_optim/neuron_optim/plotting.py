"""Generation-level voltage comparison plots for the best candidates."""

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
    metadata = []
    for rank, index in enumerate(order, start=1):
        figure, axes = plt.subplots(len(traces), 1, figsize=(11, 2.7 * len(traces)),
                                    sharex=False, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for axis, trace, simulation in zip(axes, traces, simulations[index], strict=True):
            if stage == "hyper":
                pre_exp = trace.time_ms <= trace.epoch_start_ms
                pre_sim = simulation.time_ms <= trace.epoch_start_ms
                exp_baseline = float(np.median(trace.voltage_mV[pre_exp]))
                sim_baseline = float(np.median(simulation.voltage_mV[pre_sim]))
                exp_voltage = trace.voltage_mV - exp_baseline
                sim_voltage = simulation.voltage_mV - sim_baseline
                ylabel = "ΔV (mV)"
            else:
                exp_voltage = trace.voltage_mV
                sim_voltage = simulation.voltage_mV
                ylabel = "mV"
            axis.plot(trace.time_ms, exp_voltage, color="black", linewidth=0.8, label="v_exp")
            axis.plot(simulation.time_ms, sim_voltage, color="#d62728", linewidth=0.8, label="v_sim")
            axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="#9ecae1", alpha=0.25)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{trace.trace} ({trace.protocol})")
            axis.legend(loc="upper right", fontsize=8)
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Time from simulation window start (ms)")
        figure.suptitle(f"{stage}: generation {generation}, rank {rank}, loss {losses[index]:.6g}")
        figure.savefig(output_dir / f"rank_{rank:02d}.png", dpi=dpi)
        plt.close(figure)
        metadata.append({"rank": rank, "population_index": int(population_indices[index]),
                         "loss": float(losses[index]),
                         "normalized": population[index].tolist(),
                         "physical": space.physical(population[index]).tolist(),
                         "plot": f"rank_{rank:02d}.png"})
    (output_dir / "top_candidates.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

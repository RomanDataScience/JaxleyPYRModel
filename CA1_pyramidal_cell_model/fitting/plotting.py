"""Plots written during a fit: best-fit traces and the top candidates per generation.

Taken from ``OptimizerNEURON_Combe/scripts/run_joint.py`` so this folder runs on
its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_plotter(depolarizing, hyperpolarizing, simulator):
    def plot(values: dict[str, float], path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        traces = [*depolarizing, hyperpolarizing]
        simulations = simulator(values)
        fig, axes = plt.subplots(len(traces), 1, figsize=(11, 10), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for axis, trace, (time, voltage) in zip(axes, traces, simulations, strict=True):
            axis.plot(trace.time_ms, trace.voltage_mV, color="black", lw=0.7, label="experiment")
            axis.plot(time, voltage, color="#0072B2", lw=0.8, label="NEURON")
            axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="#D55E00", alpha=0.12)
            if trace.protocol == "hyperpolarizing_pulse":
                axis.set_xlim(
                    max(float(trace.time_ms[0]), trace.epoch_start_ms - 400.0),
                    min(float(trace.time_ms[-1]), trace.epoch_stop_ms + 400.0),
                )
            axis.set_ylabel("mV")
            axis.set_title(f"{trace.name} — {trace.protocol}")
            axis.grid(True, color="0.9")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("time (ms)")
        fig.savefig(path, dpi=130)
        plt.close(fig)

        # Save a focused depolarization-onset view in addition to the complete
        # fit figure. The window is relative to each current-step onset.
        depolarizing_simulations = simulations[:len(depolarizing)]
        depolarizing_path = path.parent / "depolarizing_steps_fit.png"
        onset_figure, onset_axes = plt.subplots(
            len(depolarizing), 1, figsize=(11, 8),
            squeeze=False, constrained_layout=True,
        )
        for axis, trace, (time, voltage) in zip(
            onset_axes[:, 0], depolarizing, depolarizing_simulations, strict=True
        ):
            axis.plot(trace.time_ms, trace.voltage_mV, color="black", lw=0.7,
                      label="experiment")
            axis.plot(time, voltage, color="#0072B2", lw=0.8, label="NEURON")
            axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms,
                         color="#D55E00", alpha=0.12)
            axis.set_xlim(
                max(float(trace.time_ms[0]), trace.epoch_start_ms - 50.0),
                min(float(trace.time_ms[-1]), trace.epoch_start_ms + 150.0),
            )
            axis.set_ylabel("mV")
            axis.set_title(f"{trace.name} — depolarization onset")
            axis.grid(True, color="0.9")
        onset_axes[0, 0].legend(loc="upper right")
        onset_axes[-1, 0].set_xlabel("time (ms)")
        onset_figure.savefig(depolarizing_path, dpi=130)
        plt.close(onset_figure)
    return plot


def generation_plotter(depolarizing, hyperpolarizing, output_dir: Path, *, top_k: int, dpi: int,
                        title: str = "Combe joint optimization"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(generation: int, candidates, losses: np.ndarray, evaluated) -> None:
        order = np.argsort(losses, kind="stable")[:min(top_k, len(losses))]
        plot_dir = output_dir / "plots" / f"generation_{generation:04d}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        traces = [*depolarizing, hyperpolarizing]
        figure, axes = plt.subplots(
            len(order), len(traces), figsize=(16, max(2.2 * len(order), 4.0)),
            squeeze=False, constrained_layout=True,
        )
        for row, index in enumerate(order):
            for axis, trace, simulation in zip(
                axes[row], traces, evaluated[int(index)][2], strict=True
            ):
                axis.plot(trace.time_ms, trace.voltage_mV, color="black", lw=0.55)
                axis.plot(simulation[0], simulation[1], color="#D55E00", lw=0.65)
                axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="#56B4E9", alpha=0.16)
                axis.set_title(f"rank {row + 1} · {trace.name}", fontsize=8)
                axis.grid(True, color="0.9", lw=0.4)
                axis.tick_params(labelsize=7)
                if row == len(order) - 1:
                    axis.set_xlabel("time (ms)", fontsize=8)
            axes[row, 0].set_ylabel(f"loss={losses[index]:.4g}\nmV", fontsize=8)
        figure.suptitle(f"{title}: generation {generation}")
        figure.savefig(plot_dir / "candidates.png", dpi=dpi)
        plt.close(figure)

        # Close-up of the depolarizing steps, [-50, 150] ms around each onset.
        onset_figure, onset_axes = plt.subplots(
            len(order), len(depolarizing), figsize=(13, max(2.2 * len(order), 4.0)),
            squeeze=False, constrained_layout=True,
        )
        for row, index in enumerate(order):
            for axis, trace, simulation in zip(
                onset_axes[row], depolarizing, evaluated[int(index)][2], strict=False
            ):
                axis.plot(trace.time_ms, trace.voltage_mV, color="black", lw=0.55)
                axis.plot(simulation[0], simulation[1], color="#D55E00", lw=0.65)
                axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="#56B4E9", alpha=0.16)
                axis.set_xlim(
                    max(float(trace.time_ms[0]), trace.epoch_start_ms - 50.0),
                    min(float(trace.time_ms[-1]), trace.epoch_start_ms + 150.0),
                )
                axis.set_title(f"rank {row + 1} · {trace.name}", fontsize=8)
                axis.grid(True, color="0.9", lw=0.4)
                axis.tick_params(labelsize=7)
                if row == len(order) - 1:
                    axis.set_xlabel("time (ms)", fontsize=8)
            onset_axes[row, 0].set_ylabel(f"loss={losses[index]:.4g}\nmV", fontsize=8)
        onset_figure.suptitle(f"{title}: generation {generation} — depolarization onset")
        onset_figure.savefig(plot_dir / "candidates_onset.png", dpi=dpi)
        plt.close(onset_figure)
        records = [
            {"rank": rank, "population_index": int(index), "loss": float(losses[index]),
             "parameters": candidates[int(index)], "details": evaluated[int(index)][1]}
            for rank, index in enumerate(order, start=1)
        ]
        (plot_dir / "top_candidates.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return plot

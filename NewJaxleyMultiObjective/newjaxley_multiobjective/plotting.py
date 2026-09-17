"""Solution figures: full traces and a first-100-ms spike zoom."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def plot_solution(path: Path, records: tuple[Any, ...], predictions: Mapping[tuple[float, int], np.ndarray], buckets: tuple[Any, ...], details: Mapping[str, Any], trial_number: int, objective_values: Mapping[str, float], winners: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_trace = {}
    for bucket in buckets:
        values = np.asarray(predictions[bucket.key])
        for row, record in enumerate(bucket.records):
            by_trace[record.trace_key] = values[row, : len(record.time_ms)]
    panels = []
    for record in records:
        panels.append((record, "full"))
        if record.protocol == "depolarizing_step":
            panels.append((record, "first_100ms"))
    figure, axes = plt.subplots(len(panels), 1, figsize=(12, max(3.0, 2.5 * len(panels))), squeeze=False, constrained_layout=True)
    for axis, (record, mode) in zip(axes.flat, panels, strict=True):
        predicted = by_trace[record.trace_key]
        time = np.asarray(record.time_ms)
        mask = np.ones(len(time), dtype=bool)
        title_suffix = "full trace"
        if mode == "first_100ms":
            start = float(record.metadata["epoch_start_ms"])
            mask = (time >= start) & (time <= start + 100.0)
            title_suffix = "first 100 ms after step onset"
        axis.plot(time[mask], record.voltage_mV[mask], color="black", lw=0.8, alpha=0.65, label="experimental")
        axis.plot(time[mask], predicted[mask], color="tab:orange", lw=0.8, label="simulated")
        start = float(record.metadata["epoch_start_ms"])
        stop = float(record.metadata["epoch_stop_ms"])
        if mode == "full":
            axis.axvspan(start, stop, color="tab:blue", alpha=0.08)
        axis.set_title(f"{record.trace_id} — {record.protocol} — {title_suffix}")
        axis.set_ylabel("mV")
        axis.grid(alpha=0.2)
        if mode == "full":
            axis.axvline(start, color="tab:blue", ls="--", lw=0.6)
            axis.axvline(stop, color="tab:blue", ls="--", lw=0.6)
        if axis is axes.flat[0]:
            axis.legend(loc="best")
    axes.flat[-1].set_xlabel("time (ms)")
    summary = ", ".join(f"{key}={value:.3g}" for key, value in objective_values.items())
    winner_text = ", ".join(winners) if winners else "none"
    figure.suptitle(f"trial {trial_number} | best objectives: {winner_text}\n{summary}", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)

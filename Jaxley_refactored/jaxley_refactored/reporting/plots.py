"""Per-epoch simulated-versus-experimental trace figures."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
from typing import Iterable, Mapping

import numpy as np

from jaxley_refactored.data import TraceBucket


def plot_epoch_traces(
    directory: Path,
    buckets: Iterable[TraceBucket],
    predictions: Mapping[tuple[float, int], np.ndarray],
    *,
    epoch: int,
    loss: float,
    experimental_alpha: float = 0.4,
    experimental_linestyle: str = "-",
    filename: str | None = None,
) -> Path:
    """Save one panel per recorded trace and update ``latest.png``.

    Predictions are truncated to each record's real sample count, excluding any
    masked padding used to create a single vmap batch.
    """
    import matplotlib.pyplot as plt

    traces = []
    for bucket in buckets:
        simulated = predictions[bucket.key]
        if simulated.shape[0] != len(bucket.records):
            raise ValueError(
                f"Prediction rows {simulated.shape[0]} do not match "
                f"{len(bucket.records)} records in bucket {bucket.key}."
            )
        for row, record in enumerate(bucket.records):
            size = len(record.time_ms)
            traces.append((record, np.asarray(simulated[row, :size])))
    traces.sort(key=lambda item: (item[0].trace_id, item[0].protocol))
    if not traces:
        raise ValueError("At least one trace is required for plotting.")

    n_columns = 2 if len(traces) > 1 else 1
    n_rows = math.ceil(len(traces) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for index, (record, simulated) in enumerate(traces):
        axis = axes.flat[index]
        time_seconds = record.time_ms / 1000.0
        axis.plot(
            time_seconds,
            record.voltage_mV,
            color="black",
            linewidth=0.9,
            alpha=experimental_alpha,
            linestyle=experimental_linestyle,
            label="experimental",
        )
        axis.plot(
            time_seconds,
            simulated,
            color="tab:orange",
            linewidth=0.9,
            alpha=1.0,
            label="simulated",
        )
        axis.set_title(f"{record.trace_id} — {record.protocol}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("voltage (mV)")
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="best")
    for axis in axes.flat[len(traces) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{traces[0][0].cell_id}: epoch {epoch} — loss {loss:.6g}",
        fontsize=14,
    )

    directory.mkdir(parents=True, exist_ok=True)
    output_name = filename or f"epoch_{epoch:04d}.png"
    destination = directory / output_name
    temporary = directory / f".{output_name}.tmp.png"
    figure.savefig(temporary, dpi=140)
    plt.close(figure)
    temporary.replace(destination)
    if filename is None:
        shutil.copyfile(destination, directory / "latest.png")
    return destination

#!/usr/bin/env python3
"""Segment current-clamp traces around depolarizing and hyperpolarizing pulses."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jaxley_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Segmented_Traces"
DEFAULT_DT_MS = 0.05
DEFAULT_CURRENT_THRESHOLD_PA = 15.0
DEFAULT_EPOCH_BIN_MS = 1.0
DEFAULT_MIN_DEPOL_MS = 50.0
DEFAULT_MIN_HYPER_MS = 20.0
DEFAULT_DEPOL_PRE_MS = 200.0
DEFAULT_DEPOL_POST_MS = 500.0
DEFAULT_HYPER_PRE_MS = 500.0
DEFAULT_HYPER_POST_MS = 150.0

FLOAT_RE = re.compile(rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each paired v*ctrl.txt/i*ctrl.txt trace into a depolarizing-step "
            "window and a hyperpolarizing-pulse window."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        type=Path,
        help=(
            "Directory containing per-cell folders with paired v*ctrl.txt and "
            f"i*ctrl.txt files (default: {DEFAULT_SOURCE_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Directory for segmented traces (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dt-ms", default=DEFAULT_DT_MS, type=float)
    parser.add_argument("--current-threshold-pa", default=DEFAULT_CURRENT_THRESHOLD_PA, type=float)
    parser.add_argument("--epoch-bin-ms", default=DEFAULT_EPOCH_BIN_MS, type=float)
    parser.add_argument("--min-depol-ms", default=DEFAULT_MIN_DEPOL_MS, type=float)
    parser.add_argument("--min-hyper-ms", default=DEFAULT_MIN_HYPER_MS, type=float)
    parser.add_argument("--depol-pre-ms", default=DEFAULT_DEPOL_PRE_MS, type=float)
    parser.add_argument("--depol-post-ms", default=DEFAULT_DEPOL_POST_MS, type=float)
    parser.add_argument("--hyper-pre-ms", default=DEFAULT_HYPER_PRE_MS, type=float)
    parser.add_argument("--hyper-post-ms", default=DEFAULT_HYPER_POST_MS, type=float)
    return parser.parse_args()


def load_numeric_trace(path: Path) -> np.ndarray:
    values = [float(match) for match in FLOAT_RE.findall(path.read_bytes())]
    if not values:
        raise ValueError(f"No numeric samples found in {path}")
    return np.asarray(values, dtype=float)


def paired_current_path(voltage_path: Path) -> Path:
    return voltage_path.with_name(f"i{voltage_path.name[1:]}")


def safe_label(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "trace"


def find_cell_dirs(source_dir: Path, output_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    output_dir = output_dir.resolve()
    cell_dirs = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_dir() or path.resolve() == output_dir:
            continue
        if any(path.glob("v*ctrl.txt")):
            cell_dirs.append(path)

    if not cell_dirs:
        raise FileNotFoundError(
            f"No cell directories containing v*ctrl.txt traces found in {source_dir}"
        )
    return cell_dirs


def binned_current_epochs(
    current: np.ndarray,
    *,
    dt_ms: float,
    bin_ms: float,
    threshold_pa: float,
) -> tuple[float, list[dict[str, float | int]]]:
    baseline_samples = max(1, min(current.size, int(round(100.0 / dt_ms))))
    baseline_pa = float(np.median(current[:baseline_samples]))
    bin_samples = max(1, int(round(bin_ms / dt_ms)))

    bin_records: list[tuple[int, int, float]] = []
    for start in range(0, current.size, bin_samples):
        stop = min(current.size, start + bin_samples)
        bin_records.append((start, stop, float(np.median(current[start:stop]))))

    active = [abs(median - baseline_pa) >= threshold_pa for _, _, median in bin_records]
    epochs: list[dict[str, float | int]] = []
    index = 0
    while index < len(bin_records):
        if not active[index]:
            index += 1
            continue
        run_start = index
        while index < len(bin_records) and active[index]:
            index += 1
        run_stop = index
        sample_start = bin_records[run_start][0]
        sample_stop = bin_records[run_stop - 1][1]
        median_pa = float(np.median(current[sample_start:sample_stop]))
        epochs.append(
            {
                "start_index": sample_start,
                "stop_index": sample_stop,
                "start_ms": sample_start * dt_ms,
                "stop_ms": sample_stop * dt_ms,
                "duration_ms": (sample_stop - sample_start) * dt_ms,
                "median_pa": median_pa,
                "delta_pa": median_pa - baseline_pa,
            }
        )
    return baseline_pa, epochs


def select_depolarizing_epoch(
    epochs: list[dict[str, float | int]], *, min_depol_ms: float
) -> dict[str, float | int] | None:
    candidates = [
        epoch
        for epoch in epochs
        if float(epoch["delta_pa"]) > 0.0 and float(epoch["duration_ms"]) >= min_depol_ms
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda epoch: float(epoch["delta_pa"]) * float(epoch["duration_ms"]))


def select_hyperpolarizing_epoch(
    epochs: list[dict[str, float | int]],
    *,
    depol_epoch: dict[str, float | int] | None,
    min_hyper_ms: float,
) -> dict[str, float | int] | None:
    depol_stop = int(depol_epoch["stop_index"]) if depol_epoch is not None else -1
    candidates = [
        epoch
        for epoch in epochs
        if float(epoch["delta_pa"]) < 0.0
        and int(epoch["start_index"]) >= depol_stop
        and float(epoch["duration_ms"]) >= min_hyper_ms
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda epoch: float(epoch["delta_pa"]))


def write_vector(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%.9g")


def write_segment_plot(
    *,
    path: Path,
    time_ms: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True, constrained_layout=True)

    axes[0].plot(time_ms, voltage, color="#0072B2", lw=0.9)
    axes[0].set_ylabel("Membrane voltage (mV)")
    axes[0].set_title(title)
    axes[0].grid(True, color="0.9", lw=0.6)

    axes[1].plot(time_ms, current, color="#D55E00", lw=0.9)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Current (pA)")
    axes[1].grid(True, color="0.9", lw=0.6)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def segment_bounds(
    *,
    size: int,
    dt_ms: float,
    start_anchor_index: int,
    stop_anchor_index: int,
    pre_ms: float,
    post_ms: float,
) -> tuple[int, int, int, int]:
    pre_samples = int(round(pre_ms / dt_ms))
    post_samples = int(round(post_ms / dt_ms))
    requested_start = start_anchor_index - pre_samples
    requested_stop = stop_anchor_index + post_samples
    clipped_start = max(0, requested_start)
    clipped_stop = min(size, requested_stop)
    if clipped_stop <= clipped_start:
        raise ValueError("Requested segment does not overlap the trace")
    return requested_start, requested_stop, clipped_start, clipped_stop


def write_segment(
    *,
    output_dir: Path,
    cell_name: str,
    trace_name: str,
    segment_name: str,
    voltage: np.ndarray,
    current: np.ndarray,
    dt_ms: float,
    epoch: dict[str, float | int],
    requested_start: int,
    requested_stop: int,
    clipped_start: int,
    clipped_stop: int,
) -> dict[str, Any]:
    segment_dir = output_dir / safe_label(cell_name) / safe_label(segment_name)
    base_name = f"{safe_label(trace_name)}_{safe_label(segment_name)}"
    voltage_path = segment_dir / f"{base_name}_v.txt"
    current_path = segment_dir / f"{base_name}_i.txt"
    time_path = segment_dir / f"{base_name}_t_ms.txt"
    plot_path = segment_dir / f"{base_name}_plot.png"

    segment_voltage = voltage[clipped_start:clipped_stop]
    segment_current = current[clipped_start:clipped_stop]
    relative_time_ms = (np.arange(clipped_start, clipped_stop, dtype=float) - clipped_start) * dt_ms

    write_vector(voltage_path, segment_voltage)
    write_vector(current_path, segment_current)
    write_vector(time_path, relative_time_ms)
    write_segment_plot(
        path=plot_path,
        time_ms=relative_time_ms,
        voltage=segment_voltage,
        current=segment_current,
        title=f"{cell_name} {trace_name} {segment_name}",
    )

    return {
        "cell": cell_name,
        "trace": trace_name,
        "segment": segment_name,
        "epoch_start_ms": float(epoch["start_ms"]),
        "epoch_stop_ms": float(epoch["stop_ms"]),
        "epoch_duration_ms": float(epoch["duration_ms"]),
        "epoch_current_delta_pA": float(epoch["delta_pa"]),
        "requested_start_ms": requested_start * dt_ms,
        "requested_stop_ms": requested_stop * dt_ms,
        "segment_start_ms": clipped_start * dt_ms,
        "segment_stop_ms": clipped_stop * dt_ms,
        "segment_duration_ms": (clipped_stop - clipped_start) * dt_ms,
        "n_samples": clipped_stop - clipped_start,
        "was_clipped": requested_start != clipped_start or requested_stop != clipped_stop,
        "voltage_output": str(voltage_path),
        "current_output": str(current_path),
        "time_output": str(time_path),
        "plot_output": str(plot_path),
    }


def write_metadata(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def analyze_trace(
    *,
    voltage_path: Path,
    current_path: Path,
    output_dir: Path,
    dt_ms: float,
    epoch_bin_ms: float,
    current_threshold_pa: float,
    min_depol_ms: float,
    min_hyper_ms: float,
    depol_pre_ms: float,
    depol_post_ms: float,
    hyper_pre_ms: float,
    hyper_post_ms: float,
) -> list[dict[str, Any]]:
    voltage = load_numeric_trace(voltage_path)
    current = load_numeric_trace(current_path)
    size = min(voltage.size, current.size)
    if size < 3:
        raise ValueError(f"{voltage_path.name} has fewer than 3 aligned samples")
    voltage = voltage[:size]
    current = current[:size]

    _, epochs = binned_current_epochs(
        current,
        dt_ms=dt_ms,
        bin_ms=epoch_bin_ms,
        threshold_pa=current_threshold_pa,
    )
    depol_epoch = select_depolarizing_epoch(epochs, min_depol_ms=min_depol_ms)
    if depol_epoch is None:
        raise ValueError(f"No depolarizing current step detected in {current_path.name}")
    hyper_epoch = select_hyperpolarizing_epoch(
        epochs,
        depol_epoch=depol_epoch,
        min_hyper_ms=min_hyper_ms,
    )
    if hyper_epoch is None:
        raise ValueError(f"No hyperpolarizing pulse detected in {current_path.name}")

    records: list[dict[str, Any]] = []

    requested_start, requested_stop, clipped_start, clipped_stop = segment_bounds(
        size=size,
        dt_ms=dt_ms,
        start_anchor_index=int(depol_epoch["start_index"]),
        stop_anchor_index=int(depol_epoch["stop_index"]),
        pre_ms=depol_pre_ms,
        post_ms=depol_post_ms,
    )
    records.append(
        write_segment(
            output_dir=output_dir,
            cell_name=voltage_path.parent.name,
            trace_name=voltage_path.stem,
            segment_name="depolarizing_step",
            voltage=voltage,
            current=current,
            dt_ms=dt_ms,
            epoch=depol_epoch,
            requested_start=requested_start,
            requested_stop=requested_stop,
            clipped_start=clipped_start,
            clipped_stop=clipped_stop,
        )
    )

    requested_start, requested_stop, clipped_start, clipped_stop = segment_bounds(
        size=size,
        dt_ms=dt_ms,
        start_anchor_index=int(hyper_epoch["start_index"]),
        stop_anchor_index=int(hyper_epoch["stop_index"]),
        pre_ms=hyper_pre_ms,
        post_ms=hyper_post_ms,
    )
    records.append(
        write_segment(
            output_dir=output_dir,
            cell_name=voltage_path.parent.name,
            trace_name=voltage_path.stem,
            segment_name="hyperpolarizing_pulse",
            voltage=voltage,
            current=current,
            dt_ms=dt_ms,
            epoch=hyper_epoch,
            requested_start=requested_start,
            requested_stop=requested_stop,
            clipped_start=clipped_start,
            clipped_stop=clipped_stop,
        )
    )

    return records


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    cell_dirs = find_cell_dirs(source_dir, output_dir)

    records: list[dict[str, Any]] = []
    for cell_dir in cell_dirs:
        for voltage_path in sorted(cell_dir.glob("v*ctrl.txt")):
            current_path = paired_current_path(voltage_path)
            if not current_path.exists():
                print(f"Skipping {voltage_path}: missing {current_path.name}")
                continue
            trace_records = analyze_trace(
                voltage_path=voltage_path,
                current_path=current_path,
                output_dir=output_dir,
                dt_ms=float(args.dt_ms),
                epoch_bin_ms=float(args.epoch_bin_ms),
                current_threshold_pa=float(args.current_threshold_pa),
                min_depol_ms=float(args.min_depol_ms),
                min_hyper_ms=float(args.min_hyper_ms),
                depol_pre_ms=float(args.depol_pre_ms),
                depol_post_ms=float(args.depol_post_ms),
                hyper_pre_ms=float(args.hyper_pre_ms),
                hyper_post_ms=float(args.hyper_post_ms),
            )
            records.extend(trace_records)

    records.sort(key=lambda record: (record["cell"], record["trace"], record["segment"]))
    metadata_path = output_dir / "segment_metadata.csv"
    write_metadata(metadata_path, records)

    print(f"Wrote {len(records)} segments to {output_dir}")
    print(metadata_path)


if __name__ == "__main__":
    main()

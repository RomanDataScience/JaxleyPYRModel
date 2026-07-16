#!/usr/bin/env python3
"""Extract IPFX f-I, spike, and passive features from local current-clamp traces."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pc2b_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Figures"
DEFAULT_DT_MS = 0.05
DEFAULT_IPFX_FILTER_KHZ = 5.0
DEFAULT_CURRENT_THRESHOLD_PA = 15.0
DEFAULT_EPOCH_BIN_MS = 1.0
DEFAULT_MIN_DEPOL_MS = 50.0
DEFAULT_MIN_HYPER_MS = 20.0
DEFAULT_BASELINE_WINDOW_MS = 30.0
DEFAULT_PASSIVE_MIN_SNR = 5.0
DEFAULT_POST_STEP_ANALYSIS_MS = 700.0
DEFAULT_AHP_RECOVERY_TOLERANCE_MV = 0.5

FLOAT_RE = re.compile(rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use IPFX to extract f-I curves, spike width/interspike minima, and "
            "late hyperpolarizing-pulse passive estimates."
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
        help=f"Directory for CSV and figure outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dt-ms", default=DEFAULT_DT_MS, type=float)
    parser.add_argument("--ipfx-filter-khz", default=DEFAULT_IPFX_FILTER_KHZ, type=float)
    parser.add_argument("--current-threshold-pa", default=DEFAULT_CURRENT_THRESHOLD_PA, type=float)
    parser.add_argument("--epoch-bin-ms", default=DEFAULT_EPOCH_BIN_MS, type=float)
    parser.add_argument("--min-depol-ms", default=DEFAULT_MIN_DEPOL_MS, type=float)
    parser.add_argument("--min-hyper-ms", default=DEFAULT_MIN_HYPER_MS, type=float)
    parser.add_argument("--baseline-window-ms", default=DEFAULT_BASELINE_WINDOW_MS, type=float)
    parser.add_argument("--passive-min-snr", default=DEFAULT_PASSIVE_MIN_SNR, type=float)
    parser.add_argument("--post-step-analysis-ms", default=DEFAULT_POST_STEP_ANALYSIS_MS, type=float)
    parser.add_argument(
        "--ahp-recovery-tolerance-mv",
        default=DEFAULT_AHP_RECOVERY_TOLERANCE_MV,
        type=float,
    )
    return parser.parse_args()


def require_ipfx() -> tuple[Any, Any]:
    try:
        from ipfx.feature_extractor import SpikeFeatureExtractor
        import ipfx.subthresh_features as subthresh_features
    except ImportError as exc:
        raise RuntimeError(
            "IPFX is required for this script. Install it in the Python environment "
            "you use for electrophysiology analysis, or run with an environment "
            "where it is already available, for example:\n"
            "  python JaxleyModel/Experimental_currentClamp_Analysis/"
            "extract_ipfx_fi_passive_neurons.py"
        ) from exc
    return SpikeFeatureExtractor, subthresh_features


def safe_label(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "trace"


def load_numeric_trace(path: Path) -> np.ndarray:
    values = [float(match) for match in FLOAT_RE.findall(path.read_bytes())]
    if not values:
        raise ValueError(f"No numeric samples found in {path}")
    return np.asarray(values, dtype=float)


def paired_current_path(voltage_path: Path) -> Path:
    return voltage_path.with_name(f"i{voltage_path.name[1:]}")


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def median_interval(values: np.ndarray, start: int, stop: int) -> float:
    start = max(0, min(values.size, start))
    stop = max(start + 1, min(values.size, stop))
    return float(np.median(values[start:stop]))


def mean_interval(values: np.ndarray, start: int, stop: int) -> float:
    start = max(0, min(values.size, start))
    stop = max(start + 1, min(values.size, stop))
    return float(np.mean(values[start:stop]))


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


def dataframe_numeric_column(spikes: Any, column: str) -> np.ndarray:
    if spikes is None or len(spikes) == 0 or column not in spikes:
        return np.asarray([], dtype=float)
    return np.asarray(spikes[column], dtype=float)


def interspike_minima(voltage: np.ndarray, spikes: Any) -> list[float]:
    threshold_indexes = dataframe_numeric_column(spikes, "threshold_index")
    peak_indexes = dataframe_numeric_column(spikes, "peak_index")
    if threshold_indexes.size < 2:
        return []

    minima: list[float] = []
    for spike_index in range(threshold_indexes.size - 1):
        if math.isfinite(peak_indexes[spike_index]):
            start = int(peak_indexes[spike_index])
        else:
            start = int(threshold_indexes[spike_index])
        stop = int(threshold_indexes[spike_index + 1])
        start = max(0, min(voltage.size - 1, start))
        stop = max(start + 1, min(voltage.size, stop))
        minima.append(float(np.min(voltage[start:stop])))
    return minima


def extract_spike_features(
    *,
    SpikeFeatureExtractor: Any,
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    depol_epoch: dict[str, float | int],
    ipfx_filter_khz: float,
) -> tuple[Any, dict[str, float | int]]:
    start_s = float(depol_epoch["start_ms"]) / 1000.0
    end_s = float(depol_epoch["stop_ms"]) / 1000.0
    extractor = SpikeFeatureExtractor(start=start_s, end=end_s, filter=float(ipfx_filter_khz))
    spikes = extractor.process(time_s, voltage, current)
    spike_count = int(len(spikes))
    duration_s = max(1e-12, end_s - start_s)
    widths_s = dataframe_numeric_column(spikes, "width")
    widths_s = widths_s[np.isfinite(widths_s)]
    isi_minima = interspike_minima(voltage, spikes)
    return spikes, {
        "spike_count": spike_count,
        "firing_rate_hz": float(spike_count / duration_s),
        "mean_spike_width_ms": float(np.mean(widths_s) * 1000.0) if widths_s.size else math.nan,
        "interspike_min_v_mV": float(np.min(isi_minima)) if isi_minima else math.nan,
        "interspike_mean_min_v_mV": float(np.mean(isi_minima)) if isi_minima else math.nan,
        "mean_trough_v_mV": float(np.nanmean(dataframe_numeric_column(spikes, "trough_v")))
        if spike_count and dataframe_numeric_column(spikes, "trough_v").size
        else math.nan,
    }


def extract_passive_features(
    *,
    subthresh_features: Any,
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    hyper_epoch: dict[str, float | int] | None,
    dt_ms: float,
    baseline_window_ms: float,
    passive_min_snr: float,
) -> dict[str, float | str]:
    if hyper_epoch is None:
        return {
            "passive_status": "no_hyperpolarizing_pulse_detected",
            "hyper_start_ms": math.nan,
            "hyper_stop_ms": math.nan,
            "hyper_duration_ms": math.nan,
            "hyper_current_amp_pA": math.nan,
            "passive_baseline_current_pA": math.nan,
            "passive_baseline_voltage_mV": math.nan,
            "input_resistance_MOhm": math.nan,
            "tau_ms": math.nan,
            "membrane_capacitance_pF": math.nan,
            "reversal_potential_mV": math.nan,
            "hyperpolarizing_voltage_deflection_mV": math.nan,
        }

    start_index = int(hyper_epoch["start_index"])
    stop_index = int(hyper_epoch["stop_index"])
    start_s = float(hyper_epoch["start_ms"]) / 1000.0
    stop_s = float(hyper_epoch["stop_ms"]) / 1000.0
    baseline_samples = max(1, int(round(baseline_window_ms / dt_ms)))
    baseline_start = max(0, start_index - baseline_samples)
    baseline_current_pa = median_interval(current, baseline_start, start_index)
    centered_current = current - baseline_current_pa
    hyper_current_amp_pa = median_interval(centered_current, start_index, stop_index)
    baseline_interval_s = baseline_window_ms / 1000.0

    status = "ok"
    try:
        baseline_voltage_mv = finite_or_nan(
            subthresh_features.baseline_voltage(
                time_s,
                voltage,
                start_s,
                baseline_interval=baseline_interval_s,
            )
        )
    except Exception:
        baseline_voltage_mv = mean_interval(voltage, baseline_start, start_index)
        status = "baseline_voltage_fallback"

    try:
        input_resistance_mohm = finite_or_nan(
            subthresh_features.input_resistance(
                [time_s],
                [centered_current],
                [voltage],
                start_s,
                stop_s,
                baseline_interval=baseline_interval_s,
            )
        )
    except Exception:
        input_resistance_mohm = math.nan
        status = "input_resistance_failed"

    try:
        tau_s = finite_or_nan(
            subthresh_features.time_constant(
                time_s,
                voltage,
                centered_current,
                start_s,
                stop_s,
                baseline_interval=baseline_interval_s,
                min_snr=float(passive_min_snr),
            )
        )
    except Exception:
        tau_s = math.nan
        status = "time_constant_failed" if status == "ok" else f"{status};time_constant_failed"

    try:
        v_deflect, _ = subthresh_features.voltage_deflection(
            time_s,
            voltage,
            centered_current,
            start_s,
            stop_s,
            "min",
        )
        voltage_deflection_mv = finite_or_nan(v_deflect) - baseline_voltage_mv
    except Exception:
        voltage_deflection_mv = math.nan

    tau_ms = tau_s * 1000.0 if math.isfinite(tau_s) else math.nan
    if (
        math.isfinite(tau_s)
        and math.isfinite(input_resistance_mohm)
        and input_resistance_mohm > 0.0
    ):
        capacitance_pf = tau_s * 1e6 / input_resistance_mohm
    else:
        capacitance_pf = math.nan

    if math.isfinite(input_resistance_mohm) and math.isfinite(baseline_voltage_mv):
        reversal_potential_mv = baseline_voltage_mv - input_resistance_mohm * (
            baseline_current_pa / 1000.0
        )
    else:
        reversal_potential_mv = math.nan

    return {
        "passive_status": status,
        "hyper_start_ms": float(hyper_epoch["start_ms"]),
        "hyper_stop_ms": float(hyper_epoch["stop_ms"]),
        "hyper_duration_ms": float(hyper_epoch["duration_ms"]),
        "hyper_current_amp_pA": hyper_current_amp_pa,
        "passive_baseline_current_pA": baseline_current_pa,
        "passive_baseline_voltage_mV": baseline_voltage_mv,
        "input_resistance_MOhm": input_resistance_mohm,
        "tau_ms": tau_ms,
        "membrane_capacitance_pF": capacitance_pf,
        "reversal_potential_mV": reversal_potential_mv,
        "hyperpolarizing_voltage_deflection_mV": voltage_deflection_mv,
    }


def extract_post_step_ahp_features(
    *,
    subthresh_features: Any,
    time_s: np.ndarray,
    voltage: np.ndarray,
    depol_epoch: dict[str, float | int],
    hyper_epoch: dict[str, float | int] | None,
    dt_ms: float,
    baseline_voltage_mV: float,
    post_step_analysis_ms: float,
    ahp_recovery_tolerance_mV: float,
) -> dict[str, float | str]:
    start_index = int(depol_epoch["stop_index"])
    stop_index = min(
        voltage.size,
        start_index + max(1, int(round(post_step_analysis_ms / dt_ms))),
    )
    if hyper_epoch is not None:
        stop_index = min(stop_index, int(hyper_epoch["start_index"]))

    if stop_index <= start_index + 2:
        return {
            "post_step_ahp_status": "no_post_step_window",
            "post_step_window_start_ms": start_index * dt_ms,
            "post_step_window_stop_ms": stop_index * dt_ms,
            "post_step_ahp_abs_mV": math.nan,
            "post_step_ahp_amplitude_mV": math.nan,
            "post_step_ahp_time_ms": math.nan,
            "post_step_decay_tau_ms": math.nan,
            "post_step_decay_fit_start_ms": math.nan,
            "post_step_decay_fit_stop_ms": math.nan,
        }

    post_voltage = voltage[start_index:stop_index]
    if not np.any(np.isfinite(post_voltage)):
        return {
            "post_step_ahp_status": "nonfinite_post_step_window",
            "post_step_window_start_ms": start_index * dt_ms,
            "post_step_window_stop_ms": stop_index * dt_ms,
            "post_step_ahp_abs_mV": math.nan,
            "post_step_ahp_amplitude_mV": math.nan,
            "post_step_ahp_time_ms": math.nan,
            "post_step_decay_tau_ms": math.nan,
            "post_step_decay_fit_start_ms": math.nan,
            "post_step_decay_fit_stop_ms": math.nan,
        }

    ahp_relative_index = int(np.nanargmin(post_voltage))
    ahp_index = start_index + ahp_relative_index
    ahp_abs_mV = float(voltage[ahp_index])
    ahp_amplitude_mV = ahp_abs_mV - float(baseline_voltage_mV)
    status = "ok"

    fit_start_index = ahp_index
    fit_stop_index = stop_index
    recovery = voltage[fit_start_index:stop_index]
    close_to_baseline = np.isfinite(recovery) & (
        np.abs(recovery - float(baseline_voltage_mV)) <= float(ahp_recovery_tolerance_mV)
    )
    min_fit_samples = max(3, int(round(5.0 / dt_ms)))
    for close_index in np.flatnonzero(close_to_baseline):
        if close_index >= min_fit_samples:
            fit_stop_index = fit_start_index + int(close_index) + 1
            break

    decay_tau_ms = math.nan
    if ahp_amplitude_mV >= 0.0:
        status = "no_afterhyperpolarization_below_baseline"
    elif fit_stop_index - fit_start_index < min_fit_samples:
        status = "too_short_for_decay_fit"
    else:
        try:
            _, inv_tau, _ = subthresh_features.fit_membrane_time_constant(
                time_s,
                voltage,
                float(time_s[fit_start_index]),
                float(time_s[fit_stop_index - 1]),
                rmse_max_tol=2.0,
            )
            inv_tau = finite_or_nan(inv_tau)
            if math.isfinite(inv_tau) and inv_tau > 0.0:
                decay_tau_ms = 1000.0 / inv_tau
            else:
                status = "decay_fit_failed"
        except Exception:
            status = "decay_fit_failed"

    return {
        "post_step_ahp_status": status,
        "post_step_window_start_ms": start_index * dt_ms,
        "post_step_window_stop_ms": stop_index * dt_ms,
        "post_step_ahp_abs_mV": ahp_abs_mV,
        "post_step_ahp_amplitude_mV": ahp_amplitude_mV,
        "post_step_ahp_time_ms": ahp_relative_index * dt_ms,
        "post_step_decay_tau_ms": decay_tau_ms,
        "post_step_decay_fit_start_ms": fit_start_index * dt_ms,
        "post_step_decay_fit_stop_ms": fit_stop_index * dt_ms,
    }


def plot_metric_axis(
    ax: plt.Axes,
    currents: np.ndarray,
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    ax.plot(currents, values, "o-", color=color, lw=1.4, ms=4.5)
    finite_currents = currents[np.isfinite(currents)]
    if finite_currents.size:
        current_min = float(np.min(finite_currents))
        current_max = float(np.max(finite_currents))
        pad = max(10.0, 0.05 * max(1.0, current_max - current_min))
        ax.set_xlim(current_min - pad, current_max + pad)
    ax.set_xlabel("Depolarizing current (pA)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(True, color="0.9", lw=0.6)


def plot_cell_metrics(cell_name: str, records: list[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    currents = np.asarray([record["depol_current_amp_pA"] for record in records], dtype=float)
    metrics = [
        ("firing_rate_hz", "Firing rate (Hz)", "f-I curve", "#000000"),
        ("mean_spike_width_ms", "Width (ms)", "Average spike width", "#0072B2"),
        ("interspike_min_v_mV", "Voltage (mV)", "Minimum between-spike voltage", "#D55E00"),
        ("post_step_ahp_amplitude_mV", "mV", "Post-step AHP amplitude", "#8B4513"),
        ("post_step_decay_tau_ms", "ms", "Post-step AHP decay tau", "#6A3D9A"),
        ("input_resistance_MOhm", "MOhm", "Input resistance", "#009E73"),
        ("tau_ms", "ms", "Membrane time constant", "#CC79A7"),
        ("membrane_capacitance_pF", "pF", "Membrane capacitance", "#56B4E9"),
        ("reversal_potential_mV", "mV", "Reversal potential", "#E69F00"),
        ("hyperpolarizing_voltage_deflection_mV", "mV", "Hyperpolarizing deflection", "#7F7F7F"),
    ]

    ncols = 2
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, 2.8 * nrows),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    for ax, (field, ylabel, title, color) in zip(axes_array, metrics, strict=False):
        values = np.asarray([record[field] for record in records], dtype=float)
        plot_metric_axis(ax, currents, values, ylabel=ylabel, title=title, color=color)
    for ax in axes_array[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(f"{cell_name} IPFX f-I, spike, and passive metrics", fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_label(cell_name)}_ipfx_fi_passive_metrics.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


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


def analyze_trace(
    *,
    SpikeFeatureExtractor: Any,
    subthresh_features: Any,
    voltage_path: Path,
    current_path: Path,
    dt_ms: float,
    epoch_bin_ms: float,
    current_threshold_pa: float,
    min_depol_ms: float,
    min_hyper_ms: float,
    baseline_window_ms: float,
    ipfx_filter_khz: float,
    passive_min_snr: float,
    post_step_analysis_ms: float,
    ahp_recovery_tolerance_mV: float,
) -> dict[str, Any]:
    voltage = load_numeric_trace(voltage_path)
    current = load_numeric_trace(current_path)
    size = min(voltage.size, current.size)
    if size < 3:
        raise ValueError(f"{voltage_path.name} has fewer than 3 aligned samples")
    voltage = voltage[:size]
    current = current[:size]
    time_s = np.arange(size, dtype=float) * dt_ms / 1000.0

    baseline_current_pa, epochs = binned_current_epochs(
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

    baseline_samples = max(1, int(round(baseline_window_ms / dt_ms)))
    depol_start = int(depol_epoch["start_index"])
    depol_stop = int(depol_epoch["stop_index"])
    depol_baseline_start = max(0, depol_start - baseline_samples)
    depol_baseline_current_pa = median_interval(current, depol_baseline_start, depol_start)
    depol_current_amp_pa = median_interval(current, depol_start, depol_stop) - depol_baseline_current_pa
    depol_current_range_pa = float(
        np.max(current[depol_start:depol_stop]) - np.min(current[depol_baseline_start:depol_stop])
    )
    depol_baseline_voltage_mv = mean_interval(voltage, depol_baseline_start, depol_start)

    spikes, spike_features = extract_spike_features(
        SpikeFeatureExtractor=SpikeFeatureExtractor,
        time_s=time_s,
        voltage=voltage,
        current=current,
        depol_epoch=depol_epoch,
        ipfx_filter_khz=ipfx_filter_khz,
    )
    passive_features = extract_passive_features(
        subthresh_features=subthresh_features,
        time_s=time_s,
        voltage=voltage,
        current=current,
        hyper_epoch=hyper_epoch,
        dt_ms=dt_ms,
        baseline_window_ms=baseline_window_ms,
        passive_min_snr=passive_min_snr,
    )
    post_step_ahp_features = extract_post_step_ahp_features(
        subthresh_features=subthresh_features,
        time_s=time_s,
        voltage=voltage,
        depol_epoch=depol_epoch,
        hyper_epoch=hyper_epoch,
        dt_ms=dt_ms,
        baseline_voltage_mV=depol_baseline_voltage_mv,
        post_step_analysis_ms=post_step_analysis_ms,
        ahp_recovery_tolerance_mV=ahp_recovery_tolerance_mV,
    )

    record: dict[str, Any] = {
        "cell": voltage_path.parent.name,
        "trace": voltage_path.stem,
        "voltage_file": voltage_path.name,
        "current_file": current_path.name,
        "dt_ms": dt_ms,
        "recording_duration_ms": size * dt_ms,
        "detected_baseline_current_pA": baseline_current_pa,
        "depol_start_ms": float(depol_epoch["start_ms"]),
        "depol_stop_ms": float(depol_epoch["stop_ms"]),
        "depol_duration_ms": float(depol_epoch["duration_ms"]),
        "depol_baseline_current_pA": depol_baseline_current_pa,
        "depol_current_amp_pA": depol_current_amp_pa,
        "depol_current_range_pA": depol_current_range_pa,
        "depol_baseline_voltage_mV": depol_baseline_voltage_mv,
    }
    record.update(spike_features)
    record.update(post_step_ahp_features)
    record.update(passive_features)
    record["ipfx_spike_rows"] = int(len(spikes))
    return record


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    cell_dirs = find_cell_dirs(source_dir, output_dir)
    try:
        SpikeFeatureExtractor, subthresh_features = require_ipfx()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    written: list[Path] = []

    for cell_dir in cell_dirs:
        records: list[dict[str, Any]] = []
        for voltage_path in sorted(cell_dir.glob("v*ctrl.txt")):
            current_path = paired_current_path(voltage_path)
            if not current_path.exists():
                print(f"Skipping {voltage_path.name}: missing {current_path.name}")
                continue
            records.append(
                analyze_trace(
                    SpikeFeatureExtractor=SpikeFeatureExtractor,
                    subthresh_features=subthresh_features,
                    voltage_path=voltage_path,
                    current_path=current_path,
                    dt_ms=float(args.dt_ms),
                    epoch_bin_ms=float(args.epoch_bin_ms),
                    current_threshold_pa=float(args.current_threshold_pa),
                    min_depol_ms=float(args.min_depol_ms),
                    min_hyper_ms=float(args.min_hyper_ms),
                    baseline_window_ms=float(args.baseline_window_ms),
                    ipfx_filter_khz=float(args.ipfx_filter_khz),
                    passive_min_snr=float(args.passive_min_snr),
                    post_step_analysis_ms=float(args.post_step_analysis_ms),
                    ahp_recovery_tolerance_mV=float(args.ahp_recovery_tolerance_mv),
                )
            )

        records.sort(key=lambda record: (record["depol_current_amp_pA"], record["trace"]))
        if not records:
            continue
        cell_csv = output_dir / f"{safe_label(cell_dir.name)}_ipfx_metrics.csv"
        write_csv(cell_csv, records)
        written.append(cell_csv)
        figure_path = plot_cell_metrics(cell_dir.name, records, output_dir, dpi=200)
        written.append(figure_path)
        all_records.extend(records)

    all_records.sort(key=lambda record: (record["cell"], record["depol_current_amp_pA"]))
    combined_csv = output_dir / "all_cells_ipfx_metrics.csv"
    write_csv(combined_csv, all_records)
    written.append(combined_csv)

    print(f"Wrote {len(written)} files to {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

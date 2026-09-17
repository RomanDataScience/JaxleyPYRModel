"""Small, deterministic feature extractor for segmented voltage traces."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import numpy as np

from .config import FeatureConfig


FEATURE_LABELS = (
    "resting_membrane_potential",
    "average_fahp",
    "time_mahp",
    "amplitude_mahp",
    "ap_count",
    "last_stabilized_isi",
    "last_ap_half_width",
    "peak_input_resistance",
    "steady_state_input_resistance",
    "sag",
)


def _nan() -> float:
    return float("nan")


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else _nan()


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else _nan()


def _crossing_time(t0: float, t1: float, v0: float, v1: float, level: float) -> float:
    if v1 == v0:
        return float(t0)
    fraction = np.clip((level - v0) / (v1 - v0), 0.0, 1.0)
    return float(t0 + fraction * (t1 - t0))


def _smooth(voltage: np.ndarray, dt_ms: float, smoothing_ms: float) -> np.ndarray:
    width = max(1, int(round(smoothing_ms / dt_ms)))
    if width <= 1:
        return voltage
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(voltage, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _spikes(time: np.ndarray, voltage: np.ndarray, start: float, stop: float, config: FeatureConfig) -> list[dict[str, float | int]]:
    mask = (time >= start) & (time <= stop)
    indexes = np.flatnonzero(mask)
    if indexes.size < 2:
        return []
    smoothed = _smooth(voltage, float(np.median(np.diff(time))), config.smoothing_ms)
    candidates: list[int] = []
    for index in indexes[:-1]:
        if smoothed[index] < config.threshold_mV <= smoothed[index + 1]:
            if not candidates or time[index] - time[candidates[-1]] >= config.refractory_ms:
                candidates.append(int(index))
    result = []
    dt_ms = float(np.median(np.diff(time)))
    peak_samples = max(1, int(round(6.0 / dt_ms)))
    for crossing in candidates:
        peak_stop = min(len(voltage), crossing + peak_samples + 1)
        peak_index = crossing + int(np.argmax(voltage[crossing:peak_stop]))
        threshold_time = _crossing_time(
            time[crossing], time[crossing + 1], voltage[crossing], voltage[crossing + 1], config.threshold_mV
        )
        result.append({"crossing_index": crossing, "threshold_time_ms": threshold_time, "peak_index": peak_index, "peak_time_ms": float(time[peak_index]), "peak_voltage_mV": float(voltage[peak_index])})
    return result


def _half_width(time: np.ndarray, voltage: np.ndarray, spike: dict[str, float | int], threshold: float) -> float:
    peak = int(spike["peak_index"])
    peak_voltage = float(spike["peak_voltage_mV"])
    level = threshold + 0.5 * (peak_voltage - threshold)
    left = peak
    while left > 0 and voltage[left] >= level:
        left -= 1
    right = peak
    while right + 1 < len(voltage) and voltage[right] >= level:
        right += 1
    if left == peak or right == peak or right >= len(voltage):
        return _nan()
    rising = _crossing_time(time[left], time[left + 1], voltage[left], voltage[left + 1], level)
    falling = _crossing_time(time[right - 1], time[right], voltage[right - 1], voltage[right], level)
    return float(falling - rising)


def _epoch(record: Any) -> tuple[float, float]:
    metadata = record.metadata
    return float(metadata["epoch_start_ms"]), float(metadata["epoch_stop_ms"])


def extract_features(record: Any, voltage: np.ndarray, config: FeatureConfig) -> dict[str, Any]:
    """Extract requested features and diagnostic event metadata from one trace."""
    time = np.asarray(record.time_ms, dtype=float)
    voltage = np.asarray(voltage, dtype=float)[: len(time)]
    dt_ms = float(record.dt_ms)
    start, stop = _epoch(record)
    baseline_mask = (time >= start - config.baseline_window_ms) & (time < start)
    if not np.any(baseline_mask):
        baseline_mask = time < start
    baseline_voltage = _median(voltage[baseline_mask])
    features = {label: _nan() for label in FEATURE_LABELS}
    features["resting_membrane_potential"] = baseline_voltage
    diagnostics: dict[str, Any] = {"validity": {}, "windows": {}, "spikes": []}
    diagnostics["windows"]["baseline"] = [max(float(time[0]), start - config.baseline_window_ms), start]

    if record.protocol == "depolarizing_step":
        spikes = _spikes(time, voltage, start, stop, config)
        diagnostics["spikes"] = spikes
        features["ap_count"] = float(len(spikes))
        fahp_values = []
        for index, spike in enumerate(spikes):
            begin = int(spike["peak_index"])
            if index + 1 < len(spikes):
                end = int(spikes[index + 1]["crossing_index"])
            else:
                end = min(len(voltage), begin + max(1, int(round(config.fahp_search_ms / dt_ms))))
            if end > begin:
                fahp_values.append(float(np.min(voltage[begin:end])))
        if fahp_values:
            features["average_fahp"] = _mean(np.asarray(fahp_values))
        recovery_mask = (time > stop) & (time <= stop + config.post_step_analysis_ms)
        recovery_indexes = np.flatnonzero(recovery_mask)
        if recovery_indexes.size:
            minimum_index = int(recovery_indexes[np.argmin(voltage[recovery_indexes])])
            minimum = float(voltage[minimum_index])
            features["time_mahp"] = float(time[minimum_index] - stop)
            features["amplitude_mahp"] = float(baseline_voltage - minimum)
            diagnostics["windows"]["mahp"] = [stop, float(time[recovery_indexes[-1]])]
            diagnostics["mahp_minimum"] = {"time_ms": float(time[minimum_index]), "voltage_mV": minimum}
        if len(spikes) >= 2:
            features["last_stabilized_isi"] = float(spikes[-1]["peak_time_ms"] - spikes[-2]["peak_time_ms"])
        if spikes:
            features["last_ap_half_width"] = _half_width(time, voltage, spikes[-1], config.threshold_mV)
        diagnostics["fahp_troughs_mV"] = fahp_values

    if record.protocol in {"hyperpolarizing_pulse", "hyperpolarizing_step"}:
        epoch_mask = (time >= start) & (time <= stop)
        epoch_indexes = np.flatnonzero(epoch_mask)
        baseline_current_mask = (time >= start - config.baseline_window_ms) & (time < start)
        if not np.any(baseline_current_mask):
            baseline_current_mask = time < start
        delta_current = _median(np.asarray(record.current_nA)[epoch_mask]) - _median(np.asarray(record.current_nA)[baseline_current_mask])
        if epoch_indexes.size and math.isfinite(delta_current) and abs(delta_current) > 1e-12:
            peak_end = min(stop, start + config.hyper_peak_window_ms)
            peak_mask = (time >= start) & (time <= peak_end)
            peak_indexes = np.flatnonzero(peak_mask)
            if peak_indexes.size:
                peak_index = int(peak_indexes[np.argmin(voltage[peak_indexes])])
                steady_start = stop - (stop - start) * config.hyper_steady_state_fraction
                steady_mask = (time >= steady_start) & (time <= stop)
                steady_indexes = np.flatnonzero(steady_mask)
                if steady_indexes.size:
                    peak_voltage = float(voltage[peak_index])
                    steady_voltage = _mean(voltage[steady_indexes])
                    peak_delta = peak_voltage - baseline_voltage
                    steady_delta = steady_voltage - baseline_voltage
                    features["peak_input_resistance"] = float(peak_delta / delta_current)
                    features["steady_state_input_resistance"] = float(steady_delta / delta_current)
                    denominator = baseline_voltage - peak_voltage
                    if abs(denominator) > 1e-12:
                        features["sag"] = float((steady_voltage - peak_voltage) / denominator)
                    diagnostics["hyper"] = {"peak_voltage_mV": peak_voltage, "steady_state_voltage_mV": steady_voltage, "delta_current_nA": float(delta_current), "raw_sag_mV": float(steady_voltage - peak_voltage), "peak_window": [start, peak_end], "steady_window": [steady_start, stop]}

    for label, value in features.items():
        diagnostics["validity"][label] = bool(math.isfinite(value))
    return {"features": features, "diagnostics": diagnostics}


def feature_config_dict(config: FeatureConfig) -> dict[str, Any]:
    return asdict(config)

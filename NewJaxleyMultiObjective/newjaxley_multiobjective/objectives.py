"""Feature-error and time-course objectives."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .config import FeatureConfig, OBJECTIVE_LABELS, ObjectiveConfig
from .features import detect_spikes, extract_features


def _protocol_for(label: str) -> set[str]:
    if label in {"peak_input_resistance", "steady_state_input_resistance", "sag"}:
        return {"hyperpolarizing_pulse", "hyperpolarizing_step"}
    if label in {"average_fahp", "time_mahp", "amplitude_mahp", "ap_count", "last_stabilized_isi", "last_ap_half_width"}:
        return {"depolarizing_step"}
    return {"depolarizing_step", "hyperpolarizing_pulse", "hyperpolarizing_step"}


def _window_masks(record: Any) -> dict[str, np.ndarray]:
    time = np.asarray(record.time_ms, dtype=float)
    start = float(record.metadata["epoch_start_ms"])
    stop = float(record.metadata["epoch_stop_ms"])
    masks = {
        "baseline": time < start,
        "stimulus": (time >= start) & (time <= stop),
        "recovery": time > stop,
        "first_100ms_spiking": (time >= start) & (time < min(stop, start + 100.0)),
    }
    score = np.asarray(record.score_mask, dtype=bool)
    return {key: value & score for key, value in masks.items()}


def trajectory_overlap(
    record: Any,
    predicted: np.ndarray,
    config: ObjectiveConfig,
    feature_config: FeatureConfig | None = None,
) -> dict[str, Any]:
    """Return a local, absolute-voltage overlap loss for one trace.

    Samples near experimentally observed depolarizing spikes receive extra
    weight. This keeps the trajectory objective sensitive to spike timing and
    shape without turning it into a second AP-count objective.
    """
    observed = np.asarray(record.voltage_mV, dtype=float)
    predicted = np.asarray(predicted, dtype=float)[: len(observed)]
    error = predicted - observed
    overlap = np.exp(-0.5 * (error / config.overlap_sigma_mV) ** 2)
    masks = _window_masks(record)
    values = {}
    event_mask = np.zeros(len(observed), dtype=bool)
    experimental_spikes = []
    if getattr(record, "protocol", None) == "depolarizing_step":
        feature_config = feature_config or FeatureConfig()
        start = float(record.metadata["epoch_start_ms"])
        stop = float(record.metadata["epoch_stop_ms"])
        experimental_spikes = detect_spikes(
            np.asarray(record.time_ms, dtype=float),
            observed,
            start,
            stop,
            feature_config,
        )
        time = np.asarray(record.time_ms, dtype=float)
        for spike in experimental_spikes:
            event_mask |= np.abs(time - float(spike["peak_time_ms"])) <= config.spike_event_window_ms
    diagnostics = {
        "mae_mV": float(np.mean(np.abs(error))),
        "rmse_mV": float(np.sqrt(np.mean(error**2))),
        "within_1mV_fraction": float(np.mean(np.abs(error) <= 1.0)),
        "within_2mV_fraction": float(np.mean(np.abs(error) <= 2.0)),
        "within_5mV_fraction": float(np.mean(np.abs(error) <= 5.0)),
        "experimental_spike_count": len(experimental_spikes),
        "spike_event_samples": int(np.count_nonzero(event_mask)),
    }
    weighted_scores = []
    for name, mask in masks.items():
        if np.any(mask):
            sample_weights = np.ones(int(np.count_nonzero(mask)), dtype=float)
            if np.any(event_mask[mask]):
                sample_weights[event_mask[mask]] = config.spike_event_weight
            score = float(np.average(overlap[mask], weights=sample_weights))
            values[name] = score
            weight = float(config.overlap_window_weights.get(name, 1.0))
            weighted_scores.append((score, weight))
    if not weighted_scores:
        return {"loss": float(config.invalid_feature_penalty), "windows": values, "diagnostics": diagnostics, "valid": False, "reason": "empty_score_mask"}
    total_weight = sum(weight for _, weight in weighted_scores)
    loss = 1.0 - sum(score * weight for score, weight in weighted_scores) / total_weight
    return {"loss": float(max(0.0, loss)), "windows": values, "diagnostics": diagnostics, "valid": True, "reason": None}


def evaluate_objectives(records: tuple[Any, ...], predictions: Mapping[tuple[float, int], np.ndarray], buckets: tuple[Any, ...], experimental: Mapping[str, Mapping[str, Any]], feature_config: Any, objective_config: ObjectiveConfig) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Extract candidate features and return the eleven objective values."""
    predicted_by_trace: dict[str, np.ndarray] = {}
    for bucket in buckets:
        values = np.asarray(predictions[bucket.key])
        for row, record in enumerate(bucket.records):
            predicted_by_trace[record.trace_key] = values[row, : len(record.time_ms)]

    simulated: dict[str, Any] = {}
    for record in records:
        simulated[record.trace_key] = extract_features(record, predicted_by_trace[record.trace_key], feature_config)

    objective_values: list[float] = []
    details: dict[str, Any] = {"features_by_trace": {}, "trajectory_by_trace": {}, "objectives": {}}
    for record in records:
        key = record.trace_key
        details["features_by_trace"][key] = {"experimental": experimental[key], "simulated": simulated[key]}
        details["trajectory_by_trace"][key] = trajectory_overlap(
            record,
            predicted_by_trace[key],
            objective_config,
            feature_config,
        )

    spike_violations = []
    for record in records:
        time = np.asarray(record.time_ms, dtype=float)
        predicted_spikes = detect_spikes(
            time,
            predicted_by_trace[record.trace_key],
            float(time[0]),
            float(time[-1]),
            feature_config,
        )
        start = float(record.metadata["epoch_start_ms"])
        stop = float(record.metadata["epoch_stop_ms"])
        if record.protocol == "depolarizing_step":
            invalid_spikes = [
                spike for spike in predicted_spikes
                if not start <= float(spike["threshold_time_ms"]) <= stop
            ]
            reason = "depolarizing_spike_outside_stimulus"
        elif record.protocol in {"hyperpolarizing_pulse", "hyperpolarizing_step"}:
            invalid_spikes = predicted_spikes
            reason = "hyperpolarizing_spike_detected"
        else:
            invalid_spikes = []
            reason = ""
        if invalid_spikes:
            spike_violations.append({
                "trace_key": record.trace_key,
                "reason": reason,
                "spikes": invalid_spikes,
            })
    details["spike_violations"] = spike_violations
    ap_count_violations = []
    for record in records:
        if record.protocol != "depolarizing_step":
            continue
        key = record.trace_key
        target = float(experimental[key]["features"].get("ap_count", float("nan")))
        candidate = float(simulated[key]["features"].get("ap_count", float("nan")))
        if math.isfinite(target) and math.isfinite(candidate):
            error = abs(candidate - target)
            if error > objective_config.ap_count_tolerance:
                ap_count_violations.append({
                    "trace_key": key,
                    "target": target,
                    "candidate": candidate,
                    "absolute_error": error,
                    "tolerance": objective_config.ap_count_tolerance,
                })
    details["ap_count_violations"] = ap_count_violations
    if spike_violations or ap_count_violations:
        if spike_violations and ap_count_violations:
            loss = max(objective_config.spike_violation_loss, objective_config.ap_count_violation_loss)
            constraint_type = "hard_spike_and_ap_count_constraint"
        elif spike_violations:
            loss = objective_config.spike_violation_loss
            constraint_type = "hard_spike_constraint"
        else:
            loss = objective_config.ap_count_violation_loss
            constraint_type = "hard_ap_count_constraint"
        loss = float(loss)
        details["objectives"] = {
            label: {
                "value": loss,
                "type": constraint_type,
                "violations": spike_violations,
                "ap_count_violations": ap_count_violations,
            }
            for label in OBJECTIVE_LABELS
        }
        return tuple([loss] * len(OBJECTIVE_LABELS)), details

    for label in OBJECTIVE_LABELS:
        if label == "trajectory_overlap":
            applicable = [details["trajectory_by_trace"][record.trace_key]["loss"] for record in records]
            value = float(np.mean(applicable)) if applicable else objective_config.invalid_feature_penalty
            details["objectives"][label] = {"value": value, "type": "trajectory_overlap"}
            objective_values.append(value)
            continue
        errors = []
        invalid = []
        protocols = _protocol_for(label)
        for record in records:
            if record.protocol not in protocols:
                continue
            key = record.trace_key
            target = float(experimental[key]["features"].get(label, float("nan")))
            candidate = float(simulated[key]["features"].get(label, float("nan")))
            if math.isfinite(target) and math.isfinite(candidate):
                scale = objective_config.scales[label]
                errors.append(((candidate - target) / scale) ** 2)
            else:
                invalid.append({"trace_key": key, "target": target, "candidate": candidate, "reason": simulated[key]["diagnostics"]["validity"].get(label, False)})
        value = float(np.mean(errors)) if errors else 0.0
        if invalid:
            value += objective_config.invalid_feature_penalty
        details["objectives"][label] = {"value": value, "errors": errors, "invalid": invalid}
        objective_values.append(value)
    if len(objective_values) != len(OBJECTIVE_LABELS):
        raise AssertionError("Objective vector must contain eleven values")
    return tuple(objective_values), details

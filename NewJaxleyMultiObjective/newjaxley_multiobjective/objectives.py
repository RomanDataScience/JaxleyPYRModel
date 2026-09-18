"""Feature-error and time-course objectives."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .config import (
    DEPOLARIZING_OBJECTIVE_LABELS,
    FeatureConfig,
    MOCMA_FOUR_OBJECTIVE_LABELS,
    OBJECTIVE_LABELS,
    ObjectiveConfig,
)
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


def objective_labels_for_trace_ids(
    config: ObjectiveConfig,
    trace_ids: tuple[int, ...],
) -> tuple[str, ...]:
    """Return stable labels for a run, including one voltage objective per trace."""
    if config.mode == "mocma_four_objectives":
        return MOCMA_FOUR_OBJECTIVE_LABELS
    if config.mode != "depolarizing_fidelity":
        return config.labels
    return tuple(
        f"depolarizing_trace_{trace_id}_voltage" for trace_id in trace_ids
    ) + DEPOLARIZING_OBJECTIVE_LABELS[1:]


def _depolarizing_trace_labels(records: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(
        f"depolarizing_trace_{record.trace_id}_voltage" for record in records
    ) + DEPOLARIZING_OBJECTIVE_LABELS[1:]


def _four_objective_scale(config: ObjectiveConfig, label: str) -> float:
    return float(config.scales[label])


def _aligned_spike_waveform(
    time: np.ndarray,
    voltage: np.ndarray,
    spike: Mapping[str, Any],
    window_ms: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a baseline-subtracted AP waveform on its native time grid."""
    peak_time = float(spike["peak_time_ms"])
    dt_ms = float(np.median(np.diff(time)))
    if peak_time - window_ms < time[0] or peak_time + window_ms > time[-1]:
        return None
    relative_time = np.arange(-window_ms, window_ms + 0.5 * dt_ms, dt_ms)
    samples = np.interp(peak_time + relative_time, time, voltage)
    baseline_mask = relative_time <= -0.5 * window_ms
    baseline = float(np.median(samples[baseline_mask])) if np.any(baseline_mask) else float(samples[0])
    return relative_time, samples - baseline


def _masked_stimulus(
    record: Any,
    spikes: list[Mapping[str, Any]],
    exclusion_ms: float,
) -> np.ndarray:
    time = np.asarray(record.time_ms, dtype=float)
    start = float(record.metadata["epoch_start_ms"])
    stop = float(record.metadata["epoch_stop_ms"])
    mask = (
        (time >= start)
        & (time <= stop)
        & np.asarray(record.score_mask, dtype=bool)
    )
    for spike in spikes:
        mask &= np.abs(time - float(spike["peak_time_ms"])) > exclusion_ms
    return mask


def _mocma_four_objective_values(
    records: tuple[Any, ...],
    predicted_by_trace: Mapping[str, np.ndarray],
    experimental: Mapping[str, Mapping[str, Any]],
    simulated: Mapping[str, Mapping[str, Any]],
    details: dict[str, Any],
    config: ObjectiveConfig,
    feature_config: FeatureConfig,
) -> tuple[float, ...]:
    """Evaluate the four depolarizing objectives used by the requested MOCMA run.

    The objectives are intentionally orthogonal:
    firing rate uses event count, spike shape uses peak-aligned AP waveforms,
    recovery uses the post-step trajectory and terminal return to baseline,
    and plateau uses fixed-time inter-spike voltage including post-spike AHPs.
    """
    if not records or any(record.protocol != "depolarizing_step" for record in records):
        penalty = float(config.invalid_feature_penalty)
        return (penalty,) * len(MOCMA_FOUR_OBJECTIVE_LABELS)

    firing_rate_losses = []
    shape_losses = []
    recovery_losses = []
    plateau_losses = []
    objective_details: dict[str, Any] = {}

    for record in records:
        key = record.trace_key
        time = np.asarray(record.time_ms, dtype=float)
        observed = np.asarray(record.voltage_mV, dtype=float)
        predicted = np.asarray(predicted_by_trace[key], dtype=float)[: len(time)]
        start = float(record.metadata["epoch_start_ms"])
        stop = float(record.metadata["epoch_stop_ms"])
        duration_ms = stop - start
        if duration_ms <= 0:
            return (float(config.invalid_feature_penalty),) * len(MOCMA_FOUR_OBJECTIVE_LABELS)

        experimental_spikes = detect_spikes(time, observed, start, stop, feature_config)
        simulated_spikes = detect_spikes(time, predicted, start, stop, feature_config)

        # 1. Firing rate: AP count converted to Hz over the actual stimulus.
        expected_rate_hz = 1000.0 * len(experimental_spikes) / duration_ms
        simulated_rate_hz = 1000.0 * len(simulated_spikes) / duration_ms
        firing_rate_losses.append(
            ((simulated_rate_hz - expected_rate_hz) / _four_objective_scale(config, "firing_rate")) ** 2
        )

        # 2. Spike shape: align each corresponding AP at its own peak, then
        # compare the voltage waveform including amplitude and half-width.
        paired = min(len(experimental_spikes), len(simulated_spikes))
        event_shape_losses = []
        shape_events = []
        for expected, candidate in zip(experimental_spikes[:paired], simulated_spikes[:paired], strict=True):
            expected_waveform = _aligned_spike_waveform(
                time, observed, expected, config.spike_shape_window_ms
            )
            candidate_waveform = _aligned_spike_waveform(
                time, predicted, candidate, config.spike_shape_window_ms
            )
            if expected_waveform is None or candidate_waveform is None:
                continue
            relative_time, expected_values = expected_waveform
            _, candidate_values = candidate_waveform
            error = (candidate_values - expected_values) / _four_objective_scale(config, "spike_shape")
            event_shape_losses.append(float(np.mean(error**2)))
            shape_events.append({
                "experimental_peak_time_ms": float(expected["peak_time_ms"]),
                "simulated_peak_time_ms": float(candidate["peak_time_ms"]),
                "window_ms": [-config.spike_shape_window_ms, config.spike_shape_window_ms],
                "waveform_time_ms": relative_time.tolist(),
                "experimental_voltage_relative_mV": expected_values.tolist(),
                "simulated_voltage_relative_mV": candidate_values.tolist(),
            })
        if experimental_spikes and not event_shape_losses:
            shape_value = float(config.invalid_feature_penalty)
        else:
            shape_value = float(np.mean(event_shape_losses)) if event_shape_losses else 0.0
        shape_losses.append(shape_value)

        # 3. Recovery: compare the post-step voltage course and explicitly
        # score the final value relative to both the measured trace and its
        # pre-step baseline.
        recovery_mask = (
            (time > stop)
            & (time <= stop + feature_config.post_step_analysis_ms)
            & np.asarray(record.score_mask, dtype=bool)
        )
        recovery_indices = np.flatnonzero(recovery_mask)
        recovery_detail: dict[str, Any] = {"valid": bool(recovery_indices.size)}
        if recovery_indices.size:
            error = (predicted[recovery_indices] - observed[recovery_indices]) / _four_objective_scale(config, "post_stimulus_recovery")
            trajectory_loss = float(np.mean(error**2))
            terminal_count = max(1, int(round(0.05 * recovery_indices.size)))
            terminal = recovery_indices[-terminal_count:]
            experimental_terminal = float(np.mean(observed[terminal]))
            simulated_terminal = float(np.mean(predicted[terminal]))
            baseline = float(experimental[key]["features"].get("resting_membrane_potential", np.nan))
            terminal_trace_loss = ((simulated_terminal - experimental_terminal) / _four_objective_scale(config, "post_stimulus_recovery")) ** 2
            terminal_baseline_loss = (
                ((simulated_terminal - baseline) / _four_objective_scale(config, "post_stimulus_recovery")) ** 2
                if math.isfinite(baseline)
                else 0.0
            )
            recovery_value = 0.5 * trajectory_loss + 0.25 * terminal_trace_loss + 0.25 * terminal_baseline_loss
            recovery_detail.update({
                "window_ms": [stop, float(time[recovery_indices[-1]])],
                "trajectory_loss": trajectory_loss,
                "experimental_terminal_mV": experimental_terminal,
                "simulated_terminal_mV": simulated_terminal,
                "experimental_baseline_mV": baseline,
                "terminal_trace_loss": float(terminal_trace_loss),
                "terminal_baseline_loss": float(terminal_baseline_loss),
            })
        else:
            recovery_value = float(config.invalid_feature_penalty)
        recovery_losses.append(float(recovery_value))

        # 4. Plateau: compare fixed-time voltages after removing only the
        # narrow AP cores. The post-spike AHP remains in this mask, so a model
        # that drops below the depolarized plateau is penalized.
        plateau_mask = _masked_stimulus(
            record,
            [*experimental_spikes, *simulated_spikes],
            config.plateau_spike_exclusion_ms,
        )
        plateau_detail: dict[str, Any] = {"valid": bool(np.any(plateau_mask))}
        if np.any(plateau_mask):
            plateau_error = (predicted[plateau_mask] - observed[plateau_mask]) / _four_objective_scale(config, "depolarized_plateau")
            plateau_value = float(np.mean(plateau_error**2))
            plateau_detail.update({
                "samples": int(np.count_nonzero(plateau_mask)),
                "experimental_median_mV": float(np.median(observed[plateau_mask])),
                "simulated_median_mV": float(np.median(predicted[plateau_mask])),
                "window_ms": [start, stop],
            })
        else:
            plateau_value = float(config.invalid_feature_penalty)
        plateau_losses.append(plateau_value)

        objective_details[key] = {
            "experimental_spike_count": len(experimental_spikes),
            "simulated_spike_count": len(simulated_spikes),
            "experimental_firing_rate_Hz": expected_rate_hz,
            "simulated_firing_rate_Hz": simulated_rate_hz,
            "spike_shape": {"valid": bool(event_shape_losses), "events": shape_events},
            "recovery": recovery_detail,
            "plateau": plateau_detail,
        }

    details["mocma_four_objectives_by_trace"] = objective_details
    values = (
        float(np.mean(firing_rate_losses)),
        float(np.mean(shape_losses)),
        float(np.mean(recovery_losses)),
        float(np.mean(plateau_losses)),
    )
    for label, value in zip(MOCMA_FOUR_OBJECTIVE_LABELS, values, strict=True):
        details["objectives"][label] = {"value": value, "type": "mocma_four_objectives"}
    return values


def robust_trajectory_loss(
    record: Any,
    predicted: np.ndarray,
    config: ObjectiveConfig,
    feature_config: FeatureConfig,
) -> dict[str, Any]:
    """Compare voltage at fixed time points with a normalized Huber loss."""
    observed = np.asarray(record.voltage_mV, dtype=float)
    predicted = np.asarray(predicted, dtype=float)[: len(observed)]
    error = predicted - observed
    delta = config.trajectory_huber_delta_mV
    scaled = np.abs(error) / delta
    point_loss = np.where(scaled <= 1.0, 0.5 * scaled**2, scaled - 0.5)
    time = np.asarray(record.time_ms, dtype=float)
    start = float(record.metadata["epoch_start_ms"])
    stop = float(record.metadata["epoch_stop_ms"])
    experimental_spikes = detect_spikes(time, observed, start, stop, feature_config)
    event_mask = np.zeros(len(observed), dtype=bool)
    for spike in experimental_spikes:
        event_mask |= np.abs(time - float(spike["peak_time_ms"])) <= config.spike_event_window_ms

    values = {}
    weighted_scores = []
    for name, mask in _window_masks(record).items():
        if np.any(mask):
            sample_weights = np.ones(int(np.count_nonzero(mask)), dtype=float)
            sample_weights[event_mask[mask]] = config.spike_event_weight
            score = float(np.average(point_loss[mask], weights=sample_weights))
            values[name] = score
            weighted_scores.append((score, float(config.overlap_window_weights.get(name, 1.0))))
    diagnostics = {
        "mae_mV": float(np.mean(np.abs(error))),
        "rmse_mV": float(np.sqrt(np.mean(error**2))),
        "huber_delta_mV": delta,
        "experimental_spike_count": len(experimental_spikes),
        "spike_event_samples": int(np.count_nonzero(event_mask)),
    }
    if not weighted_scores:
        return {
            "loss": float(config.invalid_feature_penalty),
            "windows": values,
            "diagnostics": diagnostics,
            "valid": False,
            "reason": "empty_score_mask",
        }
    total_weight = sum(weight for _, weight in weighted_scores)
    loss = sum(score * weight for score, weight in weighted_scores) / total_weight
    return {
        "loss": float(loss),
        "windows": values,
        "diagnostics": diagnostics,
        "valid": True,
        "reason": None,
    }


def _feature_group_loss(
    labels: tuple[str, ...],
    records: tuple[Any, ...],
    experimental: Mapping[str, Mapping[str, Any]],
    simulated: Mapping[str, Mapping[str, Any]],
    config: ObjectiveConfig,
) -> tuple[float, list[float], list[dict[str, Any]]]:
    errors = []
    invalid = []
    for record in records:
        key = record.trace_key
        for label in labels:
            target = float(experimental[key]["features"].get(label, float("nan")))
            candidate = float(simulated[key]["features"].get(label, float("nan")))
            if math.isfinite(target) and math.isfinite(candidate):
                errors.append(((candidate - target) / config.scales[label]) ** 2)
            else:
                invalid.append({
                    "trace_key": key,
                    "feature": label,
                    "target": target,
                    "candidate": candidate,
                })
    value = float(np.mean(errors)) if errors else 0.0
    if invalid:
        value += config.invalid_feature_penalty
    return value, errors, invalid


def _depolarizing_fidelity_objectives(
    records: tuple[Any, ...],
    predicted_by_trace: Mapping[str, np.ndarray],
    experimental: Mapping[str, Mapping[str, Any]],
    simulated: Mapping[str, Mapping[str, Any]],
    details: dict[str, Any],
    config: ObjectiveConfig,
    feature_config: FeatureConfig,
    labels: tuple[str, ...],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    trajectory_labels = labels[: len(records)]
    objective_values = []
    for label, record in zip(trajectory_labels, records, strict=True):
        value = float(details["trajectory_by_trace"][record.trace_key]["loss"])
        details["objectives"][label] = {"value": value, "type": "robust_fixed_time_voltage"}
        objective_values.append(value)

    spike_trace_terms = []
    spike_metrics = {}
    for record in records:
        key = record.trace_key
        time = np.asarray(record.time_ms, dtype=float)
        start = float(record.metadata["epoch_start_ms"])
        stop = float(record.metadata["epoch_stop_ms"])
        experimental_spikes = detect_spikes(time, record.voltage_mV, start, stop, feature_config)
        # The caller has already used the configured detector for simulated
        # features; this local calculation is only for event timing details.
        simulated_spikes = detect_spikes(
            time,
            predicted_by_trace[key],
            start,
            stop,
            feature_config,
        )
        count_error = (len(simulated_spikes) - len(experimental_spikes)) / config.spike_count_scale
        trace_terms = [count_error**2]
        matched = min(len(experimental_spikes), len(simulated_spikes))
        timing_errors = []
        for expected, candidate in zip(experimental_spikes[:matched], simulated_spikes[:matched], strict=True):
            threshold_error = (
                float(candidate["threshold_time_ms"]) - float(expected["threshold_time_ms"])
            ) / config.spike_timing_scale_ms
            peak_error = (
                float(candidate["peak_time_ms"]) - float(expected["peak_time_ms"])
            ) / config.spike_timing_scale_ms
            timing_errors.append({
                "threshold_error_ms": threshold_error * config.spike_timing_scale_ms,
                "peak_error_ms": peak_error * config.spike_timing_scale_ms,
            })
            trace_terms.append(0.5 * threshold_error**2 + 0.5 * peak_error**2)
        spike_trace_terms.append(float(np.mean(trace_terms)))
        spike_metrics[key] = {
            "experimental_count": len(experimental_spikes),
            "simulated_count": len(simulated_spikes),
            "timing_errors": timing_errors,
            "loss": spike_trace_terms[-1],
        }
    spike_value = float(np.mean(spike_trace_terms)) if spike_trace_terms else config.invalid_feature_penalty
    details["spike_metrics_by_trace"] = spike_metrics
    details["objectives"]["spike_timing_count"] = {
        "value": spike_value,
        "type": "ap_count_and_event_timing",
    }
    objective_values.append(spike_value)

    waveform_value, waveform_errors, waveform_invalid = _feature_group_loss(
        ("average_fahp", "last_stabilized_isi", "last_ap_half_width"),
        records,
        experimental,
        simulated,
        config,
    )
    details["objectives"]["ap_waveform"] = {
        "value": waveform_value,
        "type": "ap_waveform_features",
        "errors": waveform_errors,
        "invalid": waveform_invalid,
    }
    objective_values.append(waveform_value)

    recovery_value, recovery_errors, recovery_invalid = _feature_group_loss(
        ("resting_membrane_potential", "time_mahp", "amplitude_mahp"),
        records,
        experimental,
        simulated,
        config,
    )
    details["objectives"]["recovery"] = {
        "value": recovery_value,
        "type": "baseline_and_recovery_features",
        "errors": recovery_errors,
        "invalid": recovery_invalid,
    }
    objective_values.append(recovery_value)
    return tuple(objective_values), details


def evaluate_objectives(
    records: tuple[Any, ...],
    predictions: Mapping[tuple[float, int], np.ndarray],
    buckets: tuple[Any, ...],
    experimental: Mapping[str, Mapping[str, Any]],
    feature_config: Any,
    objective_config: ObjectiveConfig,
    labels: tuple[str, ...] | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Extract candidate features and return the configured objective values."""
    labels = tuple(labels or (
        _depolarizing_trace_labels(records)
        if objective_config.mode == "depolarizing_fidelity"
        else MOCMA_FOUR_OBJECTIVE_LABELS
        if objective_config.mode == "mocma_four_objectives"
        else objective_config.labels
    ))
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
        if objective_config.mode == "depolarizing_fidelity":
            details["trajectory_by_trace"][key] = robust_trajectory_loss(
                record,
                predicted_by_trace[key],
                objective_config,
                feature_config,
            )
        else:
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
    # In the four-objective mode, firing rate is deliberately a smooth
    # objective. Do not turn count differences into a duplicate hard
    # constraint; the outside-stimulus spike guard remains active above.
    if objective_config.mode != "mocma_four_objectives":
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
            for label in labels
        }
        return tuple([loss] * len(labels)), details

    if objective_config.mode == "depolarizing_fidelity":
        if any(record.protocol != "depolarizing_step" for record in records):
            raise ValueError("depolarizing_fidelity mode requires depolarizing_step records only")
        return _depolarizing_fidelity_objectives(
            records,
            predicted_by_trace,
            experimental,
            simulated,
            details,
            objective_config,
            feature_config,
            labels,
        )

    if objective_config.mode == "mocma_four_objectives":
        if any(record.protocol != "depolarizing_step" for record in records):
            raise ValueError("mocma_four_objectives mode requires depolarizing_step records only")
        return _mocma_four_objective_values(
            records,
            predicted_by_trace,
            experimental,
            simulated,
            details,
            objective_config,
            feature_config,
        ), details

    for label in labels:
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
    if len(objective_values) != len(labels):
        raise AssertionError("Objective vector length does not match configured labels")
    return tuple(objective_values), details

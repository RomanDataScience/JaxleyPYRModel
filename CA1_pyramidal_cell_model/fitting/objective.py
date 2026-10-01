"""Joint voltage objective for four depolarizing and one hyperpolarizing trace."""

from __future__ import annotations

import numpy as np

from .data import Trace


def _spikes(time: np.ndarray, voltage: np.ndarray, start: float, stop: float,
            threshold: float = -20.0) -> list[tuple[float, float]]:
    mask = (time >= start) & (time <= stop)
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        return []
    result: list[tuple[float, float]] = []
    last = -np.inf
    dt = float(np.median(np.diff(time)))
    for index in indices[1:-1]:
        if voltage[index] < threshold or voltage[index] < voltage[index - 1] or voltage[index] < voltage[index + 1]:
            continue
        if time[index] - last < 2.0:
            continue
        left = max(indices[0], index - max(1, int(round(1.0 / dt))))
        if voltage[index] - voltage[left] < 5.0:
            continue
        result.append((float(time[index]), float(voltage[index])))
        last = time[index]
    return result


def _region_loss(error: np.ndarray, mask: np.ndarray, sigma: float) -> float:
    if not mask.any():
        return 0.0
    normalized = error[mask] / max(float(sigma), 1.0e-9)
    # Huber loss prevents a bad early candidate from dominating the whole CMA population.
    absolute = np.abs(normalized)
    return float(np.mean(np.where(absolute <= 3.0, 0.5 * normalized**2, 3.0 * absolute - 4.5)))


def _firing_rate_hz(spikes: list[tuple[float, float]], start: float, stop: float) -> float:
    duration_ms = float(stop) - float(start)
    if duration_ms <= 0.0:
        raise ValueError(f"Invalid spike-count epoch: start={start}, stop={stop}")
    return float(len(spikes) * 1000.0 / duration_ms)


def score_trace(trace: Trace, simulated_time: np.ndarray, simulated_voltage: np.ndarray,
                *, threshold_mV: float = -20.0, firing_rate_weight: float = 0.0,
                firing_rate_scale_hz: float = 10.0) -> tuple[float, dict]:
    if not np.isfinite([firing_rate_weight, firing_rate_scale_hz]).all():
        raise ValueError("firing_rate_weight and firing_rate_scale_hz must be finite")
    if firing_rate_weight < 0.0:
        raise ValueError("firing_rate_weight must be non-negative")
    if firing_rate_scale_hz <= 0.0:
        raise ValueError("firing_rate_scale_hz must be positive")
    sim = np.interp(trace.time_ms, simulated_time, simulated_voltage)
    pre = trace.time_ms <= trace.epoch_start_ms
    step = (trace.time_ms >= trace.epoch_start_ms) & (trace.time_ms <= trace.epoch_stop_ms)
    recovery = trace.time_ms >= trace.epoch_stop_ms
    exp_baseline = float(np.median(trace.voltage_mV[pre]))
    sim_baseline = float(np.median(sim[pre]))
    error = sim - trace.voltage_mV
    centered_error = (sim - sim_baseline) - (trace.voltage_mV - exp_baseline)
    if trace.protocol == "hyperpolarizing_pulse":
        components = {
            "pre": _region_loss(centered_error, pre, 1.5),
            "step": _region_loss(centered_error, step, 1.0),
            "recovery": _region_loss(centered_error, recovery, 1.5),
            "baseline": _region_loss(np.asarray([sim_baseline - exp_baseline]), np.asarray([True]), 1.5),
        }
        exp_deflection = float(np.min((trace.voltage_mV - exp_baseline)[step]))
        sim_deflection = float(np.min((sim - sim_baseline)[step]))
        components["deflection"] = float(((sim_deflection - exp_deflection) / 1.0) ** 2)
        value = (0.5 * components["pre"] + 3.0 * components["step"] +
                 2.0 * components["recovery"] + components["baseline"] + components["deflection"]) / 7.5
        details = {"kind": "hyperpolarizing", "components": components,
                   "baseline_mV": {"experimental": exp_baseline, "simulated": sim_baseline},
                   "deflection_mV": {"experimental": exp_deflection, "simulated": sim_deflection}}
    else:
        components = {
            "trajectory": (_region_loss(error, pre, 2.0) + 2.0 * _region_loss(error, step, 2.0) +
                            _region_loss(error, recovery, 2.0)) / 4.0,
            "centered_trajectory": (_region_loss(centered_error, pre, 2.0) +
                                     2.0 * _region_loss(centered_error, step, 2.0) +
                                     _region_loss(centered_error, recovery, 2.0)) / 4.0,
            "baseline": ((sim_baseline - exp_baseline) / 2.0) ** 2,
        }
        exp_spikes = _spikes(trace.time_ms, trace.voltage_mV, trace.epoch_start_ms, trace.epoch_stop_ms, threshold_mV)
        sim_spikes = _spikes(trace.time_ms, sim, trace.epoch_start_ms, trace.epoch_stop_ms, threshold_mV)
        components["spike_count"] = 2.0 * abs(len(exp_spikes) - len(sim_spikes))
        exp_rate_hz = _firing_rate_hz(exp_spikes, trace.epoch_start_ms, trace.epoch_stop_ms)
        sim_rate_hz = _firing_rate_hz(sim_spikes, trace.epoch_start_ms, trace.epoch_stop_ms)
        components["firing_rate"] = ((sim_rate_hz - exp_rate_hz) /
                                      max(float(firing_rate_scale_hz), 1.0e-9)) ** 2
        pairs = min(len(exp_spikes), len(sim_spikes))
        if pairs:
            timing = [(sim_spikes[i][0] - exp_spikes[i][0]) / 5.0 for i in range(pairs)]
            height = [(sim_spikes[i][1] - exp_spikes[i][1]) / 10.0 for i in range(pairs)]
            components["spike_timing"] = float(np.mean(np.square(timing)))
            components["spike_height"] = float(np.mean(np.square(height)))
        else:
            components["spike_timing"] = 0.0
            components["spike_height"] = 0.0
        value = (components["trajectory"] + components["centered_trajectory"] +
                 components["baseline"] + components["spike_count"] +
                 components["spike_timing"] + components["spike_height"] +
                 float(firing_rate_weight) * components["firing_rate"])
        details = {"kind": "depolarizing", "components": components,
                   "experimental_spikes": exp_spikes, "simulated_spikes": sim_spikes,
                   "firing_rate_hz": {"experimental": exp_rate_hz, "simulated": sim_rate_hz},
                   "baseline_mV": {"experimental": exp_baseline, "simulated": sim_baseline}}
    return float(value), details


def joint_score(depolarizing: list[Trace], hyperpolarizing: Trace, simulations: list[tuple[np.ndarray, np.ndarray]],
                *, hyper_weight: float = 2.0, firing_rate_weight: float = 0.0,
                firing_rate_scale_hz: float = 10.0,
                include_details: bool = True) -> tuple[float, dict]:
    if not np.isfinite([hyper_weight, firing_rate_weight, firing_rate_scale_hz]).all():
        raise ValueError("Objective weights and firing_rate_scale_hz must be finite")
    if firing_rate_weight < 0.0:
        raise ValueError("firing_rate_weight must be non-negative")
    if firing_rate_scale_hz <= 0.0:
        raise ValueError("firing_rate_scale_hz must be positive")
    if len(simulations) != len(depolarizing) + 1:
        raise ValueError("Expected one simulation per configured trace")
    details = []
    losses = []
    for trace, simulation in zip([*depolarizing, hyperpolarizing], simulations, strict=True):
        loss, trace_details = score_trace(
            trace, *simulation, firing_rate_weight=firing_rate_weight,
            firing_rate_scale_hz=firing_rate_scale_hz,
        )
        if trace is hyperpolarizing:
            loss *= float(hyper_weight)
        losses.append(loss)
        if include_details:
            details.append({"trace": trace.name, "loss": loss, **trace_details})
    return float(np.mean(losses)), {
        "traces": details,
        "hyper_weight": hyper_weight,
        "firing_rate_weight": firing_rate_weight,
        "firing_rate_scale_hz": firing_rate_scale_hz,
    }

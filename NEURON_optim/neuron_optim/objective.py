"""Spike detection and stage-specific finite scalar objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .data import Trace


@dataclass(frozen=True)
class Spike:
    threshold_ms: float
    peak_ms: float
    peak_mV: float


@dataclass(frozen=True)
class SimulationOutput:
    time_ms: np.ndarray
    voltage_mV: np.ndarray


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    details: dict


@dataclass(frozen=True)
class TraceObjectiveFeatures:
    """Trace-derived values that are invariant across candidates."""

    dt_ms: float
    pre_mask: np.ndarray
    step_mask: np.ndarray
    recovery_mask: np.ndarray
    experimental_centered: np.ndarray
    experimental_spikes: tuple[Spike, ...]
    baseline_mV: float
    terminal_mask: np.ndarray
    trajectory_weights: np.ndarray


@dataclass(frozen=True)
class ObjectiveContext:
    """Precomputed objective inputs shared by all candidates in a study."""

    stage: str
    traces: tuple[Trace, ...]
    features: tuple[TraceObjectiveFeatures, ...]


def build_objective_context(
    traces: Iterable[Trace], *, stage: str,
    threshold_mV: float = -20.0, refractory_ms: float = 2.0,
    prominence_mV: float = 5.0,
) -> ObjectiveContext:
    """Precompute experimental features once for a study."""
    trace_tuple = tuple(traces)
    features = []
    for trace in trace_tuple:
        dt = float(np.median(np.diff(trace.time_ms)))
        pre_mask = trace.time_ms <= trace.epoch_start_ms
        if not pre_mask.any():
            pre_mask = np.ones(trace.time_ms.shape, dtype=bool)
        step_mask = ((trace.time_ms >= trace.epoch_start_ms) &
                     (trace.time_ms <= trace.epoch_stop_ms))
        recovery_mask = trace.time_ms >= trace.epoch_stop_ms
        baseline = float(np.median(trace.voltage_mV[pre_mask]))
        experimental_centered = trace.voltage_mV - baseline
        experimental_spikes = tuple(
            detect_spikes(trace.time_ms, trace.voltage_mV,
                          threshold_mV=threshold_mV,
                          refractory_ms=refractory_ms,
                          prominence_mV=prominence_mV,
                          dt_ms=dt)
        ) if stage == "depolarizing" else ()
        terminal_count = max(1, int(round(20.0 / dt)))
        terminal_mask = recovery_mask & (
            np.arange(trace.time_ms.size) >= trace.time_ms.size - terminal_count
        )
        trajectory_weights = np.ones(trace.time_ms.size, dtype=float)
        features.append(TraceObjectiveFeatures(
            dt_ms=dt,
            pre_mask=pre_mask,
            step_mask=step_mask,
            recovery_mask=recovery_mask,
            experimental_centered=experimental_centered,
            experimental_spikes=experimental_spikes,
            baseline_mV=float(np.median(trace.voltage_mV[trace.time_ms < trace.epoch_start_ms]))
            if (trace.time_ms < trace.epoch_start_ms).any() else float(trace.voltage_mV[0]),
            terminal_mask=terminal_mask,
            trajectory_weights=trajectory_weights,
        ))
    return ObjectiveContext(stage, trace_tuple, tuple(features))


def detect_spikes(time_ms: np.ndarray, voltage_mV: np.ndarray, *,
                  threshold_mV: float = -20.0,
                  refractory_ms: float = 2.0,
                  prominence_mV: float = 5.0,
                  dt_ms: float | None = None) -> list[Spike]:
    time = np.asarray(time_ms, dtype=float)
    voltage = np.asarray(voltage_mV, dtype=float)
    crossings = np.flatnonzero((voltage[:-1] < threshold_mV) &
                               (voltage[1:] >= threshold_mV))
    dt = float(np.median(np.diff(time))) if dt_ms is None else float(dt_ms)
    result: list[Spike] = []
    for index in crossings:
        if result and time[index] - result[-1].threshold_ms < refractory_ms:
            continue
        end = min(voltage.size, index + max(2, int(round(5.0 / dt))))
        peak_index = index + int(np.argmax(voltage[index:end]))
        left = max(0, index - int(round(1.0 / dt)))
        if voltage[peak_index] - voltage[left] < prominence_mV:
            continue
        fraction = (threshold_mV - voltage[index]) / max(voltage[index + 1] - voltage[index], 1e-12)
        threshold_time = time[index] + np.clip(fraction, 0.0, 1.0) * (time[index + 1] - time[index])
        result.append(Spike(float(threshold_time), float(time[peak_index]), float(voltage[peak_index])))
    return result


def _kernel_loss(a: np.ndarray, b: np.ndarray, sigma_mV: float,
                 weights: np.ndarray | None = None) -> float:
    if sigma_mV <= 0 or a.size == 0 or a.shape != b.shape:
        return 1.0e6
    values = 1.0 - np.exp(-((a - b) ** 2) / (sigma_mV ** 2))
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != values.shape or not np.isfinite(weights).all() or weights.sum() <= 0:
        return 1.0e6
    return float(np.sum(weights * values) / np.sum(weights))


def _interp(sim: SimulationOutput, target_time: np.ndarray) -> np.ndarray:
    return np.interp(target_time, sim.time_ms, sim.voltage_mV)


def _region_mask(trace: Trace, region: str) -> np.ndarray:
    if region == "pre":
        return trace.time_ms <= trace.epoch_start_ms
    if region == "step":
        return ((trace.time_ms >= trace.epoch_start_ms) &
                (trace.time_ms <= trace.epoch_stop_ms))
    if region == "recovery":
        return trace.time_ms >= trace.epoch_stop_ms
    raise ValueError(region)


def _baseline_centered_voltage(
    trace: Trace, simulated: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return experimental and simulated voltage deflections from baseline.

    Keep absolute voltage for channel dynamics and spike detection, but remove
    the trace-specific DC voltage offset from the shape comparison.
    """
    pre_mask = trace.time_ms <= trace.epoch_start_ms
    if not pre_mask.any():
        pre_mask = np.ones(trace.time_ms.shape, dtype=bool)
    experimental_baseline = float(np.median(trace.voltage_mV[pre_mask]))
    simulated_baseline = float(np.median(simulated[pre_mask]))
    return (
        trace.voltage_mV - experimental_baseline,
        simulated - simulated_baseline,
    )


def hyperpolarizing_objective(
    traces: Iterable[Trace], simulations: Iterable[SimulationOutput], *,
    sigma_mV: float = 1.0, region_weights: Mapping[str, float] | None = None,
    deflection_weight: float = 2.0, sigma_deflection_mV: float = 1.0,
    delta_v_weight: float = 1.0, voltage_offset_weight: float = 1.0,
    sigma_offset_mV: float = 1.0,
    threshold_mV: float = -20.0, refractory_ms: float = 2.0,
    prominence_mV: float = 5.0, spike_penalty: float = 1.0e4,
    context: ObjectiveContext | None = None,
    include_details: bool = True,
) -> ObjectiveResult:
    weights = {"pre": 1.0, "step": 4.0, "recovery": 3.0}
    weights.update(region_weights or {})
    context = context or build_objective_context(
        traces, stage="hyper", threshold_mV=threshold_mV,
        refractory_ms=refractory_ms, prominence_mV=prominence_mV,
    )
    traces = context.traces
    trace_details = []
    values = []
    for trace, simulation, features in zip(traces, simulations, context.features, strict=True):
        simulated = _interp(simulation, trace.time_ms)
        spikes = detect_spikes(trace.time_ms, simulated, threshold_mV=threshold_mV,
                               refractory_ms=refractory_ms, prominence_mV=prominence_mV,
                               dt_ms=features.dt_ms)
        experimental_centered = features.experimental_centered
        simulated_baseline = float(np.median(simulated[features.pre_mask]))
        simulated_centered = simulated - simulated_baseline
        in_step = [spike for spike in spikes
                   if trace.epoch_start_ms <= spike.peak_ms <= trace.epoch_stop_ms]
        losses: dict[str, float] = {}
        # The trajectory terms compare baseline-centered voltage (Delta_V),
        # while a separate term scores the absolute pre-pulse voltage offset.
        # This preserves waveform shape and makes resting-voltage alignment an
        # explicit, independently weighted part of the objective.
        for region, mask in (("pre", features.pre_mask),
                             ("step", features.step_mask),
                             ("recovery", features.recovery_mask)):
            losses[region] = _kernel_loss(
                simulated_centered[mask], experimental_centered[mask], sigma_mV
            )
        step_mask = features.step_mask
        if not step_mask.any():
            deflection_loss = 1.0e6
            experimental_deflection = float("nan")
            simulated_deflection = float("nan")
        else:
            # Compare the pre-pulse baseline with the largest deflection during
            # the pulse. The waveform terms still score the complete trajectory.
            experimental_deflection = float(np.min(experimental_centered[step_mask]))
            simulated_deflection = float(np.min(simulated_centered[step_mask]))
            deflection_loss = _kernel_loss(
                np.asarray([simulated_deflection]),
                np.asarray([experimental_deflection]),
                sigma_deflection_mV,
            )
        losses["deflection"] = deflection_loss
        delta_v_loss = (
            sum(weights[name] * losses[name] for name in ("pre", "step", "recovery"))
            + float(deflection_weight) * deflection_loss
        ) / (sum(weights.values()) + float(deflection_weight))
        experimental_baseline = float(np.median(trace.voltage_mV[features.pre_mask]))
        voltage_offset_loss = _kernel_loss(
            np.asarray([simulated_baseline]),
            np.asarray([experimental_baseline]),
            sigma_offset_mV,
        )
        losses["delta_v"] = delta_v_loss
        losses["voltage_offset"] = voltage_offset_loss
        component_weight_total = float(delta_v_weight) + float(voltage_offset_weight)
        if component_weight_total <= 0.0:
            raise ValueError("delta_v_weight + voltage_offset_weight must be positive")
        total = (
            float(delta_v_weight) * delta_v_loss
            + float(voltage_offset_weight) * voltage_offset_loss
        ) / component_weight_total
        penalized = bool(in_step)
        if penalized:
            total = float(spike_penalty)
        values.append(total)
        if include_details:
            trace_details.append({"trace": trace.trace, "losses": losses,
                                  "deflection_mV": {
                                      "experimental": experimental_deflection,
                                      "simulated": simulated_deflection,
                                      "loss": deflection_loss,
                                  },
                                  "delta_v_loss": delta_v_loss,
                                  "voltage_offset_mV": {
                                      "experimental": experimental_baseline,
                                      "simulated": simulated_baseline,
                                      "error": simulated_baseline - experimental_baseline,
                                      "loss": voltage_offset_loss,
                                  },
                                  "component_weights": {
                                      "delta_v": float(delta_v_weight),
                                      "voltage_offset": float(voltage_offset_weight),
                                  },
                                  "spikes_ms": [s.peak_ms for s in spikes],
                                  "in_step_spikes_ms": [s.peak_ms for s in in_step],
                                  "penalized": penalized})
    details = {"traces": trace_details, "penalty": spike_penalty} if include_details else {}
    return ObjectiveResult(float(np.mean(values)), details)


def _matched_spike_loss(trace: Trace, sim: np.ndarray, exp_spikes: list[Spike],
                        sim_spikes: list[Spike], *, sigma_mV: float,
                        window_ms: float, unmatched_penalty: float,
                        dt_ms: float | None = None) -> tuple[float, dict]:
    pairs = min(len(exp_spikes), len(sim_spikes))
    losses: list[float] = []
    dt = float(np.median(np.diff(trace.time_ms))) if dt_ms is None else float(dt_ms)
    relative = np.arange(-window_ms, window_ms + 0.5 * dt, dt)
    for expected, actual in zip(exp_spikes[:pairs], sim_spikes[:pairs], strict=True):
        exp_values = np.interp(expected.peak_ms + relative, trace.time_ms, trace.voltage_mV)
        sim_values = np.interp(actual.peak_ms + relative, trace.time_ms, sim)
        losses.append(_kernel_loss(sim_values, exp_values, sigma_mV))
    unmatched = abs(len(exp_spikes) - len(sim_spikes))
    value = float(np.mean(losses)) if losses else 0.0
    value += unmatched_penalty * unmatched
    return value, {"matched": pairs, "unmatched": unmatched, "losses": losses}


def depolarizing_objective(
    traces: Iterable[Trace], simulations: Iterable[SimulationOutput], *,
    sigma_trajectory_mV: float = 2.0, sigma_spike_mV: float = 2.0,
    sigma_plateau_mV: float = 2.0, sigma_recovery_mV: float = 2.0,
    sigma_terminal_mV: float = 2.0, weights: Mapping[str, float] | None = None,
    threshold_mV: float = -20.0, refractory_ms: float = 2.0,
    prominence_mV: float = 5.0, spike_window_ms: float = 5.0,
    plateau_exclusion_ms: float = 3.0, unmatched_spike_penalty: float = 100.0,
    invalid_plateau_penalty: float = 100.0,
    extra_spike_penalty: float = 1.0e4,
    return_alpha: float = 0.7,
    context: ObjectiveContext | None = None,
    include_details: bool = True,
) -> ObjectiveResult:
    component_weights = {"trajectory": 1.0, "spike_shape": 3.0,
                         "plateau": 2.0, "return_baseline": 3.0}
    component_weights.update(weights or {})
    context = context or build_objective_context(
        traces, stage="depolarizing", threshold_mV=threshold_mV,
        refractory_ms=refractory_ms, prominence_mV=prominence_mV,
    )
    traces = context.traces
    all_details = []
    totals = []
    for trace, simulation, features in zip(traces, simulations, context.features, strict=True):
        sim = _interp(simulation, trace.time_ms)
        exp_spikes = [s for s in features.experimental_spikes
                      if trace.epoch_start_ms <= s.peak_ms <= trace.epoch_stop_ms]
        sim_spikes_all = detect_spikes(trace.time_ms, sim, threshold_mV=threshold_mV,
                                       refractory_ms=refractory_ms, prominence_mV=prominence_mV,
                                       dt_ms=features.dt_ms)
        sim_spikes = [s for s in sim_spikes_all if trace.epoch_start_ms <= s.peak_ms <= trace.epoch_stop_ms]
        allowed_start = trace.epoch_start_ms - 50.0
        allowed_stop = trace.epoch_stop_ms + 50.0
        outside = [s for s in sim_spikes_all if s.peak_ms < allowed_start or s.peak_ms > allowed_stop]

        l_trajectory = _kernel_loss(
            sim, trace.voltage_mV, sigma_trajectory_mV, features.trajectory_weights
        )
        l_spike, spike_details = _matched_spike_loss(
            trace, sim, exp_spikes, sim_spikes, sigma_mV=sigma_spike_mV,
            window_ms=spike_window_ms, unmatched_penalty=unmatched_spike_penalty,
            dt_ms=features.dt_ms)

        step_mask = features.step_mask
        exclusion = np.zeros(trace.time_ms.size, dtype=bool)
        for spike in exp_spikes + sim_spikes:
            exclusion |= np.abs(trace.time_ms - spike.peak_ms) <= plateau_exclusion_ms
        plateau_mask = step_mask & ~exclusion
        l_plateau = (invalid_plateau_penalty if not plateau_mask.any() else
                     _kernel_loss(sim[plateau_mask], trace.voltage_mV[plateau_mask], sigma_plateau_mV))

        recovery_mask = features.recovery_mask
        l_recovery = _kernel_loss(sim[recovery_mask], trace.voltage_mV[recovery_mask], sigma_recovery_mV)
        terminal = features.terminal_mask
        baseline = features.baseline_mV
        terminal_value = float(np.mean(sim[terminal])) if terminal.any() else float(sim[-1])
        l_terminal = float(1.0 - np.exp(-((terminal_value - baseline) ** 2) / sigma_terminal_mV ** 2))
        l_return = return_alpha * l_recovery + (1.0 - return_alpha) * l_terminal

        components = {"trajectory": l_trajectory, "spike_shape": l_spike,
                      "plateau": l_plateau, "return_baseline": l_return}
        total = sum(component_weights[key] * components[key] for key in components) / sum(component_weights.values())
        penalized = len(outside) > 1
        if penalized:
            total = float(extra_spike_penalty)
        totals.append(total)
        if include_details:
            all_details.append({"trace": trace.trace, "components": components,
                                "spikes_experimental_ms": [s.peak_ms for s in exp_spikes],
                                "spikes_simulated_ms": [s.peak_ms for s in sim_spikes_all],
                                "out_of_window_spikes_ms": [s.peak_ms for s in outside],
                                "penalized": penalized, "spike_shape": spike_details})
    details = {"traces": all_details, "weights": component_weights} if include_details else {}
    return ObjectiveResult(float(np.mean(totals)), details)

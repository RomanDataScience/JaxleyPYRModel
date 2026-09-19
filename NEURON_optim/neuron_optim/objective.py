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


def detect_spikes(time_ms: np.ndarray, voltage_mV: np.ndarray, *,
                  threshold_mV: float = -20.0,
                  refractory_ms: float = 2.0,
                  prominence_mV: float = 5.0) -> list[Spike]:
    time = np.asarray(time_ms, dtype=float)
    voltage = np.asarray(voltage_mV, dtype=float)
    crossings = np.flatnonzero((voltage[:-1] < threshold_mV) &
                               (voltage[1:] >= threshold_mV))
    result: list[Spike] = []
    for index in crossings:
        if result and time[index] - result[-1].threshold_ms < refractory_ms:
            continue
        end = min(voltage.size, index + max(2, int(round(5.0 / np.median(np.diff(time))))) )
        peak_index = index + int(np.argmax(voltage[index:end]))
        left = max(0, index - int(round(1.0 / np.median(np.diff(time)))))
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
    threshold_mV: float = -20.0, refractory_ms: float = 2.0,
    prominence_mV: float = 5.0, spike_penalty: float = 1.0e4,
) -> ObjectiveResult:
    weights = {"pre": 1.0, "step": 4.0, "recovery": 3.0}
    weights.update(region_weights or {})
    trace_details = []
    values = []
    for trace, simulation in zip(traces, simulations, strict=True):
        simulated = _interp(simulation, trace.time_ms)
        spikes = detect_spikes(trace.time_ms, simulated, threshold_mV=threshold_mV,
                               refractory_ms=refractory_ms, prominence_mV=prominence_mV)
        experimental_centered, simulated_centered = _baseline_centered_voltage(trace, simulated)
        in_step = [spike for spike in spikes if trace.epoch_start_ms <= spike.peak_ms <= trace.epoch_stop_ms]
        losses: dict[str, float] = {}
        for region in ("pre", "step", "recovery"):
            mask = _region_mask(trace, region)
            losses[region] = _kernel_loss(
                simulated_centered[mask], experimental_centered[mask], sigma_mV
            )
        step_mask = _region_mask(trace, "step")
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
        total = (
            sum(weights[name] * losses[name] for name in ("pre", "step", "recovery"))
            + float(deflection_weight) * deflection_loss
        ) / (sum(weights.values()) + float(deflection_weight))
        penalized = bool(in_step)
        if penalized:
            total = float(spike_penalty)
        values.append(total)
        trace_details.append({"trace": trace.trace, "losses": losses,
                              "deflection_mV": {
                                  "experimental": experimental_deflection,
                                  "simulated": simulated_deflection,
                                  "loss": deflection_loss,
                              },
                              "spikes_ms": [s.peak_ms for s in spikes],
                              "in_step_spikes_ms": [s.peak_ms for s in in_step],
                              "penalized": penalized})
    return ObjectiveResult(float(np.mean(values)), {"traces": trace_details,
                                                     "penalty": spike_penalty})


def _matched_spike_loss(trace: Trace, sim: np.ndarray, exp_spikes: list[Spike],
                        sim_spikes: list[Spike], *, sigma_mV: float,
                        window_ms: float, unmatched_penalty: float) -> tuple[float, dict]:
    pairs = min(len(exp_spikes), len(sim_spikes))
    losses: list[float] = []
    for expected, actual in zip(exp_spikes[:pairs], sim_spikes[:pairs], strict=True):
        relative = np.arange(-window_ms, window_ms + 0.5 * np.median(np.diff(trace.time_ms)),
                             np.median(np.diff(trace.time_ms)))
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
) -> ObjectiveResult:
    component_weights = {"trajectory": 1.0, "spike_shape": 3.0,
                         "plateau": 2.0, "return_baseline": 3.0}
    component_weights.update(weights or {})
    all_details = []
    totals = []
    for trace, simulation in zip(traces, simulations, strict=True):
        sim = _interp(simulation, trace.time_ms)
        exp_spikes = [s for s in detect_spikes(trace.time_ms, trace.voltage_mV,
                         threshold_mV=threshold_mV, refractory_ms=refractory_ms,
                         prominence_mV=prominence_mV)
                      if trace.epoch_start_ms <= s.peak_ms <= trace.epoch_stop_ms]
        sim_spikes_all = detect_spikes(trace.time_ms, sim, threshold_mV=threshold_mV,
                                       refractory_ms=refractory_ms, prominence_mV=prominence_mV)
        sim_spikes = [s for s in sim_spikes_all if trace.epoch_start_ms <= s.peak_ms <= trace.epoch_stop_ms]
        allowed_start = trace.epoch_start_ms - 50.0
        allowed_stop = trace.epoch_stop_ms + 50.0
        outside = [s for s in sim_spikes_all if s.peak_ms < allowed_start or s.peak_ms > allowed_stop]

        trajectory = np.ones(trace.time_ms.size)
        for region, factor in (("pre", 1.0), ("step", 1.0), ("recovery", 1.0)):
            trajectory[_region_mask(trace, region)] = factor
        l_trajectory = _kernel_loss(sim, trace.voltage_mV, sigma_trajectory_mV, trajectory)
        l_spike, spike_details = _matched_spike_loss(
            trace, sim, exp_spikes, sim_spikes, sigma_mV=sigma_spike_mV,
            window_ms=spike_window_ms, unmatched_penalty=unmatched_spike_penalty)

        step_mask = _region_mask(trace, "step")
        exclusion = np.zeros(trace.time_ms.size, dtype=bool)
        for spike in exp_spikes + sim_spikes:
            exclusion |= np.abs(trace.time_ms - spike.peak_ms) <= plateau_exclusion_ms
        plateau_mask = step_mask & ~exclusion
        l_plateau = (invalid_plateau_penalty if not plateau_mask.any() else
                     _kernel_loss(sim[plateau_mask], trace.voltage_mV[plateau_mask], sigma_plateau_mV))

        recovery_mask = _region_mask(trace, "recovery")
        l_recovery = _kernel_loss(sim[recovery_mask], trace.voltage_mV[recovery_mask], sigma_recovery_mV)
        terminal_count = max(1, int(round(20.0 / np.median(np.diff(trace.time_ms)))))
        terminal = recovery_mask & (np.arange(trace.time_ms.size) >= trace.time_ms.size - terminal_count)
        baseline_mask = trace.time_ms < trace.epoch_start_ms
        baseline = float(np.median(trace.voltage_mV[baseline_mask])) if baseline_mask.any() else float(trace.voltage_mV[0])
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
        all_details.append({"trace": trace.trace, "components": components,
                            "spikes_experimental_ms": [s.peak_ms for s in exp_spikes],
                            "spikes_simulated_ms": [s.peak_ms for s in sim_spikes_all],
                            "out_of_window_spikes_ms": [s.peak_ms for s in outside],
                            "penalized": penalized, "spike_shape": spike_details})
    return ObjectiveResult(float(np.mean(totals)), {"traces": all_details,
                                                     "weights": component_weights})

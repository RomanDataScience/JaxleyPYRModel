"""Spike-onset metrics of one trace (from ``OptimizerNEURON_Combe/scripts/sweep_onset_plateau.py``)."""

from __future__ import annotations

import numpy as np

from .objective import _spikes


def onset_metrics(time: np.ndarray, voltage: np.ndarray, start: float, stop: float) -> dict:
    """First-spike latency and threshold, AHP troughs and inter-spike voltage."""
    spikes = _spikes(time, voltage, start, stop)
    dvdt = np.gradient(voltage, time)
    result: dict = {"n_spikes": len(spikes)}
    if not spikes:
        mask = (time >= start + 50.0) & (time <= stop)
        result.update(latency_ms=None, threshold_mV=None, mean_isi_mV=float(np.mean(voltage[mask])))
        return result
    first = spikes[0][0]
    result["latency_ms"] = first - start
    window = np.flatnonzero((time >= start) & (time <= first) & (dvdt >= 10.0))
    result["threshold_mV"] = float(voltage[window[0]]) if window.size else None
    result["isi_ms"] = [b[0] - a[0] for a, b in zip(spikes[:4], spikes[1:5])]
    troughs, between = [], []
    for (t0, _), (t1, _) in zip(spikes, spikes[1:]):
        segment = (time > t0 + 1.0) & (time < t1 - 1.0)
        if segment.any():
            troughs.append(float(np.min(voltage[segment])))
            between.append(voltage[segment])
    result["ahp_trough_mV"] = float(np.median(troughs)) if troughs else None
    result["mean_isi_mV"] = float(np.mean(np.concatenate(between))) if between else None
    return result

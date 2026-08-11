"""Auditable, non-differentiable feature metrics for candidate reporting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from jaxley_refactored.data import TraceBucket


def spike_feature_metrics(
    buckets: Iterable[TraceBucket],
    predictions: Mapping[tuple[float, int], np.ndarray],
    *,
    threshold_mV: float = -20.0,
) -> dict:
    """Return per-trace hard spike counts/rates inside stimulus windows."""

    traces = []
    rate_errors = []
    total_forbidden_spikes = 0
    for bucket in buckets:
        predicted = np.asarray(predictions[bucket.key])
        stimulus_masks = np.asarray(bucket.window_masks["stimulus"], dtype=bool)
        for row, record in enumerate(bucket.records):
            size = len(record.time_ms)
            mask = stimulus_masks[row, :size]
            observed = np.asarray(record.voltage_mV)
            simulated = predicted[row, :size]
            pair_mask = mask[:-1] & mask[1:]

            def count(voltage, crossing_mask):
                crossings = (voltage[:-1] < threshold_mV) & (
                    voltage[1:] >= threshold_mV
                )
                return int(np.sum(crossing_mask & crossings))

            observed_count = count(observed, pair_mask)
            simulated_count = count(simulated, pair_mask)
            if record.protocol == "depolarizing_step":
                forbidden_mask = np.asarray(
                    bucket.window_masks["outside_stimulus"][row, :size],
                    dtype=bool,
                )[1:]
            elif record.protocol == "hyperpolarizing_pulse":
                forbidden_mask = np.ones(size - 1, dtype=bool)
            else:
                forbidden_mask = np.zeros(size - 1, dtype=bool)
            forbidden_count = count(simulated, forbidden_mask)
            total_forbidden_spikes += forbidden_count
            duration_s = max(float(np.sum(pair_mask) * record.dt_ms * 1e-3), 1e-9)
            observed_rate = observed_count / duration_s
            simulated_rate = simulated_count / duration_s
            error = simulated_rate - observed_rate
            rate_errors.append(error)
            traces.append(
                {
                    "trace_key": record.trace_key,
                    "threshold_mV": float(threshold_mV),
                    "stimulus_duration_s": duration_s,
                    "experimental_spike_count": observed_count,
                    "simulated_spike_count": simulated_count,
                    "experimental_rate_hz": observed_rate,
                    "simulated_rate_hz": simulated_rate,
                    "rate_error_hz": error,
                    "forbidden_spike_count": forbidden_count,
                }
            )
    errors = np.asarray(rate_errors, dtype=float)
    return {
        "spiking": {
            "threshold_mV": float(threshold_mV),
            "rms_rate_error_hz": (
                float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0
            ),
            "mean_absolute_rate_error_hz": (
                float(np.mean(np.abs(errors))) if len(errors) else 0.0
            ),
            "traces": traces,
        },
        "constraints": {
            "eligible": total_forbidden_spikes == 0,
            "forbidden_spike_count": total_forbidden_spikes,
        },
    }

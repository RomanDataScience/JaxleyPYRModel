"""Study steady-state and driven state occupancy of ``Nav16_a.mod``.

This is intentionally dependency-light: it implements the four-state kinetic
scheme in ``channels_converted/mod/Nav16_a.mod`` directly, so it can be used
without importing JAX/Jaxley or building the full morphology.  The sampled
parameter ranges are the active Nav1.6 ranges in ``neuron_optim.parameters``.

The state order is C1, O1, I1, I2.  ``O1`` is the open fraction and therefore
the fraction multiplying ``gbar * (v - ena)``.  The optional driven analysis
integrates the channel under the measured v75ctrl voltage trace using the same
implicit-Euler update used by the Jaxley translation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = ("C1", "O1", "I1", "I2")
NAV16_RANGES = {
    "C1O1v2": (-43.0, -41.0),
    "C1O1k2": (-3.75, -3.25),
    "C1I1b2": (0.15, 0.20),
    "C1I1v2": (-53.0, -50.0),
    "C1I1k2": (-10.0, -8.0),
    "O1I1b2": (8.0, 10.0),
    "O1I1v2": (0.0, 5.0),
    "O1I1k2": (-10.0, -8.0),
    "persist": (0.004, 0.010),
    "persist_vhalf": (-56.5, -55.0),
    "persist_k": (-0.8, -0.4),
}

# Constants copied from channels_converted/mod/Nav16_a.mod.  These are the
# parameters that are not exposed in the current optimization catalog.
FIXED = {
    "C1O1b2": 14.0,
    "O1C1b1": 4.0,
    "O1C1v1": -48.0,
    "O1C1k1": 9.0,
    "O1C1b2": 0.0,
    "O1C1v2": 0.0,
    "O1C1k2": -5.1,
    "O1I1b1": 1.0,
    "O1I1v1": -42.0,
    "O1I1k1": 12.0,
    "I1C1b1": 0.2,
    "I1C1v1": -65.0,
    "I1C1k1": 10.0,
    "I1I2b2": 0.022,
    "I1I2v2": -25.0,
    "I1I2k2": -5.0,
    "I2I1b1": 0.0018,
    "I2I1v1": -50.0,
    "I2I1k1": 12.0,
}


def _rate(v: float | np.ndarray, b: float | np.ndarray,
          v_half: float | np.ndarray, slope: float | np.ndarray):
    argument = np.clip((v - v_half) / slope, -50.0, 50.0)
    return b / (1.0 + np.exp(argument))


def rates(v: float, params: dict[str, float], *, dist: float = 1.35,
          slowdown: float = 0.15, fast_scale: float = 0.7,
          slow_scale: float = 1.0) -> np.ndarray:
    """Return the eight directed rates in the order used by Nav16_a.mod."""
    p = FIXED | params
    q10 = 3.0 ** 1.4  # celsius=34 degC in the Combe model.
    c1o1 = q10 * _rate(v, p["C1O1b2"], p["C1O1v2"], p["C1O1k2"])
    o1c1 = q10 * (
        _rate(v, p["O1C1b1"], p["O1C1v1"], p["O1C1k1"])
        + _rate(v, p["O1C1b2"], p["O1C1v2"], p["O1C1k2"])
    )
    o1i1 = 0.5 * q10 * (
        _rate(v, p["O1I1b1"], p["O1I1v1"], p["O1I1k1"])
        + _rate(v, p["O1I1b2"], p["O1I1v2"], p["O1I1k2"])
    ) / fast_scale
    persist_arg = (v - p["persist_vhalf"]) / p["persist_k"]
    persist_gate = 1.0 / (1.0 + np.exp(np.clip(persist_arg, -50.0, 50.0)))
    i1o1 = p["persist"] * persist_gate * o1i1
    i1c1 = q10 * _rate(v, p["I1C1b1"], p["I1C1v1"], p["I1C1k1"]) / fast_scale
    c1i1 = q10 * _rate(v, p["C1I1b2"], p["C1I1v2"], p["C1I1k2"]) / fast_scale
    i1i2 = slowdown * dist * q10 * _rate(
        v, p["I1I2b2"], p["I1I2v2"], p["I1I2k2"]
    ) / slow_scale
    i2i1 = slowdown * q10 * _rate(
        v, p["I2I1b1"], p["I2I1v1"], p["I2I1k1"]
    ) / slow_scale
    return np.asarray((c1o1, o1c1, o1i1, i1o1, i1c1, c1i1, i1i2, i2i1))


def rate_matrix(v: float, params: dict[str, float]) -> np.ndarray:
    c1o1, o1c1, o1i1, i1o1, i1c1, c1i1, i1i2, i2i1 = rates(v, params)
    return np.asarray([
        [-c1o1 - c1i1, o1c1, i1c1, 0.0],
        [c1o1, -o1c1 - o1i1, i1o1, 0.0],
        [c1i1, o1i1, -i1o1 - i1c1 - i1i2, i2i1],
        [0.0, 0.0, i1i2, -i2i1],
    ])


def steady_state(v: float, params: dict[str, float]) -> np.ndarray:
    matrix = rate_matrix(v, params)
    matrix[-1, :] = 1.0
    state = np.linalg.solve(matrix, np.asarray((0.0, 0.0, 0.0, 1.0)))
    state = np.maximum(state, 0.0)
    return state / state.sum()


def sample_parameters(samples: int, seed: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    names = list(NAV16_RANGES)
    values = np.column_stack([
        rng.uniform(low, high, samples)
        for low, high in (NAV16_RANGES[name] for name in names)
    ])
    return names, values


def sampled_states(voltages: np.ndarray, names: list[str], values: np.ndarray):
    states = np.empty((values.shape[0], voltages.size, 4))
    for i, row in enumerate(values):
        params = dict(zip(names, row, strict=True))
        states[i] = np.asarray([steady_state(float(v), params) for v in voltages])
    return states


def driven_states(time_ms: np.ndarray, voltage_mV: np.ndarray,
                  params: dict[str, float]) -> np.ndarray:
    dt = float(np.median(np.diff(time_ms)))
    states = np.empty((time_ms.size, 4))
    state = steady_state(float(voltage_mV[0]), params)
    states[0] = state
    identity = np.eye(4)
    for index in range(1, time_ms.size):
        state = np.linalg.solve(
            identity - dt * rate_matrix(float(voltage_mV[index]), params), state
        )
        state = np.maximum(state, 0.0)
        state /= state.sum()
        states[index] = state
    return states


def _write_csv(path: Path, names: list[str], values: np.ndarray,
               states_by_voltage: dict[float, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", *names, *[
            f"{state}_{voltage:g}mV" for voltage in states_by_voltage
            for state in STATE_NAMES
        ]])
        for index, row in enumerate(values):
            writer.writerow([index, *row, *[
                value for voltage in states_by_voltage
                for value in states_by_voltage[voltage][index]
            ]])


def run(output_dir: Path, samples: int, seed: int,
        trace_root: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    voltages = np.arange(-80.0, 60.01, 0.5)
    names, values = sample_parameters(samples, seed)
    sampled = sampled_states(voltages, names, values)
    state_at = {
        voltage: sampled[:, np.argmin(np.abs(voltages - voltage)), :]
        for voltage in (-65.0, -50.0, -40.0, 0.0, 40.0)
    }
    _write_csv(output_dir / "nav16_state_samples.csv", names, values, state_at)

    peak_indices = sampled[:, :, 1].argmax(axis=1)
    peak_open = sampled[np.arange(samples), peak_indices, 1]
    peak_voltage = voltages[peak_indices]
    summary = {
        "mechanism": "channels_converted/mod/Nav16_a.mod",
        "state_order": list(STATE_NAMES),
        "sample_count": samples,
        "seed": seed,
        "assumptions": {
            "temperature_C": 34.0,
            "soma_dist": 1.35,
            "soma_slowdown": 0.15,
            "fast_inactivation_tau_scale": 0.7,
            "slow_recovery_tau_scale": 1.0,
        },
        "parameter_ranges": NAV16_RANGES,
        "steady_state": {
            str(int(voltage)): {
                state: {
                    "min": float(state_at[voltage][:, i].min()),
                    "median": float(np.median(state_at[voltage][:, i])),
                    "max": float(state_at[voltage][:, i].max()),
                }
                for i, state in enumerate(STATE_NAMES)
            }
            for voltage in state_at
        },
        "open_fraction_peak": {
            "min": float(peak_open.min()),
            "median": float(np.median(peak_open)),
            "max": float(peak_open.max()),
            "voltage_min": float(peak_voltage.min()),
            "voltage_median": float(np.median(peak_voltage)),
            "voltage_max": float(peak_voltage.max()),
        },
    }

    if trace_root is not None:
        time_path = trace_root / "m20240527cd/depolarizing_step/v75ctrl_depolarizing_step_t_ms.txt"
        voltage_path = trace_root / "m20240527cd/depolarizing_step/v75ctrl_depolarizing_step_v.txt"
        time_ms = np.loadtxt(time_path)
        voltage_mV = np.loadtxt(voltage_path)
        dynamic = driven_states(time_ms, voltage_mV,
                                dict(zip(names, values.mean(axis=0), strict=True)))
        dynamic_path = output_dir / "v75ctrl_nav16_states.csv"
        np.savetxt(dynamic_path, np.column_stack((time_ms, voltage_mV, dynamic)),
                   delimiter=",", header="time_ms,voltage_mV,C1,O1,I1,I2", comments="")
        step = (time_ms >= 200.0) & (time_ms <= 500.0)
        positive = step & (voltage_mV > 0.0)
        summary["driven_v75ctrl_mean_parameters"] = {
            state: float(dynamic[step, i].mean()) for i, state in enumerate(STATE_NAMES)
        }
        summary["driven_v75ctrl_peak_positive_voltage"] = {
            state: float(dynamic[positive, i].max()) for i, state in enumerate(STATE_NAMES)
        }
        summary["driven_v75ctrl_peak_open_time_ms"] = float(
            time_ms[np.argmax(np.where(step, dynamic[:, 1], -np.inf))]
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for i, state in enumerate(STATE_NAMES):
        q05, q50, q95 = np.quantile(sampled[:, :, i], (0.05, 0.50, 0.95), axis=0)
        axes[0].fill_between(voltages, q05, q95, alpha=0.16)
        axes[0].plot(voltages, q50, label=state)
    axes[0].axvline(-50.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="Voltage (mV)", ylabel="Steady-state fraction",
                title="Nav16_a steady-state occupancy")
    axes[0].legend(frameon=False, ncol=2)
    if trace_root is not None:
        axes[1].plot(time_ms, voltage_mV, color="black", linewidth=0.7, label="v75ctrl V")
        axes[1].plot(time_ms, dynamic[:, 1] * 100.0, label="O1 × 100")
        axes[1].plot(time_ms, dynamic[:, 2] * 100.0, label="I1 × 100")
        axes[1].plot(time_ms, dynamic[:, 3] * 100.0, label="I2 × 100")
        axes[1].axvspan(200.0, 500.0, color="tab:blue", alpha=0.08)
        axes[1].set(xlabel="Time (ms)", ylabel="mV / percent",
                    title="Driven states under measured v75ctrl")
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].axis("off")
    figure.savefig(output_dir / "nav16_state_region.png", dpi=150)
    plt.close(figure)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--trace-root", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.samples, args.seed, args.trace_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

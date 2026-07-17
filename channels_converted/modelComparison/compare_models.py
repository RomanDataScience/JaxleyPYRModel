from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jaxley_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channels_converted.modelComparison.jaxley_model import run_jaxley_step  # noqa: E402
from channels_converted.modelComparison.neuron_model import run_neuron_step  # noqa: E402
from channels_converted.modelComparison.protocol import StepProtocol  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def voltage_metrics(neuron_trace: dict[str, np.ndarray], jaxley_trace: dict[str, np.ndarray]):
    start = max(neuron_trace["time"][0], jaxley_trace["time"][0])
    stop = min(neuron_trace["time"][-1], jaxley_trace["time"][-1])
    mask = (neuron_trace["time"] >= start) & (neuron_trace["time"] <= stop)
    time = neuron_trace["time"][mask]

    neuron_v = neuron_trace["voltage"][mask]
    jaxley_v = np.interp(time, jaxley_trace["time"], jaxley_trace["voltage"])
    error = jaxley_v - neuron_v

    return {
        "rmse_mV": float(np.sqrt(np.mean(error**2))),
        "mae_mV": float(np.mean(np.abs(error))),
        "max_abs_mV": float(np.max(np.abs(error))),
    }


def save_trace(path: Path, trace: dict[str, np.ndarray]) -> None:
    np.savez(
        path,
        time=trace["time"],
        voltage=trace["voltage"],
        current=trace["current"],
    )


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def plot_comparison(
    path: Path,
    neuron_trace: dict[str, np.ndarray],
    jaxley_trace: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)

    axes[0].plot(neuron_trace["time"], neuron_trace["voltage"], label="NEURON Combe2023")
    axes[0].plot(jaxley_trace["time"], jaxley_trace["voltage"], label="Jaxley Combe port")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].legend(frameon=False)

    axes[1].plot(neuron_trace["time"], neuron_trace["current"], label="NEURON")
    axes[1].plot(jaxley_trace["time"], jaxley_trace["current"], "--", label="Jaxley")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Current (nA)")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Combe2023 NEURON model with the repo Jaxley counterpart."
    )
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tstop", type=float, default=500.0)
    parser.add_argument("--delay", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--amp", type=float, default=0.3, help="Current step amplitude in nA.")
    parser.add_argument("--v-init", type=float, default=-72.0)
    parser.add_argument("--jaxley-d-lambda", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--verbose-neuron",
        action="store_true",
        help="Show the HOC setup output instead of silencing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = StepProtocol(
        dt=args.dt,
        tstop=args.tstop,
        delay=args.delay,
        duration=args.duration,
        amplitude=args.amp,
        v_init=args.v_init,
    )
    protocol.validate()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    neuron_trace = run_neuron_step(
        protocol,
        quiet=not args.verbose_neuron,
        d_lambda=args.jaxley_d_lambda,
    )
    jaxley_trace = run_jaxley_step(protocol, d_lambda=args.jaxley_d_lambda)
    metrics = voltage_metrics(neuron_trace, jaxley_trace)

    neuron_path = output_dir / "neuron_step.npz"
    jaxley_path = output_dir / "jaxley_step.npz"
    plot_path = output_dir / "combe_vs_jaxley_step.png"
    metrics_path = output_dir / "metrics.csv"

    save_trace(neuron_path, neuron_trace)
    save_trace(jaxley_path, jaxley_trace)
    save_metrics(metrics_path, metrics)
    plot_comparison(plot_path, neuron_trace, jaxley_trace)

    print(f"Saved NEURON trace: {neuron_path}")
    print(f"Saved Jaxley trace: {jaxley_path}")
    print(f"Saved comparison plot: {plot_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Voltage error: "
        f"RMSE={metrics['rmse_mV']:.4g} mV, "
        f"MAE={metrics['mae_mV']:.4g} mV, "
        f"max_abs={metrics['max_abs_mV']:.4g} mV"
    )


if __name__ == "__main__":
    main()

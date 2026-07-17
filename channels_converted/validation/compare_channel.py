#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from channels_converted.validation.channel_registry import CHANNELS, channel_keys
from channels_converted.validation.jaxley_runner import run_jaxley_channel
from channels_converted.validation.neuron_runner import DEFAULT_MOD_DIR, compile_mods, run_neuron_channel
from channels_converted.validation.protocols import voltage_step_protocol


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_param_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Parameter override must be NAME=VALUE, got '{item}'.")
        name, raw_value = item.split("=", 1)
        overrides[name] = float(raw_value)
    return overrides


def compare_arrays(neuron_values: np.ndarray, jaxley_values: np.ndarray) -> dict[str, float]:
    diff = jaxley_values - neuron_values
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = float(np.max(np.abs(neuron_values)))
    rel_rmse = rmse / max(denom, 1e-12)
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "neuron_peak_abs": denom,
    }


def save_npz(path: Path, time, voltage, neuron_output, jaxley_output) -> None:
    payload = {
        "time": time,
        "voltage": voltage,
    }
    payload.update({f"neuron_{key}": value for key, value in neuron_output.items()})
    payload.update({f"jaxley_{key}": value for key, value in jaxley_output.items()})
    np.savez(path, **payload)


def save_plot(path: Path, channel_key: str, time, voltage, neuron_output, jaxley_output) -> None:
    import matplotlib.pyplot as plt

    labels = list(neuron_output)
    fig, axes = plt.subplots(len(labels) + 1, 1, figsize=(9, 2.2 * (len(labels) + 1)), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(time, voltage, color="black", linewidth=1.0)
    axes[0].set_ylabel("mV")
    axes[0].set_title(channel_key)

    for axis, label in zip(axes[1:], labels):
        axis.plot(time, neuron_output[label], label="NEURON", linewidth=1.0)
        axis.plot(time, jaxley_output[label], label="Jaxley", linewidth=1.0, linestyle="--")
        axis.set_ylabel(label)
        axis.legend(loc="best", frameon=False)

    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metrics_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt

    labels = [f"{row['channel']}:{row['variable']}" for row in rows]
    values = np.asarray([row["rel_rmse"] for row in rows], dtype=float)
    colors = ["#2e7d32" if row["passed"] else "#c62828" for row in rows]
    y = np.arange(len(labels))

    fig_height = max(4.0, 0.28 * len(labels))
    fig, axis = plt.subplots(figsize=(10, fig_height))
    axis.barh(y, values, color=colors)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("relative RMSE")
    axis.set_xscale("symlog", linthresh=1e-12)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_channel(args, channel_key: str, param_overrides: dict[str, float]) -> list[dict[str, object]]:
    spec = CHANNELS[channel_key]
    protocol = voltage_step_protocol(
        dt=args.dt,
        hold_mv=args.hold_mv,
        hold_ms=args.hold_ms,
        step_ms=args.step_ms,
        tail_ms=args.tail_ms,
        step_mvs=args.step_mv,
    )

    neuron_output = run_neuron_channel(
        spec,
        protocol,
        mod_dir=args.mod_dir,
        param_overrides=param_overrides,
    )
    jaxley_output = run_jaxley_channel(
        spec,
        protocol,
        param_overrides=param_overrides,
    )

    results: list[dict[str, object]] = []
    for label in neuron_output:
        metrics = compare_arrays(neuron_output[label], jaxley_output[label])
        passed = metrics["max_abs"] <= args.atol or metrics["rel_rmse"] <= args.rtol
        results.append(
            {
                "channel": channel_key,
                "variable": label,
                "passed": passed,
                **metrics,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(
        args.output_dir / f"{channel_key}_traces.npz",
        protocol.time,
        protocol.voltage,
        neuron_output,
        jaxley_output,
    )
    if args.plot:
        save_plot(
            args.output_dir / f"{channel_key}_overlay.png",
            channel_key,
            protocol.time,
            protocol.voltage,
            neuron_output,
            jaxley_output,
        )

    return results


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = ["channel", "variable", "passed", "max_abs", "rmse", "rel_rmse", "neuron_peak_abs"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare converted Jaxley channel kinetics against compiled NEURON MOD mechanisms.",
    )
    parser.add_argument("--channel", choices=sorted(CHANNELS), help="Single channel key to validate.")
    parser.add_argument("--all", action="store_true", help="Validate every non-skipped channel.")
    parser.add_argument("--include-skipped", action="store_true", help="Include cal4/d3 in --all runs.")
    parser.add_argument("--list", action="store_true", help="List known channels and exit.")
    parser.add_argument("--compile", action="store_true", help="Run nrnivmodl in the MOD directory first.")
    parser.add_argument("--plot", action="store_true", help="Save overlay plots for each channel.")
    parser.add_argument("--plot-summary", action="store_true", help="Save one summary plot of all metric errors.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any variable exceeds tolerance.")
    parser.add_argument("--mod-dir", type=Path, default=DEFAULT_MOD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--hold-mv", type=float, default=-80.0)
    parser.add_argument("--hold-ms", type=float, default=100.0)
    parser.add_argument("--step-ms", type=float, default=100.0)
    parser.add_argument("--tail-ms", type=float, default=50.0)
    parser.add_argument(
        "--step-mv",
        type=float,
        nargs="+",
        default=[-90.0, -70.0, -50.0, -30.0, -10.0, 10.0, 30.0],
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override a parameter as NAME=VALUE. Accepts either Jaxley full names or unprefixed MOD names.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for key, spec in CHANNELS.items():
            marker = "skip-by-default" if spec.skip_by_default else "default"
            print(f"{key:10s} {spec.mechanism:10s} {marker} {spec.note}")
        return 0

    if args.compile:
        compile_mods(args.mod_dir)

    if args.all:
        selected_channel_keys = channel_keys(include_skipped=args.include_skipped)
    elif args.channel:
        selected_channel_keys = [args.channel]
    else:
        raise SystemExit("Pass --channel CHANNEL, --all, or --list.")

    param_overrides = parse_param_overrides(args.param)
    all_rows: list[dict[str, object]] = []
    for key in selected_channel_keys:
        spec = CHANNELS[key]
        if spec.note:
            print(f"{key}: {spec.note}")
        rows = validate_channel(args, key, param_overrides)
        all_rows.extend(rows)
        for row in rows:
            status = "PASS" if row["passed"] else "FAIL"
            print(
                f"{status} {row['channel']:10s} {row['variable']:10s} "
                f"max_abs={row['max_abs']:.6g} rmse={row['rmse']:.6g} rel_rmse={row['rel_rmse']:.6g}"
            )

    write_metrics(args.output_dir / "metrics.csv", all_rows)
    if args.plot_summary or (args.plot and args.all):
        save_metrics_plot(args.output_dir / "all_channels_metric_summary.png", all_rows)
    failed = [row for row in all_rows if not row["passed"]]
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

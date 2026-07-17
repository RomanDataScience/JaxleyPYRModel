from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import argparse
import csv
import os
from pathlib import Path
import warnings

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jaxley_mpl")

import jax
import jax.numpy as jnp
import jaxley as jx
import jaxley.optimize.transforms as transform_module
from jaxley.optimize.transforms import SigmoidTransform
from jaxley.optimize.utils import l2_norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_Combe import (
    Combe2023,
    SEGMENTED_TRACES_DIR,
    bounds,
    params,
    set_fitted_parameters,
)


warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "Fit_Results_Combe"
SEGMENT_ALIASES = {
    "depolarizing_pulse": "depolarizing_step",
    "hyperpolarizing_step": "hyperpolarizing_pulse",
}


def save_exp_compatible(x, max_value: float = 20.0):
    return jnp.exp(jnp.clip(x, max=max_value))


transform_module.save_exp = save_exp_compatible


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the Combe2023 Jaxley model to one segmented trace.")
    parser.add_argument("--cell-name", default="m20240527cd")
    parser.add_argument("--trace-name", default="v75ctrl")
    parser.add_argument("--segment-name", default="depolarizing_step")
    parser.add_argument("--segmented-dir", default=SEGMENTED_TRACES_DIR, type=Path)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--experimental-dt", default=0.05, type=float)
    parser.add_argument("--delta-t", default=0.05, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr-scale", default=0.03, type=float)
    parser.add_argument("--beta", default=0.9, type=float)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--d-lambda", default=0.3, type=float)
    parser.add_argument("--plot-every", default=1, type=int)
    parser.add_argument(
        "--loss-pre-ms",
        default=100.0,
        type=float,
        help="Milliseconds before detected current onset to include in the loss.",
    )
    parser.add_argument(
        "--loss-post-ms",
        default=800.0,
        type=float,
        help="Milliseconds after detected current offset to include in the loss.",
    )
    parser.add_argument(
        "--stim-threshold-nA",
        default=None,
        type=float,
        help="Current deviation threshold for stimulus detection. Default: infer from trace.",
    )
    parser.add_argument("--disable-calcium-diffusion", action="store_true")
    parser.add_argument(
        "--fit-group",
        action="append",
        choices=sorted(params),
        help="Parameter group to fit. Default: conductances and passive.",
    )
    parser.add_argument(
        "--fit-key",
        action="append",
        default=[],
        help="Fit an explicit parameter key instead of the group defaults. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-key",
        action="append",
        default=[],
        help="Remove one parameter from the selected fit set. Can be repeated.",
    )
    parser.add_argument("--list-parameters", action="store_true")
    return parser.parse_args()


def segment_name(name: str) -> str:
    return SEGMENT_ALIASES.get(name, name)


def segment_file(args, suffix: str) -> Path:
    name = segment_name(args.segment_name)
    path = (
        Path(args.segmented_dir)
        / args.cell_name
        / name
        / f"{args.trace_name}_{name}_{suffix}.txt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_segment(path: Path, *, source_dt: float, target_dt: float, scale: float = 1.0):
    data = np.atleast_1d(np.loadtxt(path, dtype=float)) * scale
    data = jnp.asarray(data, dtype=jnp.float64)

    target_steps = max(1, int(round(data.size * source_dt / target_dt)))
    if target_steps == data.size:
        return data

    indices = jnp.floor(jnp.arange(target_steps) * target_dt / source_dt).astype(jnp.int32)
    return data[jnp.minimum(indices, data.size - 1)]


def loss_window_from_stimulus(
    stimulus,
    *,
    delta_t: float,
    pre_ms: float,
    post_ms: float,
    threshold_nA: float | None = None,
):
    current = np.asarray(stimulus, dtype=float)
    n_samples = current.size
    if n_samples == 0:
        raise ValueError("Cannot build a loss window from an empty stimulus.")

    baseline_count = max(1, min(n_samples, int(round(pre_ms / delta_t))))
    baseline = float(np.median(current[:baseline_count]))
    deviation = np.abs(current - baseline)
    max_deviation = float(np.max(deviation))

    if threshold_nA is None:
        noise = float(np.median(np.abs(current[:baseline_count] - baseline)))
        threshold_nA = max(1e-9, 10.0 * noise, 0.01 * max_deviation)

    active = deviation > threshold_nA
    if not np.any(active):
        warnings.warn(
            "Could not detect a current step in the stimulus; using the full trace for the loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.arange(n_samples, dtype=np.int32), {
            "baseline_nA": baseline,
            "threshold_nA": float(threshold_nA),
            "onset_index": 0,
            "offset_index": n_samples - 1,
            "start_index": 0,
            "stop_index": n_samples,
        }

    active_indices = np.flatnonzero(active)
    onset = int(active_indices[0])
    offset = int(active_indices[-1])
    pre_steps = int(round(pre_ms / delta_t))
    post_steps = int(round(post_ms / delta_t))
    start = max(0, onset - pre_steps)
    stop = min(n_samples, offset + 1 + post_steps)

    return np.arange(start, stop, dtype=np.int32), {
        "baseline_nA": baseline,
        "threshold_nA": float(threshold_nA),
        "onset_index": onset,
        "offset_index": offset,
        "start_index": start,
        "stop_index": stop,
    }


def all_values():
    values = {}
    for group in params.values():
        values.update(group)
    return values


def parameter_keys(args):
    if args.fit_key:
        keys = list(dict.fromkeys(args.fit_key))
    else:
        groups = args.fit_group or ["conductances", "passive"]
        keys = []
        for group in groups:
            keys.extend(params[group])
        keys = list(dict.fromkeys(keys))

    excluded = set(args.exclude_key)
    keys = [key for key in keys if key not in excluded]
    known = all_values()
    missing = [key for key in keys if key not in known]
    missing_bounds = [key for key in keys if key not in bounds]
    if missing:
        raise KeyError(f"Unknown fit parameter(s): {', '.join(missing)}")
    if missing_bounds:
        raise KeyError(f"Missing bounds for fit parameter(s): {', '.join(missing_bounds)}")
    if not keys:
        raise ValueError("No fit parameters selected.")
    return keys


def initial_parameters(keys):
    values = all_values()
    initial = jnp.asarray([values[key] for key in keys], dtype=jnp.float64)
    lower, upper = parameter_bounds(keys)
    margin = 1e-6 * (upper - lower)
    return jnp.minimum(jnp.maximum(initial, lower + margin), upper - margin)


def parameter_bounds(keys):
    return (
        jnp.asarray([bounds[key][0] for key in keys], dtype=jnp.float64),
        jnp.asarray([bounds[key][1] for key in keys], dtype=jnp.float64),
    )


def experimental_v_final(observed, *, offset_index: int, delta_t: float, window_ms: float = 5.0):
    values = np.asarray(observed, dtype=float)
    if values.size == 0:
        return None

    stop = min(values.size, max(0, offset_index) + 1)
    window_steps = max(1, int(round(window_ms / delta_t)))
    start = max(0, stop - window_steps)
    return float(np.mean(values[start:stop]))


def save_fit_plot(path, time, observed, simulated, current, title, loss_window=None, v_final=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(time, observed, color="black", lw=0.9, label="experimental")
    axes[0].plot(time, simulated, color="#2b8cbe", lw=0.9, label="simulated")
    if v_final is not None:
        axes[0].axhline(
            v_final,
            color="black",
            lw=0.9,
            ls=(0, (4, 3)),
            alpha=0.8,
            label="experimental v_final",
        )
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(title)
    axes[0].legend(frameon=False)

    axes[1].plot(time[: current.size], current, color="#636363", lw=0.9)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Current (nA)")
    if loss_window is not None:
        for ax in axes:
            ax.axvspan(loss_window[0], loss_window[1], color="#f0b44c", alpha=0.16, lw=0)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_parameter_list():
    for group_name, group_params in params.items():
        print(f"[{group_name}]")
        for key, value in group_params.items():
            low, high = bounds[key]
            print(f"{key}: initial={value} bounds=[{low}, {high}]")


def main():
    args = parse_args()
    if args.list_parameters:
        write_parameter_list()
        return
    if args.d_lambda <= 0.0:
        raise ValueError("--d-lambda must be positive")
    if args.delta_t <= 0.0 or args.experimental_dt <= 0.0:
        raise ValueError("--delta-t and --experimental-dt must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.plot_every < 0:
        raise ValueError("--plot-every must be non-negative")
    if args.loss_pre_ms < 0.0 or args.loss_post_ms < 0.0:
        raise ValueError("--loss-pre-ms and --loss-post-ms must be non-negative")
    if args.stim_threshold_nA is not None and args.stim_threshold_nA <= 0.0:
        raise ValueError("--stim-threshold-nA must be positive when provided")

    args.segment_name = segment_name(args.segment_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stimulus = load_segment(
        segment_file(args, "i"),
        source_dt=args.experimental_dt,
        target_dt=args.delta_t,
        scale=1e-3,
    )
    observed = load_segment(
        segment_file(args, "v"),
        source_dt=args.experimental_dt,
        target_dt=args.delta_t,
    )
    if args.max_steps is not None:
        stimulus = stimulus[: args.max_steps]
        observed = observed[: args.max_steps]

    n_samples = min(stimulus.size, observed.size)
    stimulus = stimulus[:n_samples]
    observed = observed[:n_samples]
    time = np.arange(n_samples) * args.delta_t
    label = f"{args.cell_name}_{args.trace_name}_{args.segment_name}_combe"
    loss_indices_np, loss_window = loss_window_from_stimulus(
        stimulus,
        delta_t=args.delta_t,
        pre_ms=args.loss_pre_ms,
        post_ms=args.loss_post_ms,
        threshold_nA=args.stim_threshold_nA,
    )
    loss_indices = jnp.asarray(loss_indices_np, dtype=jnp.int32)
    loss_window_ms = (
        loss_window["start_index"] * args.delta_t,
        (loss_window["stop_index"] - 1) * args.delta_t,
    )
    print(
        "loss window: "
        f"{loss_window_ms[0]:.3f}-{loss_window_ms[1]:.3f} ms "
        f"({loss_indices_np.size}/{n_samples} samples), "
        f"stimulus onset={loss_window['onset_index'] * args.delta_t:.3f} ms, "
        f"offset={loss_window['offset_index'] * args.delta_t:.3f} ms"
    )
    v_final = None
    if args.segment_name == "depolarizing_step":
        v_final = experimental_v_final(
            observed,
            offset_index=loss_window["offset_index"],
            delta_t=args.delta_t,
        )
        print(f"experimental v_final={v_final:.3f} mV")

    cell = Combe2023(
        d_lambda=args.d_lambda,
        enable_calcium_diffusion=not args.disable_calcium_diffusion,
    )
    cell.delete_stimuli()
    cell.delete_recordings()
    cell.soma.branch(0).loc(0.5).stimulate(stimulus)
    cell.soma.branch(0).loc(0.5).record()
    cell.set("v", float(observed[0]))
    cell.init_states()

    keys = parameter_keys(args)
    lower, upper = parameter_bounds(keys)
    start_params = initial_parameters(keys)
    transform = SigmoidTransform(lower=lower, upper=upper)
    opt_params = transform.inverse(start_params)
    checkpoint = int(np.ceil(float(stimulus.size) ** 0.5))
    checkpoints = [checkpoint, checkpoint]

    def simulate(fit_params):
        pstate = set_fitted_parameters(cell, keys, fit_params)
        voltage = jx.integrate(
            cell,
            param_state=pstate,
            delta_t=args.delta_t,
            checkpoint_lengths=checkpoints,
        )[0]
        return voltage[: observed.size]

    def loss(opt_values):
        fit_params = transform.forward(opt_values)
        predicted = simulate(fit_params)
        residual = predicted[loss_indices] - observed[loss_indices]
        return jnp.mean(residual**2), predicted

    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    simulate = jax.jit(simulate)

    history = []
    best_mse = np.inf
    best_params = None
    best_voltage = None
    best_plot_dir = args.output_dir / f"{label}_best_by_epoch"
    current_plot_dir = args.output_dir / f"{label}_current_by_epoch"
    write_epoch_plots = args.plot_every > 0
    if write_epoch_plots:
        best_plot_dir.mkdir(parents=True, exist_ok=True)
        current_plot_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        (value, predicted), grad = value_and_grad(opt_params)
        grad_norm = l2_norm(grad)
        predicted_np = np.asarray(predicted)
        value_float = float(value)
        rmse_float = float(jnp.sqrt(value))
        if value_float < best_mse:
            best_mse = value_float
            best_params = np.asarray(transform.forward(opt_params))
            best_voltage = predicted_np

        should_plot_epoch = write_epoch_plots and (
            epoch % args.plot_every == 0 or epoch == args.epochs - 1
        )
        if should_plot_epoch:
            save_fit_plot(
                current_plot_dir / f"{label}_current_epoch_{epoch:03d}.png",
                time,
                np.asarray(observed),
                predicted_np,
                np.asarray(stimulus),
                f"Current epoch {epoch}, RMSE={rmse_float:.3f} mV",
                loss_window_ms,
                v_final,
            )
            save_fit_plot(
                best_plot_dir / f"{label}_best_epoch_{epoch:03d}.png",
                time,
                np.asarray(observed),
                best_voltage,
                np.asarray(stimulus),
                f"Best fit through epoch {epoch}, RMSE={np.sqrt(best_mse):.3f} mV",
                loss_window_ms,
                v_final,
            )

        opt_params = opt_params - args.lr_scale * grad / (grad_norm**args.beta + 1e-12)
        history.append(
            {
                "epoch": epoch,
                "mse": value_float,
                "rmse_mV": rmse_float,
                "grad_norm": float(grad_norm),
            }
        )
        print(
            f"epoch {epoch:03d} mse={value_float:.6g} "
            f"rmse_mV={rmse_float:.6g} grad_norm={float(grad_norm):.6g}"
        )

    fitted_params = jnp.asarray(best_params) if best_params is not None else transform.forward(opt_params)
    fitted_voltage = best_voltage if best_voltage is not None else np.asarray(simulate(fitted_params))

    history_path = args.output_dir / f"{label}_history.csv"
    params_path = args.output_dir / f"{label}_params.csv"
    figure_path = args.output_dir / f"{label}_fit.png"
    loss_window_path = args.output_dir / f"{label}_loss_window.csv"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mse", "rmse_mV", "grad_norm"])
        writer.writeheader()
        writer.writerows(history)

    start_by_key = dict(zip(keys, np.asarray(start_params), strict=True))
    with params_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "initial", "lower", "upper", "fitted"])
        writer.writeheader()
        for key, fitted in zip(keys, np.asarray(fitted_params), strict=True):
            low, high = bounds[key]
            writer.writerow(
                {
                    "parameter": key,
                    "initial": float(start_by_key[key]),
                    "lower": float(low),
                    "upper": float(high),
                    "fitted": float(fitted),
                }
            )

    with loss_window_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "start_index",
            "stop_index",
            "onset_index",
            "offset_index",
            "start_ms",
            "stop_ms",
            "onset_ms",
            "offset_ms",
            "baseline_nA",
        "threshold_nA",
        "loss_samples",
        "total_samples",
        "experimental_v_final_mV",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "start_index": loss_window["start_index"],
                "stop_index": loss_window["stop_index"],
                "onset_index": loss_window["onset_index"],
                "offset_index": loss_window["offset_index"],
                "start_ms": loss_window["start_index"] * args.delta_t,
                "stop_ms": (loss_window["stop_index"] - 1) * args.delta_t,
                "onset_ms": loss_window["onset_index"] * args.delta_t,
                "offset_ms": loss_window["offset_index"] * args.delta_t,
                "baseline_nA": loss_window["baseline_nA"],
                "threshold_nA": loss_window["threshold_nA"],
                "loss_samples": int(loss_indices_np.size),
                "total_samples": int(n_samples),
                "experimental_v_final_mV": v_final,
            }
        )

    save_fit_plot(
        figure_path,
        time,
        np.asarray(observed),
        np.asarray(fitted_voltage),
        np.asarray(stimulus),
        f"Best Combe fit, RMSE={np.sqrt(best_mse):.3f} mV",
        loss_window_ms,
        v_final,
    )

    print(history_path)
    print(params_path)
    print(loss_window_path)
    print(figure_path)
    if write_epoch_plots:
        print(current_plot_dir)
        print(best_plot_dir)


if __name__ == "__main__":
    main()

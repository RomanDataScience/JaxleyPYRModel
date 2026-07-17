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

from model import L5PC, SEGMENTED_TRACES_DIR, bounds, params, set_fitted_parameters

warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)


MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "Fit_Results"
SEGMENT_ALIASES = {
    "depolarizing_pulse": "depolarizing_step",
    "hyperpolarizing_step": "hyperpolarizing_pulse",
}


def save_exp_compatible(x, max_value: float = 20.0):
    return jnp.exp(jnp.clip(x, max=max_value))


transform_module.save_exp = save_exp_compatible


def parse_args():
    parser = argparse.ArgumentParser(description="Fit L5PC to one segmented current-clamp trace.")
    parser.add_argument("--cell-name", default="m20240527cd")
    parser.add_argument("--trace-name", default="v75ctrl")
    parser.add_argument("--segment-name", default="depolarizing_step")
    parser.add_argument("--segmented-dir", default=SEGMENTED_TRACES_DIR, type=Path)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--experimental-dt", default=0.05, type=float)
    parser.add_argument("--delta-t", default=0.05, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr-scale", default=0.1, type=float)
    parser.add_argument("--beta", default=0.9, type=float)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--d-lambda", default=0.1, type=float)
    parser.add_argument("--plot-every", default=1, type=int)
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


def parameter_keys():
    return list(bounds.keys())


def initial_parameters(keys):
    values = {}
    for group_params in params.values():
        values.update(group_params)
    return jnp.asarray([values[key] for key in keys], dtype=jnp.float64)


def parameter_bounds(keys):
    return (
        jnp.asarray([bounds[key][0] for key in keys], dtype=jnp.float64),
        jnp.asarray([bounds[key][1] for key in keys], dtype=jnp.float64),
    )


def save_fit_plot(path, time, observed, simulated, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), constrained_layout=True)
    ax.plot(time, observed, color="black", lw=0.9, label="experimental")
    ax.plot(time, simulated, color="#41ae76", lw=0.9, label="simulated")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    if args.d_lambda <= 0.0:
        raise ValueError("--d-lambda must be positive")
    if args.plot_every < 0:
        raise ValueError("--plot-every must be non-negative")
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
    time = np.arange(observed.size) * args.delta_t
    label = f"{args.cell_name}_{args.trace_name}_{args.segment_name}"

    cell = L5PC(d_lambda=args.d_lambda)
    cell.delete_stimuli()
    cell.delete_recordings()
    cell.soma.branch(0).loc(0.5).stimulate(stimulus)
    cell.soma.branch(0).loc(0.5).record()
    cell.set("v", float(observed[0]))
    cell.init_states()

    keys = parameter_keys()
    lower, upper = parameter_bounds(keys)
    transform = SigmoidTransform(lower=lower, upper=upper)
    opt_params = transform.inverse(initial_parameters(keys))
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
        return jnp.mean((predicted - observed) ** 2), predicted

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
    step_scale = args.lr_scale * l2_norm(jnp.ones(len(opt_params)))
    for epoch in range(args.epochs):
        (value, predicted), grad = value_and_grad(opt_params)
        grad_norm = l2_norm(grad)
        predicted = np.asarray(predicted)
        if float(value) < best_mse:
            best_mse = float(value)
            best_params = np.asarray(transform.forward(opt_params))
            best_voltage = predicted
        should_plot_epoch = write_epoch_plots and (
            epoch % args.plot_every == 0 or epoch == args.epochs - 1
        )
        if should_plot_epoch:
            save_fit_plot(
                current_plot_dir / f"{label}_current_epoch_{epoch:03d}.png",
                time,
                np.asarray(observed),
                predicted,
                f"Current epoch {epoch}, RMSE={float(jnp.sqrt(value)):.3f} mV",
            )
            save_fit_plot(
                best_plot_dir / f"{label}_best_epoch_{epoch:03d}.png",
                time,
                np.asarray(observed),
                best_voltage,
                f"Best fit through epoch {epoch}, RMSE={np.sqrt(best_mse):.3f} mV",
            )
        opt_params = opt_params - value * step_scale * grad / (grad_norm**args.beta + 1e-12)
        history.append({"epoch": epoch, "mse": float(value), "rmse_mV": float(jnp.sqrt(value))})
        print(f"epoch {epoch:03d} mse={float(value):.6g} rmse_mV={float(jnp.sqrt(value)):.6g}")

    fitted_params = jnp.asarray(best_params) if best_params is not None else transform.forward(opt_params)
    fitted_voltage = best_voltage if best_voltage is not None else np.asarray(simulate(fitted_params))

    history_path = args.output_dir / f"{label}_history.csv"
    params_path = args.output_dir / f"{label}_params.csv"
    figure_path = args.output_dir / f"{label}_fit.png"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mse", "rmse_mV"])
        writer.writeheader()
        writer.writerows(history)

    with params_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "initial", "fitted"])
        writer.writeheader()
        for key, start, fitted in zip(keys, initial_parameters(keys), fitted_params, strict=True):
            writer.writerow({"parameter": key, "initial": float(start), "fitted": float(fitted)})

    save_fit_plot(
        figure_path,
        time,
        np.asarray(observed),
        np.asarray(fitted_voltage),
        f"Best fit, RMSE={np.sqrt(best_mse):.3f} mV" if best_voltage is not None else "Fit",
    )

    print(history_path)
    print(params_path)
    print(figure_path)
    if write_epoch_plots:
        print(current_plot_dir)
        print(best_plot_dir)


if __name__ == "__main__":
    main()

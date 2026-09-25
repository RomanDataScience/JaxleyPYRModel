"""Plot separate soma membrane-current components for one archived candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuron_optim.config import load_config
from neuron_optim.data import load_protocol_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--potassium-scale", type=float, default=1.0)
    parser.add_argument("--rm-soma-scale", type=float, default=1.0)
    parser.add_argument("--ra-soma-scale", type=float, default=1.0)
    parser.add_argument("--gna-scale", type=float, default=1.0)
    parser.add_argument("--gnadend-scale", type=float, default=1.0)
    parser.add_argument("--region", choices=("soma", "axon"), default="soma")
    args = parser.parse_args()

    import jax.numpy as jnp
    import jaxley as jx
    from JaxleyModel.model.model_Combe import (
        Combe2023,
        SUPPORTED_FIT_PARAMETER_KEYS,
        set_fitted_parameters,
    )

    config = load_config(args.config)
    with args.candidate.open() as handle:
        values = json.load(handle)["physical_by_name"]
    potassium_keys = (
        "gkdrsoma", "gkv2soma", "soma_kap", "soma_km", "soma_kca", "mykca_init",
        "axongkdr", "gkv2axon", "axon_kap",
    )
    for key in potassium_keys:
        values[key] *= args.potassium_scale
    values["RmSoma"] *= args.rm_soma_scale
    values["RaSoma"] *= args.ra_soma_scale
    values["gna"] *= args.gna_scale
    values["gnadend"] *= args.gnadend_scale

    traces = load_protocol_traces(
        config.data_root,
        cell=config.cell,
        protocol="depolarizing_step",
        trace_names=("v75ctrl",),
        pre_ms=0.0,
        post_ms=0.0,
        full_trial=False,
        center_current=False,
    )
    trace = traces[0]

    cell = Combe2023(morphology_source="hoc")
    keys = tuple(key for key in values if key in SUPPORTED_FIT_PARAMETER_KEYS)
    param_state = set_fitted_parameters(
        cell, keys, jnp.asarray([values[key] for key in keys], dtype=jnp.float64)
    )

    # Jaxley normally aggregates mechanisms sharing i_K/i_Ca. Give every
    # soma mechanism a private current name for this diagnostic run. This does
    # not alter the channel equations or the voltage integration.
    current_names = []
    all_current_names = []
    channel_labels = []
    original_current_names = list(cell.membrane_current_names)
    region_view = getattr(cell, args.region)
    region_channel_types = {id(channel) for channel in region_view.channels}
    for index, channel in enumerate(cell.channels):
        label = type(channel).__name__.lower()
        channel.current_name = f"i_chan{index}_{label}"
        all_current_names.append(channel.current_name)
        if id(channel) in region_channel_types:
            current_names.append(channel.current_name)
            channel_labels.append(label)
    cell.membrane_current_names = all_current_names + [
        name for name in original_current_names if name not in all_current_names
    ]

    dt = float(np.median(np.diff(trace.time_ms)))
    current = 0.5 * (trace.current_nA[:-1] + trace.current_nA[1:])
    current = np.concatenate((current, trace.current_nA[-1:]))
    cell.delete_stimuli()
    cell.delete_recordings()
    compartment = region_view.branch(0).loc(0.5)
    compartment.stimulate(jnp.asarray(current, dtype=jnp.float64))
    compartment.record("v", verbose=False)
    for name in current_names:
        compartment.record(name, verbose=False)
    cell.set("v", float(trace.voltage_mV[0]))

    # Match the production simulator's functional state initialization.
    from neuron_optim.jaxley_simulator import _init_states_with_param_state

    _init_states_with_param_state(cell, param_state, delta_t=dt)
    result = jx.integrate(
        cell,
        param_state=param_state,
        delta_t=dt,
        solver="bwd_euler",
        voltage_solver="jaxley.dhs.cpu",
    )
    result = np.asarray(result, dtype=float)
    names = ["v", *current_names]
    if result.ndim != 2 or result.shape[0] != len(names):
        raise RuntimeError(f"Unexpected recording shape: {result.shape}")
    result = result.T
    n = min(result.shape[0], trace.time_ms.size)
    sim_time = trace.time_ms[:n] - trace.time_ms[0]
    sim = result[:n]
    onset = float(trace.epoch_start_ms - trace.time_ms[0])
    first_window = (sim_time >= onset) & (sim_time <= onset + 20.0)
    print(
        f"first simulated peak={sim[first_window, 0].max():.4f} mV; "
        f"first experimental peak={trace.voltage_mV[:n][first_window].max():.4f} mV"
    )

    fig, axes = plt.subplots(
        len(current_names) + 1, 1, figsize=(12, 2.1 * (len(current_names) + 1)),
        sharex=True, constrained_layout=True,
    )
    axes[0].plot(sim_time, trace.voltage_mV[:n], color="black", lw=1.2, label="experiment")
    axes[0].plot(sim_time, sim[:, 0], color="tab:red", lw=1.0, label="simulation")
    axes[0].set_ylabel("V (mV)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"b000 best candidate — {args.region} currents, v75ctrl")
    for index, name in enumerate(current_names, start=1):
        axes[index].plot(sim_time, sim[:, index], lw=0.9)
        axes[index].axhline(0.0, color="0.7", lw=0.5)
        axes[index].set_ylabel(f"{channel_labels[index - 1]}\n(mA/cm²)")
    axes[-1].set_xlabel("Time from trace start (ms)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")
    for index, name in enumerate(names[1:], start=1):
        values_at_spike = sim[:, index]
        print(f"{name} min={values_at_spike.min():.6g} max={values_at_spike.max():.6g}")


if __name__ == "__main__":
    main()

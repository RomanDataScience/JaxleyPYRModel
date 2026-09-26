"""Plot separate soma membrane-current components for one archived candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ``JaxleyModel`` lives beside NEURON_optim, at the repository root.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from neuron_optim.config import load_config
from neuron_optim.data import load_protocol_traces
from neuron_optim.parameters import upgrade_legacy_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--potassium-scale", type=float, default=1.0)
    parser.add_argument("--rm-soma-scale", type=float, default=1.0)
    parser.add_argument("--ra-soma-scale", type=float, default=1.0)
    parser.add_argument("--ra-soma-value", type=float)
    parser.add_argument("--ra-proximal-axon-scale", type=float, default=1.0,
                        help="Scale Ra only in the first axon compartment")
    parser.add_argument("--rm-soma-value", type=float)
    parser.add_argument("--cm-soma-value", type=float)
    parser.add_argument("--gna-scale", type=float, default=1.0)
    parser.add_argument("--gnadend-scale", type=float, default=1.0)
    parser.add_argument("--gkdrsoma-scale", type=float, default=1.0)
    parser.add_argument("--soma-kap-scale", type=float, default=1.0)
    parser.add_argument("--gkv2soma-scale", type=float, default=1.0)
    parser.add_argument("--region", choices=("soma", "axon"), default="soma")
    parser.add_argument("--protocol", choices=("depolarizing_step", "hyperpolarizing_pulse"),
                        default="depolarizing_step")
    parser.add_argument("--hillock", action="store_true")
    parser.add_argument("--ais-proximal-scale", type=float, default=1.0)
    parser.add_argument("--ais-proximal-radius-scale", type=float, default=1.0,
                        help="Scale radius/area of proximal axon compartments")
    args = parser.parse_args()

    import jax.numpy as jnp
    import jaxley as jx
    from JaxleyModel.model.model_Combe import (
        NEURON_UPDATE_MODE,
        Combe2023,
        SUPPORTED_FIT_PARAMETER_KEYS,
        set_fitted_parameters,
    )

    config = load_config(args.config)
    with args.candidate.open() as handle:
        candidate_data = json.load(handle)
    if isinstance(candidate_data, list):
        candidate_data = candidate_data[0]
    if "physical_by_name" in candidate_data:
        values = dict(candidate_data["physical_by_name"])
    else:
        # best_so_far.json sits beside manifest.json; archived candidates
        # sit a couple of directories below it.
        manifest = next(parent / "manifest.json" for parent in args.candidate.resolve().parents
                        if (parent / "manifest.json").exists())
        parameter_keys = json.load(manifest.open())["parameter_keys"]
        values = dict(zip(parameter_keys, candidate_data["physical"]))
    values = upgrade_legacy_values(values)
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
    values["gkdrsoma"] *= args.gkdrsoma_scale
    values["soma_kap"] *= args.soma_kap_scale
    values["gkv2soma"] *= args.gkv2soma_scale
    if args.ra_soma_value is not None:
        values["RaSoma"] = args.ra_soma_value
    if args.rm_soma_value is not None:
        values["RmSoma"] = args.rm_soma_value
    if args.cm_soma_value is not None:
        values["CmSomaOnly"] = args.cm_soma_value

    traces = load_protocol_traces(
        config.data_root,
        cell=config.cell,
        protocol=args.protocol,
        # Same traces and window as the optimizer's stages (_load_traces).
        trace_names=(config.hyperpolarizing_trace_names
                     if args.protocol == "hyperpolarizing_pulse" else ("v75ctrl",)),
        pre_ms=config.simulation_pre_ms,
        post_ms=config.simulation_post_ms,
        full_trial=True,
        **({"center_current": False} if args.protocol == "hyperpolarizing_pulse" else {}),
    )
    trace = traces[0]

    cell = Combe2023(d_lambda=config.d_lambda, morphology_source="hoc")
    # Match JaxleySimulator: NEURON_optim final-compartment parameter rules
    # rather than the frozen-HOC anchored profiles.
    cell._combe_parameter_update_mode = NEURON_UPDATE_MODE
    keys = tuple(key for key in values if key in SUPPORTED_FIT_PARAMETER_KEYS)
    param_state = set_fitted_parameters(
        cell, keys, jnp.asarray([values[key] for key in keys], dtype=jnp.float64)
    )

    axon_indices = np.asarray(cell.axon._nodes_in_view, dtype=int)
    if args.hillock:
        # Diagnostic tapered-hillock approximation: reshape only the first
        # proximal axon compartments; the HOC topology remains unchanged.
        radii = np.asarray(cell.nodes.loc[axon_indices, "radius"], dtype=float)
        soma_radius = float(np.median(cell.nodes.loc[cell.soma._nodes_in_view, "radius"]))
        taper_count = min(4, radii.size)
        taper = np.linspace(soma_radius * 0.5, radii[taper_count - 1], taper_count)
        cell.nodes.loc[axon_indices[:taper_count], "radius"] = taper
        lengths = np.asarray(cell.nodes.loc[axon_indices[:taper_count], "length"], dtype=float)
        cell.nodes.loc[axon_indices[:taper_count], "area"] = 2.0 * np.pi * taper * lengths
    if args.ais_proximal_radius_scale != 1.0:
        proximal_count = max(1, axon_indices.size // 3)
        proximal = axon_indices[:proximal_count]
        radii = np.asarray(cell.nodes.loc[proximal, "radius"], dtype=float)
        lengths = np.asarray(cell.nodes.loc[proximal, "length"], dtype=float)
        radii *= args.ais_proximal_radius_scale
        cell.nodes.loc[proximal, "radius"] = radii
        cell.nodes.loc[proximal, "area"] = 2.0 * np.pi * radii * lengths
    if args.ais_proximal_scale != 1.0:
        proximal_count = max(1, axon_indices.size // 3)
        proximal = axon_indices[:proximal_count]
        for update in param_state:
            if update["key"] == "nax_gbar":
                indices = np.asarray(update["indices"], dtype=int)
                values_array = np.asarray(update["val"], dtype=float).copy()
                for index in proximal:
                    matches = np.flatnonzero(indices == index)
                    if matches.size:
                        values_array[matches[0]] *= args.ais_proximal_scale
                update["val"] = jnp.asarray(values_array, dtype=jnp.float64)

    if args.ra_proximal_axon_scale != 1.0:
        # This is intentionally a local diagnostic perturbation, not a fitted
        # parameter.  The first axon compartment is the segment attached to
        # soma(1.0) in the HOC morphology.
        proximal_index = int(axon_indices[0])
        current_ra = float(cell.nodes.loc[proximal_index, "axial_resistivity"])
        cell.nodes.loc[proximal_index, "axial_resistivity"] = (
            current_ra * args.ra_proximal_axon_scale
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
    # Record both sides of the anatomical junction explicitly.  These are
    # separate from the midpoint voltage used for the current plot.
    cell.soma.loc(1.0).record("v", verbose=False)
    cell.axon.loc(0.0).record("v", verbose=False)
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
    expected_rows = len(names) + 2
    if result.ndim != 2 or result.shape[0] != expected_rows:
        raise RuntimeError(f"Unexpected recording shape: {result.shape}")
    result = result.T
    n = min(result.shape[0], trace.time_ms.size)
    sim_time = trace.time_ms[:n] - trace.time_ms[0]
    sim = result[:n]
    soma_boundary = sim[:, 1]
    axon_boundary = sim[:, 2]
    # Convert the first axon compartment's axial resistance to ohms.  HOC/Jaxley
    # morphology lengths and radii are in micrometres; Ra is in ohm*cm.
    proximal = int(axon_indices[0])
    length_cm = float(cell.nodes.loc[proximal, "length"]) * 1.0e-4
    radius_cm = float(cell.nodes.loc[proximal, "radius"]) * 1.0e-4
    ra_ohm = float(cell.nodes.loc[proximal, "axial_resistivity"]) * length_cm / (
        np.pi * radius_cm ** 2
    )
    axial_current_nA = (axon_boundary - soma_boundary) * 1000.0 / ra_ohm
    boundary_drop = axon_boundary - soma_boundary
    onset = float(trace.epoch_start_ms - trace.time_ms[0])
    upstroke = (sim_time >= onset) & (sim_time <= onset + 5.0)
    print(
        f"AIS boundary: max |axon(0)-soma(1)|={np.max(np.abs(boundary_drop[upstroke])):.4f} mV; "
        f"max estimated axial current={np.max(np.abs(axial_current_nA[upstroke])):.4f} nA; "
        f"first-axon-compartment R_axial={ra_ohm:.4g} ohm"
    )
    first_window = (sim_time >= onset) & (sim_time <= onset + 20.0)
    peak_index = np.flatnonzero(first_window)[np.argmax(sim[first_window, 0])]
    print(
        f"first simulated peak={sim[first_window, 0].max():.4f} mV "
        f"at {sim_time[peak_index]:.3f} ms; "
        f"first experimental peak={trace.voltage_mV[:n][first_window].max():.4f} mV"
    )

    fig, axes = plt.subplots(
        len(current_names) + 3, 1, figsize=(12, 2.1 * (len(current_names) + 3)),
        sharex=True, constrained_layout=True,
    )
    axes[0].plot(sim_time, trace.voltage_mV[:n], color="black", lw=1.2, label="experiment")
    axes[0].plot(sim_time, sim[:, 0], color="tab:red", lw=1.0, label="simulation")
    axes[0].set_ylabel("V (mV)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{args.candidate.stem} — {args.region} currents, {trace.trace} ({trace.protocol})")
    axes[1].plot(sim_time, soma_boundary, label="soma(1.0)")
    axes[1].plot(sim_time, axon_boundary, label="axon(0.0)")
    axes[1].set_ylabel("junction V (mV)")
    axes[1].legend(loc="upper right")
    axes[2].plot(sim_time, boundary_drop, label="axon(0)-soma(1)")
    axes[2].plot(sim_time, axial_current_nA, label="Iax estimate (nA)")
    axes[2].set_ylabel("drop / Iax")
    axes[2].legend(loc="upper right")
    for index, name in enumerate(current_names, start=3):
        axes[index].plot(sim_time, sim[:, index], lw=0.9)
        axes[index].axhline(0.0, color="0.7", lw=0.5)
        axes[index].set_ylabel(f"{channel_labels[index - 3]}\n(mA/cm²)")
    axes[-1].set_xlabel("Time from trace start (ms)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")
    for index, name in enumerate(names[1:], start=3):
        values_at_spike = sim[:, index]
        print(f"{name} min={values_at_spike.min():.6g} max={values_at_spike.max():.6g}")


if __name__ == "__main__":
    main()

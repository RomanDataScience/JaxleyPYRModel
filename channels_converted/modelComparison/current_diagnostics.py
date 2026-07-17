from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jaxley_mpl")
os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
import jaxley as jx  # noqa: E402
from neuron import h  # noqa: E402

from channels_converted.modelComparison.neuron_model import build_combe_neuron_model  # noqa: E402
from channels_converted.modelComparison.protocol import StepProtocol, step_current  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
JAXLEY_MODEL_DIR = REPO_ROOT / "JaxleyModel" / "model"
if str(JAXLEY_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(JAXLEY_MODEL_DIR))

from model_Combe import Combe2023  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "diagnostics"

SOMA_STATES = (
    "na16a_O1",
    "kd_m",
    "kd_h",
    "Kv2like_m",
    "Kv2like_h1",
    "Kv2like_h2",
    "h_n",
    "kap_n",
    "kap_l",
    "km_m",
    "cal_m",
    "cat_m",
    "cat_h",
    "car_m",
    "car_h",
    "icand_Po",
)
AXON_STATES = (
    "nax_m",
    "nax_h",
    "kd_m",
    "kd_h",
    "Kv2like_m",
    "Kv2like_h1",
    "Kv2like_h2",
    "kap_n",
    "kap_l",
    "km_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record soma/axon current diagnostics for NEURON vs Jaxley.")
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tstop", type=float, default=160.0)
    parser.add_argument("--delay", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--amp", type=float, default=1.5)
    parser.add_argument("--v-init", type=float, default=-72.0)
    parser.add_argument("--jaxley-d-lambda", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plot-window-start", type=float, default=95.0)
    parser.add_argument("--plot-window-stop", type=float, default=120.0)
    return parser.parse_args()


def safe_get(obj, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return default


def segment_current_snapshot(seg, site: str) -> dict[str, float]:
    values = {"v": float(seg.v)}

    values["leak"] = safe_get(seg, "g_pas") * (float(seg.v) - safe_get(seg, "e_pas"))

    if site == "soma":
        na16a = seg.na16a
        kd = seg.kd
        kv2 = seg.Kv2like
        hchan = seg.h
        kap = seg.kap
        km = seg.km
        cal = seg.cal
        cat = seg.cat
        car = seg.car
        icand = seg.icand

        values.update(
            {
                "na16a": safe_get(na16a, "gbar") * safe_get(na16a, "O1") * (float(seg.v) - float(seg.ena)),
                "kd": safe_get(kd, "i"),
                "Kv2like": safe_get(kv2, "gbar")
                * safe_get(kv2, "m") ** 2
                * (0.5 * safe_get(kv2, "h1") + 0.5 * safe_get(kv2, "h2"))
                * (float(seg.v) - float(seg.ek)),
                "h": safe_get(hchan, "i"),
                "kap": safe_get(kap, "i"),
                "km": safe_get(km, "gbar") * safe_get(km, "m") ** safe_get(km, "st", 1.0) * (float(seg.v) - float(seg.ek)),
                "cal": safe_get(cal, "ica"),
                "cat": safe_get(cat, "ica"),
                "car": safe_get(car, "ica"),
                "icand": safe_get(icand, "gbar") * safe_get(icand, "Po") * (float(seg.v) - safe_get(icand, "erev")),
                "na16a_O1": safe_get(na16a, "O1"),
                "kd_m": safe_get(kd, "m"),
                "kd_h": safe_get(kd, "h"),
                "Kv2like_m": safe_get(kv2, "m"),
                "Kv2like_h1": safe_get(kv2, "h1"),
                "Kv2like_h2": safe_get(kv2, "h2"),
                "h_n": safe_get(hchan, "n"),
                "kap_n": safe_get(kap, "n"),
                "kap_l": safe_get(kap, "l"),
                "km_m": safe_get(km, "m"),
                "cal_m": safe_get(cal, "m"),
                "cat_m": safe_get(cat, "m"),
                "cat_h": safe_get(cat, "h"),
                "car_m": safe_get(car, "m"),
                "car_h": safe_get(car, "h"),
                "icand_Po": safe_get(icand, "Po"),
            }
        )
    else:
        nax = seg.nax
        kd = seg.kd
        kv2 = seg.Kv2like
        kap = seg.kap
        km = seg.km
        values.update(
            {
                "nax": safe_get(nax, "gbar") * safe_get(nax, "m") ** 3 * safe_get(nax, "h") * (float(seg.v) - float(seg.ena)),
                "kd": safe_get(kd, "i"),
                "Kv2like": safe_get(kv2, "gbar")
                * safe_get(kv2, "m") ** 2
                * (0.5 * safe_get(kv2, "h1") + 0.5 * safe_get(kv2, "h2"))
                * (float(seg.v) - float(seg.ek)),
                "kap": safe_get(kap, "i"),
                "km": safe_get(km, "gbar") * safe_get(km, "m") ** safe_get(km, "st", 1.0) * (float(seg.v) - float(seg.ek)),
                "nax_m": safe_get(nax, "m"),
                "nax_h": safe_get(nax, "h"),
                "kd_m": safe_get(kd, "m"),
                "kd_h": safe_get(kd, "h"),
                "Kv2like_m": safe_get(kv2, "m"),
                "Kv2like_h1": safe_get(kv2, "h1"),
                "Kv2like_h2": safe_get(kv2, "h2"),
                "kap_n": safe_get(kap, "n"),
                "kap_l": safe_get(kap, "l"),
                "km_m": safe_get(km, "m"),
            }
        )

    current_keys = [key for key in values if key not in {"v"} and not key.endswith(("_m", "_h", "_n", "_l", "_O1", "_Po", "_h1", "_h2"))]
    values["total_membrane"] = float(sum(values[key] for key in current_keys))
    return values


def run_neuron_diagnostics(protocol: StepProtocol, d_lambda: float) -> dict[str, dict[str, np.ndarray]]:
    soma = build_combe_neuron_model(quiet=True, d_lambda=d_lambda)
    axon = h.axon[0]

    cvode = h.CVode()
    cvode.active(0)
    h.dt = protocol.dt
    h.tstop = protocol.tstop

    clamp = h.IClamp(soma(0.5))
    clamp.delay = protocol.delay
    clamp.dur = protocol.duration
    clamp.amp = protocol.amplitude

    n_steps = int(round(protocol.tstop / protocol.dt))
    rows = {"soma": [], "axon": []}
    times = []

    h.finitialize(protocol.v_init)
    h.fcurrent()
    for index in range(n_steps + 1):
        times.append(float(h.t))
        rows["soma"].append(segment_current_snapshot(soma(0.5), "soma"))
        rows["axon"].append(segment_current_snapshot(axon(0.5), "axon"))
        if index < n_steps:
            h.fadvance()

    time = np.asarray(times, dtype=float)
    out: dict[str, dict[str, np.ndarray]] = {}
    for site, site_rows in rows.items():
        keys = sorted(site_rows[0])
        out[site] = {"time": time}
        for key in keys:
            out[site][key] = np.asarray([row[key] for row in site_rows], dtype=float)
    return out


def add_recordings(view, states: tuple[str, ...]) -> int:
    rec_index = int(view.nodes.index[0])
    view.record("v", verbose=False)
    for state in states:
        view.record(state, verbose=False)
    return rec_index


def recording_map(cell, output: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    records = {}
    for row_index, row in cell.recordings.reset_index(drop=True).iterrows():
        records[(int(row["rec_index"]), str(row["state"]))] = np.asarray(output[row_index], dtype=float)
    return records


def node_value(cell, rec_index: int, key: str) -> float:
    return float(cell.nodes.loc[rec_index, key])


def jaxley_currents(cell, rec_index: int, recs: dict[tuple[int, str], np.ndarray], site: str) -> dict[str, np.ndarray]:
    v = recs[(rec_index, "v")]
    values: dict[str, np.ndarray] = {"v": v}
    values["leak"] = node_value(cell, rec_index, "Leak_gLeak") * (v - node_value(cell, rec_index, "Leak_eLeak"))

    def state(name: str) -> np.ndarray:
        return recs[(rec_index, name)]

    if site == "soma":
        values["na16a"] = node_value(cell, rec_index, "na16a_gbar") * state("na16a_O1") * (v - node_value(cell, rec_index, "eNa"))
        values["kd"] = node_value(cell, rec_index, "kd_gbar") * state("kd_m") * state("kd_h") * (v - node_value(cell, rec_index, "eK"))
        values["Kv2like"] = (
            node_value(cell, rec_index, "Kv2like_gbar")
            * state("Kv2like_m") ** 2
            * (0.5 * state("Kv2like_h1") + 0.5 * state("Kv2like_h2"))
            * (v - node_value(cell, rec_index, "eK"))
        )
        values["h"] = node_value(cell, rec_index, "h_gbar") * state("h_n") * (v - node_value(cell, rec_index, "h_eh"))
        values["kap"] = node_value(cell, rec_index, "kap_gkabar") * state("kap_n") * state("kap_l") * (v - node_value(cell, rec_index, "eK"))
        values["km"] = node_value(cell, rec_index, "km_gbar") * state("km_m") ** node_value(cell, rec_index, "km_st") * (v - node_value(cell, rec_index, "eK"))
        values["car"] = node_value(cell, rec_index, "car_gcabar") * state("car_m") ** 3 * state("car_h") * (v - node_value(cell, rec_index, "eCa"))
        values["icand"] = node_value(cell, rec_index, "icand_gbar") * state("icand_Po") * (v - node_value(cell, rec_index, "icand_erev"))
        for name in SOMA_STATES:
            values[name] = state(name)
    else:
        values["nax"] = node_value(cell, rec_index, "nax_gbar") * state("nax_m") ** 3 * state("nax_h") * (v - node_value(cell, rec_index, "eNa"))
        values["kd"] = node_value(cell, rec_index, "kd_gbar") * state("kd_m") * state("kd_h") * (v - node_value(cell, rec_index, "eK"))
        values["Kv2like"] = (
            node_value(cell, rec_index, "Kv2like_gbar")
            * state("Kv2like_m") ** 2
            * (0.5 * state("Kv2like_h1") + 0.5 * state("Kv2like_h2"))
            * (v - node_value(cell, rec_index, "eK"))
        )
        values["kap"] = node_value(cell, rec_index, "kap_gkabar") * state("kap_n") * state("kap_l") * (v - node_value(cell, rec_index, "eK"))
        values["km"] = node_value(cell, rec_index, "km_gbar") * state("km_m") ** node_value(cell, rec_index, "km_st") * (v - node_value(cell, rec_index, "eK"))
        for name in AXON_STATES:
            values[name] = state(name)

    current_keys = [key for key in values if key not in {"v"} and not key.endswith(("_m", "_h", "_n", "_l", "_O1", "_Po", "_h1", "_h2"))]
    values["total_membrane"] = np.sum([values[key] for key in current_keys], axis=0)
    return values


def run_jaxley_diagnostics(protocol: StepProtocol, d_lambda: float) -> dict[str, dict[str, np.ndarray]]:
    time, current = step_current(protocol)
    cell = Combe2023(d_lambda=d_lambda)
    cell.delete_stimuli()
    cell.delete_recordings()

    soma_view = cell.soma.branch(0).loc(0.5)
    axon_view = cell.axon.branch(0).loc(0.5)
    soma_index = add_recordings(soma_view, SOMA_STATES)
    axon_index = add_recordings(axon_view, AXON_STATES)

    cell.soma.branch(0).loc(0.5).stimulate(jnp.asarray(current, dtype=jnp.float64))
    cell.set("v", protocol.v_init)
    cell.init_states()
    output = np.asarray(jx.integrate(cell, delta_t=protocol.dt), dtype=float)
    recs = recording_map(cell, output)

    out = {
        "soma": {"time": time[: output.shape[1]], **jaxley_currents(cell, soma_index, recs, "soma")},
        "axon": {"time": time[: output.shape[1]], **jaxley_currents(cell, axon_index, recs, "axon")},
    }
    return out


def metrics_for(neuron: dict[str, np.ndarray], jaxley: dict[str, np.ndarray], key: str, mask: np.ndarray) -> dict[str, float]:
    time = neuron["time"][mask]
    n = neuron[key][mask]
    j = np.interp(time, jaxley["time"], jaxley[key])
    diff = j - n
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "neuron_peak_abs": float(np.max(np.abs(n))),
        "jaxley_peak_abs": float(np.max(np.abs(j))),
    }


def spike_times(time: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    crossings = np.where((voltage[:-1] < 0.0) & (voltage[1:] >= 0.0))[0]
    return time[crossings]


def write_metrics(path: Path, neuron: dict[str, dict[str, np.ndarray]], jaxley: dict[str, dict[str, np.ndarray]]) -> None:
    rows = []
    for site in ("soma", "axon"):
        nsite = neuron[site]
        jsite = jaxley[site]
        nspikes = spike_times(nsite["time"], nsite["v"])
        first_spike = float(nspikes[0]) if nspikes.size else float(nsite["time"][-1])
        mask = nsite["time"] < first_spike
        for key in sorted(set(nsite) & set(jsite) - {"time"}):
            values = metrics_for(nsite, jsite, key, mask)
            rows.append({"site": site, "variable": key, "window": f"pre_neuron_spike<{first_spike:g}ms", **values})

    rows.sort(key=lambda row: row["rmse"], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["site", "variable", "window", "rmse", "mae", "max_abs", "neuron_peak_abs", "jaxley_peak_abs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, prefix: str, data: dict[str, dict[str, np.ndarray]]) -> None:
    payload = {}
    for site, site_data in data.items():
        for key, value in site_data.items():
            payload[f"{prefix}_{site}_{key}"] = value
    np.savez(path, **payload)


def plot_overlay(path: Path, neuron: dict[str, dict[str, np.ndarray]], jaxley: dict[str, dict[str, np.ndarray]], start: float, stop: float) -> None:
    panels = [
        ("soma", "v"),
        ("axon", "v"),
        ("axon", "nax"),
        ("axon", "kd"),
        ("soma", "na16a"),
        ("soma", "kd"),
        ("soma", "Kv2like"),
        ("soma", "kap"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.8 * len(panels)), sharex=True)
    for axis, (site, key) in zip(axes, panels):
        for data, label, style in ((neuron, "NEURON", "-"), (jaxley, "Jaxley", "--")):
            mask = (data[site]["time"] >= start) & (data[site]["time"] <= stop)
            axis.plot(data[site]["time"][mask], data[site][key][mask], style, lw=0.9, label=label)
        axis.set_ylabel(f"{site} {key}")
        axis.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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

    neuron = run_neuron_diagnostics(protocol, args.jaxley_d_lambda)
    jaxley = run_jaxley_diagnostics(protocol, args.jaxley_d_lambda)

    save_npz(output_dir / "diagnostic_traces.npz", "neuron", neuron)
    save_npz(output_dir / "diagnostic_traces_jaxley.npz", "jaxley", jaxley)
    write_metrics(output_dir / "diagnostic_metrics.csv", neuron, jaxley)
    plot_overlay(output_dir / "diagnostic_overlay.png", neuron, jaxley, args.plot_window_start, args.plot_window_stop)

    n_soma_spikes = spike_times(neuron["soma"]["time"], neuron["soma"]["v"])
    j_soma_spikes = spike_times(jaxley["soma"]["time"], jaxley["soma"]["v"])
    print(f"Saved diagnostics to {output_dir}")
    print(f"NEURON soma spikes: {n_soma_spikes}")
    print(f"Jaxley soma spikes: {j_soma_spikes}")


if __name__ == "__main__":
    main()

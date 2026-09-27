"""How the whole morphology charges during the depolarizing step.

Runs the NEURON model under an idealized version of the recorded step
(holding current, then holding + step) and records voltage and capacitive
current in every segment. Three model variants share the candidate's passive
parameters:

* ``passive``    every active conductance set to zero;
* ``passive_h``  passive plus the candidate's Ih;
* ``candidate``  the candidate as fitted (sodium off for Stage A).

Each variant is also run with the holding current alone, and the step response
is ``V_step - V_hold``. For the passive variant that difference is exact. For
the active ones it removes any slow drift left after the lead-in.

A long passive step gives the true steady state and the membrane time
constants. The recorded hyperpolarizing pulse, the only subthreshold
recording, checks the passive charging speed against data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.optimize import curve_fit

from neuron_optim.config import load_config
from neuron_optim.data import load_protocol_traces
from neuron_optim.objective import detect_spikes
from neuron_optim.parameters import (
    DEFAULTS,
    complete_parameter_mapping,
    make_parameter_space,
    passive_model_values,
)
from neuron_optim.simulator import NeuronSimulator, _group, apply_parameters

REGIONS = ("soma", "basal", "apical", "axon")
REGION_COLORS = {"soma": "#222222", "basal": "#1f77b4", "apical": "#d62728", "axon": "#2ca02c"}
VARIANT_COLORS = {"passive": "#1f77b4", "passive_h": "#9467bd", "candidate": "#d62728"}


@dataclass
class Recording:
    time_ms: np.ndarray        # relative to the start of the (post lead-in) trial
    voltage_mV: np.ndarray     # (segments, time)
    icap_nA: np.ndarray        # (segments, time), outward positive


def load_candidate(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if "physical_by_name" in data:  # stage best.json
        values = dict(DEFAULTS)
        values.update(data["physical_by_name"])
        values.update(data.get("fixed_values", {}))
        return values
    manifest = json.loads((path.parent / "manifest.json").read_text())
    space = make_parameter_space(include=manifest["parameter_keys"])
    return complete_parameter_mapping(space, data["normalized"], manifest.get("fixed_values"))


def segment_table(h, soma):
    h.distance(0.0, sec=soma)
    rows = []
    for sec in h.allsec():
        for seg in sec:
            rows.append((sec, seg, _group(sec.name()), float(h.distance(seg.x, sec=sec)),
                         float(seg.area())))
    return rows


def simulate(h, soma, segments, values, *, dt, lead_ms, duration_ms, current_fn,
             v_init, record_dt) -> Recording:
    apply_parameters(soma, values, h)
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = dt
    total = lead_ms + duration_ms
    play_t = np.arange(0.0, total + dt, dt)
    clamp = h.IClamp(soma(0.5))
    clamp.delay = 0.0
    clamp.dur = total + dt
    tvec = h.Vector(play_t)
    ivec = h.Vector(current_fn(play_t - lead_ms))
    ivec.play(clamp._ref_amp, tvec, 1)
    sample_t = h.Vector(np.arange(lead_ms, total + 1e-9, record_dt))
    v_vecs, i_vecs = [], []
    for _sec, seg, *_ in segments:
        v_vecs.append(h.Vector().record(seg._ref_v, sample_t))
        i_vecs.append(h.Vector().record(seg._ref_i_cap, sample_t))
    h.finitialize(v_init)
    h.continuerun(total)
    n = int(sample_t.size())
    voltage = np.array([np.asarray(v)[:n] for v in v_vecs])
    area = np.array([row[4] for row in segments])[:, None]
    # mA/cm2 * um2 -> nA
    icap = np.array([np.asarray(i)[:n] for i in i_vecs]) * area * 1e-2
    ivec.play_remove()
    return Recording(np.asarray(sample_t)[:n] - lead_ms, voltage, icap)


def first_crossing(time, trace, level):
    above = np.flatnonzero(trace >= level)
    return float(time[above[0]]) if above.size else np.nan


def spike_free(trace, *, pre_ms=3.0, post_ms=8.0):
    spikes = detect_spikes(trace.time_ms, trace.voltage_mV)
    mask = np.ones(trace.time_ms.size, dtype=bool)
    for spike in spikes:
        mask &= ~((trace.time_ms >= spike.threshold_ms - pre_ms) &
                  (trace.time_ms <= spike.peak_ms + post_ms))
    return mask, spikes


def slope_mV_per_100ms(time, voltage, start, stop):
    window = (time >= start) & (time <= stop) & np.isfinite(voltage)
    if window.sum() < 3:
        return np.nan
    return float(np.polyfit(time[window], voltage[window], 1)[0] * 100.0)


def two_exp_charge(t, v_inf, c0, tau0, c1, tau1):
    return v_inf * (1.0 - c0 * np.exp(-t / tau0) - c1 * np.exp(-t / tau1))


def morphology_lines(h, segments):
    """3D polylines for each segment (for a shape plot) keyed by segment index."""
    index = {(sec.name(), round(seg.x, 9)): k for k, (sec, seg, *_rest) in enumerate(segments)}
    lines, owners = [], []
    for sec in h.allsec():
        n = int(sec.n3d())
        if n < 2:
            continue
        xyz = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n)])
        arc = np.array([sec.arc3d(i) for i in range(n)]) / max(sec.L, 1e-9)
        seg_x = [round(seg.x, 9) for seg in sec]
        nseg = len(seg_x)
        for i in range(n - 1):
            mid = 0.5 * (arc[i] + arc[i + 1])
            owner = index[(sec.name(), seg_x[min(int(mid * nseg), nseg - 1)])]
            lines.append(xyz[i:i + 2, :2])
            owners.append(owner)
    return lines, np.array(owners)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/depolarizing_two_stage.yaml"))
    parser.add_argument("--candidate", type=Path,
                        default=Path("runs/depol_two_stage/stageA_subthreshold/seed_000/best_so_far.json"))
    parser.add_argument("--trace", default="v75ctrl")
    parser.add_argument("--lead-ms", type=float, default=2000.0)
    parser.add_argument("--record-dt", type=float, default=0.25)
    parser.add_argument("--long-step-ms", type=float, default=3000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/passive_charging"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    candidate = load_candidate(args.candidate)
    variants = {
        "passive": passive_model_values(candidate),
        "candidate": candidate,
    }
    passive_h = passive_model_values(candidate)
    passive_h["soma_hbar"] = candidate["soma_hbar"]
    passive_h["h_tau_scale"] = candidate["h_tau_scale"]
    variants = {"passive": variants["passive"], "passive_h": passive_h,
                "candidate": variants["candidate"]}

    depol = load_protocol_traces(config.data_root, cell=config.cell, protocol="depolarizing_step",
                                 trace_names=(args.trace,), pre_ms=config.simulation_pre_ms,
                                 post_ms=config.simulation_post_ms, full_trial=True)[0]
    hyper = load_protocol_traces(config.data_root, cell=config.cell,
                                 protocol="hyperpolarizing_pulse", trace_names=(args.trace,),
                                 pre_ms=config.simulation_pre_ms,
                                 post_ms=config.simulation_post_ms, full_trial=True)[0]
    dt = float(np.median(np.diff(depol.time_ms)))
    on, off = depol.epoch_start_ms, depol.epoch_stop_ms
    pre = depol.time_ms < on
    during = (depol.time_ms > on + 5.0) & (depol.time_ms < off)
    i_hold = float(np.median(depol.current_nA[pre]))
    i_step = float(np.median(depol.current_nA[during])) - i_hold
    duration = float(depol.time_ms[-1])

    simulator = NeuronSimulator(d_lambda=config.d_lambda, quiet=True)
    h, soma = simulator._model()
    segments = segment_table(h, soma)
    soma_idx = next(k for k, row in enumerate(segments)
                    if row[2] == "soma" and abs(row[1].x - 0.5) < 0.5 / row[0].nseg + 1e-9)
    regions = np.array([row[2] for row in segments])
    distance = np.array([row[3] for row in segments])
    area = np.array([row[4] for row in segments])
    print(f"{len(segments)} segments; step {i_step:.4f} nA on holding {i_hold:.4f} nA; "
          f"step {on:.0f}-{off:.0f} ms")

    step_fn = lambda t: np.where((t >= on) & (t < off), i_hold + i_step, i_hold)
    hold_fn = lambda t: np.full_like(t, i_hold)
    common = dict(dt=dt, lead_ms=args.lead_ms, duration_ms=duration,
                  v_init=float(depol.voltage_mV[0]), record_dt=args.record_dt)
    runs = {}
    for name, values in variants.items():
        step = simulate(h, soma, segments, values, current_fn=step_fn, **common)
        hold = simulate(h, soma, segments, values, current_fn=hold_fn, **common)
        runs[name] = (step, hold)
        print(f"  {name}: soma V_hold {hold.voltage_mV[soma_idx, 0]:.2f} mV, "
              f"hold drift over trial {hold.voltage_mV[soma_idx, -1] - hold.voltage_mV[soma_idx, 0]:+.3f} mV")

    # Long passive step: true steady state and time constants.
    long_fn = lambda t: np.where(t >= 0.0, i_hold + i_step, i_hold)
    long_step = simulate(h, soma, segments, variants["passive"], current_fn=long_fn,
                         dt=dt, lead_ms=args.lead_ms, duration_ms=args.long_step_ms,
                         v_init=float(depol.voltage_mV[0]), record_dt=1.0)
    long_dv = long_step.voltage_mV - long_step.voltage_mV[:, :1]
    dv_inf = long_dv[:, -1]
    rin = dv_inf[soma_idx] / i_step
    fit_t = long_step.time_ms[1:]
    popt, _ = curve_fit(two_exp_charge, fit_t, long_dv[soma_idx, 1:],
                        p0=(dv_inf[soma_idx], 0.8, 50.0, 0.2, 5.0), maxfev=20000)
    taus = sorted([(popt[2], popt[1]), (popt[4], popt[3])], reverse=True)

    # Hyperpolarizing pulse: recorded current, passive and candidate models.
    hyper_runs = {}
    h_dt = float(np.median(np.diff(hyper.time_ms)))
    h_pre = hyper.time_ms < hyper.epoch_start_ms
    h_hold = float(np.median(hyper.current_nA[h_pre]))
    h_during = (hyper.time_ms > hyper.epoch_start_ms + 2) & (hyper.time_ms < hyper.epoch_stop_ms)
    h_step = float(np.median(hyper.current_nA[h_during])) - h_hold
    h_fn = lambda t: np.where((t >= hyper.epoch_start_ms) & (t < hyper.epoch_stop_ms),
                              h_hold + h_step, h_hold)
    for name in ("passive", "candidate"):
        rec = simulate(h, soma, segments, variants[name], current_fn=h_fn, dt=h_dt,
                       lead_ms=args.lead_ms, duration_ms=float(hyper.time_ms[-1]),
                       v_init=float(hyper.voltage_mV[0]), record_dt=h_dt)
        hyper_runs[name] = rec.voltage_mV[soma_idx]
    hyper_t = rec.time_ms

    # ---------------- metrics ----------------
    t = runs["passive"][0].time_ms
    in_step = (t >= on) & (t <= off)
    end_i = int(np.argmin(np.abs(t - off)))
    summary = {"segments": len(segments), "i_hold_nA": i_hold, "i_step_nA": i_step,
               "passive_Rin_MOhm": rin, "passive_tau_ms": [float(x[0]) for x in taus],
               "passive_tau_weights": [float(x[1]) for x in taus],
               "passive_fraction_of_steady_state_at_step_end":
                   float(runs["passive"][0].voltage_mV[soma_idx, end_i]
                         - runs["passive"][1].voltage_mV[soma_idx, end_i]) / dv_inf[soma_idx],
               "variants": {}}
    mask, spikes = spike_free(depol)
    exp_v = np.where(mask, depol.voltage_mV, np.nan)
    exp_base = float(np.median(depol.voltage_mV[pre & (depol.time_ms > on - 100)]))
    windows = ((on + 50, on + 100), (on + 100, on + 200), (on + 200, off), (on + 100, off))
    summary["experiment"] = {
        "spikes": len(spikes),
        "spike_free_slope_mV_per_100ms": {f"{a - on:.0f}-{b - on:.0f}":
                                          slope_mV_per_100ms(depol.time_ms, exp_v, a, b)
                                          for a, b in windows},
    }
    per_segment = {}
    for name, (step, hold) in runs.items():
        dv = step.voltage_mV - hold.voltage_mV
        dv_end = dv[:, end_i]
        soma_dv = dv[soma_idx]
        frac = soma_dv / dv_end[soma_idx]
        stats = {
            "soma_dV_end_mV": float(dv_end[soma_idx]),
            "soma_peak_dV_mV": float(soma_dv[in_step].max()),
            "soma_V_end_mV": float(step.voltage_mV[soma_idx, end_i]),
            "soma_slope_mV_per_100ms": {f"{a - on:.0f}-{b - on:.0f}":
                                        slope_mV_per_100ms(t, soma_dv, a, b) for a, b in windows},
            "soma_fraction_of_end_at_ms": {str(ms): float(np.interp(on + ms, t, frac))
                                           for ms in (10, 25, 50, 100, 200)},
            "soma_t50_ms": first_crossing(t[in_step], frac[in_step], 0.5) - on,
            "soma_t90_ms": first_crossing(t[in_step], frac[in_step], 0.9) - on,
        }
        after = t >= off
        tail = soma_dv[after] / dv_end[soma_idx]
        stats["soma_tail_fraction_at_ms_after_off"] = {
            str(ms): float(np.interp(off + ms, t[after], tail)) for ms in (25, 50, 100, 200, 400)}
        # Per-segment charging time, relative to each segment's own end-of-step value.
        seg_t90 = np.array([first_crossing(t[in_step], dv[k, in_step] / dv_end[k], 0.9) - on
                            if dv_end[k] > 1e-6 else np.nan for k in range(len(segments))])
        region_stats = {}
        charge = np.trapezoid(step.icap_nA - hold.icap_nA, t, axis=1)  # pC
        for region in REGIONS:
            sel = regions == region
            region_stats[region] = {
                "n_segments": int(sel.sum()),
                "area_um2": float(area[sel].sum()),
                "median_t90_ms": float(np.nanmedian(seg_t90[sel])),
                "max_t90_ms": float(np.nanmax(seg_t90[sel])),
                "dV_end_over_soma": float(np.average(dv_end[sel], weights=area[sel])
                                          / dv_end[soma_idx]),
            }
        stats["regions"] = region_stats
        per_segment[name] = (dv, seg_t90, dv_end)
        summary["variants"][name] = stats

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "recordings.npz", time_ms=t, regions=regions, distance_um=distance,
                        area_um2=area, **{f"{n}_dV": per_segment[n][0] for n in per_segment},
                        **{f"{n}_icap_nA": runs[n][0].icap_nA - runs[n][1].icap_nA for n in runs})

    # ---------------- figure 1: soma ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.plot(depol.time_ms, depol.voltage_mV, color="0.75", lw=0.6, label=f"experiment {args.trace}")
    ax.plot(depol.time_ms, exp_v, color="k", lw=0.8, label="experiment, spikes removed")
    for name, (step, _hold) in runs.items():
        ax.plot(t, step.voltage_mV[soma_idx], color=VARIANT_COLORS[name], lw=1.2, label=name)
    ax.set(xlim=(on - 100, duration), ylim=(-80, 0), xlabel="time (ms)", ylabel="soma V (mV)",
           title="Soma voltage, idealized step")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name in runs:
        dv = per_segment[name][0][soma_idx]
        ax.plot(t - on, dv / per_segment[name][2][soma_idx], color=VARIANT_COLORS[name], label=name)
    long_frac = long_dv[soma_idx] / per_segment["passive"][2][soma_idx]
    ax.plot(long_step.time_ms, long_frac, color=VARIANT_COLORS["passive"], ls=":",
            label="passive, long step")
    exp_dv = exp_v - exp_base
    exp_end = np.nanmedian(exp_dv[(depol.time_ms > off - 30) & (depol.time_ms < off)])
    ax.plot(depol.time_ms - on, exp_dv / exp_end, color="k", lw=0.6, alpha=0.6,
            label="experiment (spike-free)")
    ax.axhline(1.0, color="0.6", lw=0.5)
    ax.set(xlim=(-20, duration - on), ylim=(-0.2, 1.3), xlabel="time from step onset (ms)",
           ylabel="ΔV / ΔV(step end)", title="Normalized soma step response")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    kernel = np.ones(int(round(10.0 / args.record_dt)))
    kernel /= kernel.size
    for name in runs:
        dv = per_segment[name][0][soma_idx]
        slope = np.gradient(np.convolve(dv, kernel, mode="same"), t) * 100.0
        ax.plot(t - on, slope, color=VARIANT_COLORS[name], label=name)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set(xlim=(0, off - on + 200), ylim=(-15, 30), xlabel="time from step onset (ms)",
           ylabel="dV/dt (mV / 100 ms), 10 ms smoothing", title="Soma slope")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(hyper.time_ms - hyper.epoch_start_ms, hyper.voltage_mV
            - np.median(hyper.voltage_mV[h_pre][-400:]), color="k", lw=0.7, label="experiment")
    for name, v in hyper_runs.items():
        base = np.median(v[(hyper_t < hyper.epoch_start_ms) & (hyper_t > hyper.epoch_start_ms - 20)])
        ax.plot(hyper_t - hyper.epoch_start_ms, v - base, color=VARIANT_COLORS[name], label=name)
    ax.set(xlim=(-20, hyper.time_ms[-1] - hyper.epoch_start_ms), xlabel="time from pulse onset (ms)",
           ylabel="ΔV (mV)", title=f"Hyperpolarizing pulse ({h_step * 1e3:.0f} pA, recorded)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "soma_charging.png", dpi=130)
    plt.close(fig)

    # ---------------- figure 2: whole morphology ----------------
    for name in runs:
        dv, seg_t90, dv_end = per_segment[name]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax = axes[0, 0]
        for region in REGIONS:
            sel = regions == region
            ax.scatter(distance[sel], seg_t90[sel], s=6, color=REGION_COLORS[region], label=region)
        ax.axhline(summary["variants"][name]["soma_t90_ms"], color="k", lw=0.6, ls="--")
        ax.set(xlabel="path distance from soma (µm)", ylabel="t90 (ms from onset)",
               title="Time to 90% of each segment's end-of-step ΔV")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        for region in REGIONS:
            sel = regions == region
            ax.scatter(distance[sel], dv_end[sel], s=6, color=REGION_COLORS[region], label=region)
        ax.set(xlabel="path distance from soma (µm)", ylabel="ΔV at step end (mV)",
               title="Steady depolarization along the cell")

        ax = axes[1, 0]
        step, hold = runs[name]
        icap = step.icap_nA - hold.icap_nA
        for region in REGIONS:
            ax.plot(t - on, icap[regions == region].sum(axis=0), color=REGION_COLORS[region],
                    label=f"{region} ({summary['variants'][name]['regions'][region]['area_um2'] / 1e3:.0f}k µm²)")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set(xlim=(-5, off - on + 300), xlabel="time from step onset (ms)",
               ylabel="capacitive current (nA)", yscale="symlog",
               title="Where the injected current goes: C dV/dt by region")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        order = np.lexsort((distance, np.searchsorted(REGIONS, regions, sorter=np.argsort(REGIONS))))
        rel = dv[order] / dv_end[soma_idx]
        image = ax.imshow(rel, aspect="auto", cmap="viridis", vmin=0, vmax=1.1,
                          extent=(t[0] - on, t[-1] - on, len(order), 0), interpolation="nearest")
        bounds = np.cumsum([0] + [int((regions == r).sum()) for r in sorted(REGIONS)])
        for b, r in zip(bounds[:-1], sorted(REGIONS)):
            ax.text(t[-1] - on, b + 1, f" {r}", va="top", fontsize=8, color="w", ha="right")
            ax.axhline(b, color="w", lw=0.5)
        ax.set(xlim=(-20, duration - on), xlabel="time from step onset (ms)",
               ylabel="segment (grouped by region, sorted by distance)",
               title="ΔV / soma ΔV(step end)")
        fig.colorbar(image, ax=ax)
        fig.suptitle(f"{name}: charging across the morphology")
        fig.tight_layout()
        fig.savefig(out / f"morphology_{name}.png", dpi=130)
        plt.close(fig)

    # ---------------- figure 3: morphology snapshots (passive) ----------------
    lines, owners = morphology_lines(h, segments)
    dv, _seg_t90, dv_end = per_segment["passive"]
    snaps = (5, 20, 50, 100, off - on)
    fig, axes = plt.subplots(1, len(snaps), figsize=(4 * len(snaps), 5.5))
    for ax, ms in zip(axes, snaps):
        k = int(np.argmin(np.abs(t - (on + ms))))
        values = dv[:, k] / np.where(np.abs(dv_end) > 1e-9, dv_end, np.nan)
        collection = LineCollection(lines, array=values[owners], cmap="viridis", linewidths=1.0)
        collection.set_clim(0, 1)
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{ms:.0f} ms after onset")
    fig.colorbar(collection, ax=axes, shrink=0.6, label="ΔV / own ΔV(step end)")
    fig.suptitle("Passive model: fraction charged in each segment")
    fig.savefig(out / "morphology_snapshots_passive.png", dpi=130)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

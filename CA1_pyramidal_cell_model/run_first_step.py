#!/usr/bin/env python3
"""Replay the first depolarizing step (v75ctrl) in the CA1 pyramidal cell model.

The cell is built exactly as in ``loader_script_opt_Luca_model_soma_and_axon.py``
(same HOC loader, parameter vector and F-factor spine correction), but instead
of the loader's fixed 0.25 nA step the recorded current of the experimental
trace is played into the soma. The result is plotted against the recording.

The HOC setup inserts a ``vmax`` mechanism that is not in ``mod_files``; it only
records peak voltage, so a current-free stand-in is compiled with the other
mechanisms in a private build directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fitting.data import load_dataset  # noqa: E402
from fitting.objective import score_trace  # noqa: E402
from fitting.onset import onset_metrics  # noqa: E402

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

LOADER_HOC = "load_model_na_inhomo_minimal_model_full_soma_f_fact_true_diam_active_spine_KA_fact_50Ra.hoc"

# Parameter vector from loader_script_opt_Luca_model_soma_and_axon.py.
V = [-47.7233744163709, 0.011401720134011, 0.17866165872572, 0.110533546361937,
     0.018195328014435, 4.13376819751696E-06, 0.558280689990683, -55.1977064419009,
     0.000102007454842, 2.35151579716906, 0.00018773285843, 0.008177991377281,
     0.082608363126436, 3.69975735209366, 0.00058114570326, 3.69310750687799,
     5.1266274824495E-05, 1.15317561526235E-05, -25.8009959332103, 9.72470557972213,
     9.75671751943343]

VMAX_STUB = """NEURON {
    SUFFIX vmax
    RANGE vm
}
ASSIGNED {
    v (millivolt)
    vm (millivolt)
}
INITIAL {
    vm = v
}
BREAKPOINT {
    if (v > vm) { vm = v }
}
"""


def build_mechanisms(force: bool = False) -> Path:
    build = HERE / ".cache" / "mod"
    if force and build.exists():
        shutil.rmtree(build)
    if not list(build.glob("*/special")) and not list(build.glob("*/.libs/libnrnmech.*")):
        build.mkdir(parents=True, exist_ok=True)
        for mod in (HERE / "mod_files").glob("*.mod"):
            shutil.copy2(mod, build / mod.name)
        (build / "vmax.mod").write_text(VMAX_STUB, encoding="utf-8")
        subprocess.run(["nrnivmodl"], cwd=build, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return build


def build_cell(mod_dir: Path):
    import neuron
    from neuron import h

    neuron.load_mechanisms(str(mod_dir))
    old_cwd = Path.cwd()
    os.chdir(HERE)
    try:
        h.load_file(LOADER_HOC)
    finally:
        os.chdir(old_cwd)
    apply_loader_parameters(h, V)
    h.add_F_factor_to_all_dendrites()
    return h


def apply_loader_parameters(h, v: list[float]) -> None:
    """Verbatim port of the parameter assignment in the loader script."""
    for sec in h.all:
        sec.gmax_Leak_pyr = v[5]
        sec.e_Leak_pyr = v[7]
        sec.cm = 1
        sec.Ra = 50
    for sec in h.all_dendrites:
        sec.gkd_kd_params3 = v[10]
        sec.shift_K_A_prox = v[19]
        sec.shift_K_A_dist = v[19]
        sec.X_k0_K_A_prox = v[20]
        sec.X_k0_K_A_dist = v[20]
        sec.Y_v0_Na_BG_dend = v[0]
        sec.gmax_K_DRS4_params_voltage_dep = v[1]
        sec.gbar_km_q10_2 = v[8]
        sec.pbar_car = v[17]
        sec.X_v0_Na_BG_dend = v[18]
        for seg in sec:
            h.distance(sec=h.soma)
            dist = h.distance(seg.x, sec=sec)
            seg.gmax_Na_BG_dend = v[2] + v[2] * (-0.002) * dist
            seg.gk_sKCa = v[16] + v[16] * 0.06 * dist
            if 100 < dist <= 150:
                seg.gmax_H_CA1pyr_dist = v[6] * 0.00002 + v[6] * 4e-07 * dist
                seg.gmax_H_CA1pyr_prox = 0
                seg.gmax_K_A_prox = 0
                seg.gmax_K_A_dist = 0.0035 * v[9] + v[9] * 5.5e-05 * dist
                seg.pbar_CaL_pool2_ghk = 0.002 * 0.2 * v[13] + v[13] * 0.2 * (-1.3333e-05) * dist
            elif 150.0 < dist <= 400.0:
                seg.gmax_H_CA1pyr_dist = v[6] * 0.00002 + v[6] * 4e-07 * dist
                seg.gmax_H_CA1pyr_prox = 0
                seg.gmax_K_A_prox = 0
                seg.gmax_K_A_dist = 0.0035 * v[9] + v[9] * 5.5e-05 * dist
                seg.pbar_CaL_pool2_ghk = 0.002 * 0.2 * v[13] + v[13] * 0.2 * (-1.3333e-05) * 150
            elif 0.0 < dist <= 100.0:
                seg.gmax_H_CA1pyr_dist = 0
                seg.gmax_H_CA1pyr_prox = v[6] * 0.00002 + v[6] * 4e-07 * dist
                seg.gmax_K_A_prox = 0.0035 * v[9] + v[9] * 5.5e-05 * dist
                seg.gmax_K_A_dist = 0
                seg.pbar_CaL_pool2_ghk = 0.002 * 0.2 * v[13] + v[13] * 0.2 * (-1.3333e-05) * dist
            elif dist > 400.0:
                seg.gmax_H_CA1pyr_dist = 0.00018 * v[6]
                seg.gmax_H_CA1pyr_prox = 0
                seg.gmax_K_A_prox = 0
                seg.gmax_K_A_dist = 0.0255 * v[9]
                seg.pbar_CaL_pool2_ghk = 0.002 * 0.2 * v[13] + v[13] * 0.2 * (-1.3333e-05) * 150
                sec.gmax_Na_BG_dend = 0
            else:
                seg.gmax_H_CA1pyr_prox = 0.00002 * v[6]
                seg.gmax_H_CA1pyr_dist = 0
                seg.gmax_K_A_prox = 0.0035 * v[9]
                seg.gmax_K_A_dist = 0
                seg.pbar_CaL_pool2_ghk = 0.002 * 0.2 * v[13]
    for sec in h.oblique_dendrites:
        sec.pbar_cat3 = 0.000072608 * 0.01 * v[15]
        sec.gmax_Na_BG_dend = v[2] * (2 / 3)
    for sec in h.trunk:
        sec.pbar_cat3 = 0.000024292 * 0.01 * v[15]
    for sec in h.basal_dendrites:
        sec.pbar_cat3 = 0.000069983 * 0.01 * v[15]
        sec.gmax_Na_BG_dend = v[2] * (2 / 3)
    for sec in h.tuft:
        sec.pbar_cat3 = 0.000043133 * 0.01 * v[15]
        sec.gmax_Na_BG_dend = v[2] / 3
    for sec in h.soma:
        sec.gmax_Na_BG_soma = v[2]
        sec.Y_v0_Na_BG_soma = v[0]
        sec.gmax_K_DRS4_params_voltage_dep = v[3]
        sec.gmax_H_CA1pyr_prox = 0.00002 * v[6]
        sec.gbar_km_q10_2 = v[8]
        sec.gmax_K_A_prox = 0.0035 * v[9]
        sec.gkd_kd_params3 = v[10]
        sec.pbar_CaN_BG_pool2_ghk = v[11]
        sec.gbar_bk_ch_pool = v[12]
        sec.pbar_CaL_pool2_ghk = 0.00021 * 0.2 * v[13]
        sec.gmax_K_AHP3_lpool = v[14]
        sec.pbar_cat3 = 0.000002125 * 0.01 * v[15]
        sec.gk_sKCa = v[16]
        sec.X_v0_Na_BG_soma = v[18]
        sec.shift_K_A_prox = v[19]
        sec.X_k0_K_A_prox = v[20]
    for sec in h.all_axon:
        sec.gmax_Na_BG_axon = v[2] * 40
        sec.Y_v0_Na_BG_axon = v[0] - 3
        sec.gmax_K_DRS4_params_voltage_dep = v[3] * 40
        sec.gmax_H_CA1pyr_prox = 0.00002 * v[6]
        sec.gbar_km_q10_2 = v[8] * 4
        sec.gkd_kd_params3 = v[4]
        sec.X_v0_Na_BG_axon = v[18] - 3


def simulate(h, trace, prerun_ms: float, v_init: float, holding_nA: float):
    """Play the recorded current, preceded by ``prerun_ms`` at its first value."""
    dt = float(np.median(np.diff(trace.time_ms)))
    shift = round(prerun_ms / dt) * dt
    time = np.concatenate([np.arange(0.0, shift, dt), trace.time_ms + shift])
    current = np.concatenate([np.full(time.size - trace.time_ms.size, trace.current_nA[0]),
                              trace.current_nA]) + holding_nA
    clamp = h.IClamp(h.soma(0.5))
    clamp.delay = 0.0
    clamp.dur = 1e9
    tvec, ivec = h.Vector(time), h.Vector(current)
    ivec.play(clamp._ref_amp, tvec, 1)
    rec_t = h.Vector().record(h._ref_t)
    rec_v = h.Vector().record(h.soma(0.5)._ref_v)
    rec_ax = h.Vector().record(h.axon[0](0.5)._ref_v)
    h.CVode().active(0)
    h.dt = dt
    h.steps_per_ms = 1.0 / dt
    h.finitialize(v_init)
    h.continuerun(float(time[-1]))
    t = np.asarray(rec_t).copy() - shift
    keep = t >= -1e-9
    return t[keep], np.asarray(rec_v)[keep], np.asarray(rec_ax)[keep]


def resting_potential(h, trace, prerun_ms: float, v_init: float, holding_nA: float) -> float:
    """Model voltage at the end of the pre-step period for a given holding current."""
    dt = float(np.median(np.diff(trace.time_ms)))
    clamp = h.IClamp(h.soma(0.5))
    clamp.delay = 0.0
    clamp.dur = 1e9
    clamp.amp = float(np.median(trace.current_nA[trace.time_ms < trace.epoch_start_ms])) + holding_nA
    h.CVode().active(0)
    h.dt = dt
    h.steps_per_ms = 1.0 / dt
    h.finitialize(v_init)
    h.continuerun(prerun_ms + trace.epoch_start_ms)
    return float(h.soma(0.5).v)


def match_baseline(h, trace, prerun_ms: float, v_init: float, tol_mV: float = 0.1) -> float:
    """Secant search for the holding current that puts rest at the recorded baseline."""
    target = float(np.median(trace.voltage_mV[trace.time_ms <= trace.epoch_start_ms]))
    holds = [0.0, -0.05]
    rests = [resting_potential(h, trace, prerun_ms, v_init, hold) for hold in holds]
    for _ in range(6):
        if abs(rests[-1] - target) <= tol_mV:
            break
        slope = (rests[-1] - rests[-2]) / (holds[-1] - holds[-2])
        holds.append(holds[-1] + (target - rests[-1]) / slope)
        rests.append(resting_potential(h, trace, prerun_ms, v_init, holds[-1]))
    print(f"holding current {holds[-1] * 1e3:+.1f} pA -> rest {rests[-1]:.2f} mV "
          f"(target {target:.2f} mV)", flush=True)
    return round(holds[-1], 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="v75ctrl")
    parser.add_argument("--cell", default="m20240527cd")
    parser.add_argument("--prerun-ms", type=float, default=1000.0)
    parser.add_argument("--v-init", type=float, default=-70.0)
    parser.add_argument("--holding-nA", type=float, default=0.0,
                        help="extra constant current added to the recorded one")
    parser.add_argument("--match-baseline", action="store_true",
                        help="choose --holding-nA so the model rests at the recorded baseline")
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "first_step")
    args = parser.parse_args()

    data_root = HERE / "Experimental_currentClamp_Analysis" / "Segmented_Traces"
    depolarizing, _ = load_dataset(data_root, args.cell, [args.trace], args.trace)
    trace = depolarizing[0]
    h = build_cell(build_mechanisms())
    if args.match_baseline:
        args.holding_nA = match_baseline(h, trace, args.prerun_ms, args.v_init)
    time, soma_v, axon_v = simulate(h, trace, args.prerun_ms, args.v_init, args.holding_nA)

    loss, details = score_trace(trace, time, soma_v)
    sim_on_grid = np.interp(trace.time_ms, time, soma_v)
    result = {
        "trace": args.trace,
        "holding_nA": args.holding_nA,
        "step_current_nA": float(np.median(trace.current_nA[(trace.time_ms > 210) & (trace.time_ms < 490)])
                                 - np.median(trace.current_nA[trace.time_ms < 200])),
        "experiment": onset_metrics(trace.time_ms, trace.voltage_mV, trace.epoch_start_ms, trace.epoch_stop_ms),
        "model": onset_metrics(trace.time_ms, sim_on_grid, trace.epoch_start_ms, trace.epoch_stop_ms),
        "baseline_mV": details["baseline_mV"],
        "trace_loss_combe_objective": loss,
        "loss_components": details["components"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.cell}_{args.trace}" + (f"_hold{args.holding_nA:+.3f}nA" if args.holding_nA else "")
    (args.output_dir / f"{tag}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(args.output_dir / f"{tag}_simulation.npz", time_ms=time, soma_mV=soma_v, axon_mV=axon_v)

    start = trace.epoch_start_ms
    figure, (full, zoom) = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for axis, (lo, hi) in ((full, (0.0, trace.time_ms[-1])), (zoom, (start - 50.0, start + 150.0))):
        axis.axvspan(trace.epoch_start_ms, trace.epoch_stop_ms, color="#D55E00", alpha=0.12,
                     label="current step")
        axis.plot(trace.time_ms, trace.voltage_mV, color="black", lw=0.9, label="experiment")
        axis.plot(time, soma_v, color="#009E73", lw=0.9, label="CA1 model soma")
        axis.set_xlim(lo, hi)
        axis.set_ylabel("Voltage (mV)")
        axis.grid(alpha=0.2)
    full.set_title(f"{args.trace} — CA1_pyramidal_cell_model vs experiment "
                   f"(step {result['step_current_nA'] * 1e3:.0f} pA"
                   + (f", holding {args.holding_nA * 1e3:+.0f} pA" if args.holding_nA else "") + ")")
    full.legend(frameon=False, ncol=3, loc="upper right")
    zoom.set_xlabel("Time (ms)")
    figure.savefig(args.output_dir / f"{tag}_vs_experiment.png", dpi=160)
    plt.close(figure)
    print(json.dumps({k: result[k] for k in ("experiment", "model", "baseline_mV",
                                             "trace_loss_combe_objective")}, indent=2))
    print(args.output_dir / f"{tag}_vs_experiment.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

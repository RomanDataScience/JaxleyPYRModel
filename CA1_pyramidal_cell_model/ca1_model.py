"""NEURON construction, parameter application and trace replay for the CA1 model.

The cell is built as in ``loader_script_opt_Luca_model_soma_and_axon.py``: the
HOC loader constructs morphology and channels, the fitted parameter vector is
applied, then the F-factor spine correction is added to every dendrite.

Every quantity that ``add_F_factor_to_all_dendrites`` scales is reassigned
absolutely by ``apply_parameters`` first, so parameters can be reapplied to one
built cell without rebuilding it.

The HOC setup inserts a ``vmax`` mechanism that is not in ``mod_files``; it only
records peak voltage, so a current-free stand-in is compiled with the other
mechanisms in a private build directory.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ca1_parameters import DEFAULTS, VECTOR_KEYS

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

HERE = Path(__file__).resolve().parent
LOADER_HOC = "load_model_na_inhomo_minimal_model_full_soma_f_fact_true_diam_active_spine_KA_fact_50Ra.hoc"

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
    # nrnivmodl builds into a per-architecture folder (arm64/, x86_64/), so a
    # build copied from another machine does not count.
    arch = build / platform.machine()
    if not (arch / "special").exists() and not list(arch.glob(".libs/libnrnmech.*")):
        build.mkdir(parents=True, exist_ok=True)
        for mod in (HERE / "mod_files").glob("*.mod"):
            shutil.copy2(mod, build / mod.name)
        (build / "vmax.mod").write_text(VMAX_STUB, encoding="utf-8")
        # Fall back to the env's bin/ when it is not on PATH (python called by full path).
        nrnivmodl = shutil.which("nrnivmodl") or str(Path(sys.executable).with_name("nrnivmodl"))
        subprocess.run([nrnivmodl], cwd=build, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return build


def build_cell(mod_dir: Path | None = None):
    """Load mechanisms and the HOC model once per process; return ``h``."""
    import neuron
    from neuron import h

    try:
        neuron.load_mechanisms(str(mod_dir or build_mechanisms()))
    except RuntimeError as exc:
        if "user defined name already exists" not in str(exc).lower():
            raise
    old_cwd = Path.cwd()
    os.chdir(HERE)
    try:
        h.load_file(LOADER_HOC)
    finally:
        os.chdir(old_cwd)
    return h


def apply_parameters(h, values: dict[str, float]) -> None:
    """Apply a named parameter set (see ``ca1_parameters``) and the F factor."""
    v = [float(values[key]) for key in VECTOR_KEYS]
    cm = float(values.get("cm", DEFAULTS["cm"]))
    ra = float(values.get("Ra", DEFAULTS["Ra"]))
    for sec in h.all:
        sec.gmax_Leak_pyr = v[5]
        sec.e_Leak_pyr = v[7]
        sec.cm = cm
        sec.Ra = ra
    h.distance(sec=h.soma)
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
    h.add_F_factor_to_all_dendrites()


def simulate(h, trace, *, prerun_ms: float = 500.0, v_init: float = -70.0,
             holding_nA: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Play the recorded current, preceded by ``prerun_ms`` at its first value.

    Returns time (ms, on the trace's clock), soma and axon[0] voltage.
    """
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
    h.secondorder = 0
    h.dt = dt
    h.steps_per_ms = 1.0 / dt
    h.finitialize(v_init)
    h.continuerun(float(time[-1]))
    t = np.asarray(rec_t).copy() - shift
    keep = t >= -1e-9
    soma_v = np.asarray(rec_v).copy()[keep]
    axon_v = np.asarray(rec_ax).copy()[keep]
    ivec.play_remove()
    if not np.isfinite(soma_v).all():
        raise RuntimeError("NEURON returned a non-finite voltage trace")
    return t[keep], soma_v, axon_v

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

import neuron  # noqa: E402
from neuron import h  # noqa: E402

from channels_converted.modelComparison.protocol import (
    StepProtocol,
    current_at_time_points,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
COMBE_DIR = REPO_ROOT / "Combe2023"
MOD_DIR = REPO_ROOT / "channels_converted" / "mod"


@contextlib.contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextlib.contextmanager
def _quiet_stdout(enabled: bool):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    old_stdout = os.dup(1)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            yield
        finally:
            ctypes.CDLL(None).fflush(None)
            sys.stdout.flush()
            os.dup2(old_stdout, 1)
            os.close(old_stdout)


def _delete_sections() -> None:
    for section in list(h.allsec()):
        h.delete_section(sec=section)


def _load_mechanisms(mod_dir: Path = MOD_DIR) -> None:
    mod_dir = mod_dir.resolve()
    loaded = neuron.load_mechanisms(str(mod_dir))
    has_compiled_library = bool(list(mod_dir.glob("*/special")))
    if not loaded and not has_compiled_library:
        raise RuntimeError(
            f"No compiled NEURON mechanisms found under {mod_dir}. "
            "Run nrnivmodl in channels_converted/mod first."
        )


def build_combe_neuron_model(*, quiet: bool = True, d_lambda: float | None = None):
    """Load Combe2023/cell_setup_pc2b_CCh_driven.hoc and return soma[0]."""
    with _quiet_stdout(quiet):
        _load_mechanisms()
        _delete_sections()

        with _pushd(COMBE_DIR):
            h.load_file("stdrun.hoc")
            h.load_file("template/ObliquePath.hoc")
            h.load_file("template/BasalPath.hoc")

            h("strdef morphology_location, morpho_path")
            h("strdef ObliqueTrunkSection, BasalTrunkSection")
            h("objref vRP, vAPEX")
            h(
                """
                proc xopen_morphology(){
                    sprint(morpho_path,"%s/%s",morphology_location,$s1)
                    xopen(morpho_path)
                }
                """
            )

            h.morphology_location = "pc2b"
            h.ObliqueTrunkSection = "trunk[17]"
            h.BasalTrunkSection = "trunk[7]"

            h.xopen("pc2b/cell.hoc")
            h.xopen("pc2b/cell-analysis-simple.hoc")
            h.xopen("lib/TP-lib.hoc")
            h('Tip_sections(apical_non_trunk_list,apical_trunk_list,"Apical")')
            h("objref apical_tip_list")
            h("apical_tip_list=TP_list")
            h.xopen("lib/Oblique-lib.hoc")
            h("oblique_sections(apical_tip_list,apical_trunk_list,num_tips)")
            h.xopen("lib/vector-distance.hoc")
            h.xopen("cell_setup_pc2b_CCh_driven.hoc")

            if d_lambda is not None:
                h.d_lambda = float(d_lambda)
                h("forall { nseg = int((L/(d_lambda*lambda_f(freq))+0.9)/2)*2 + 5 }")

    return h.soma[0]


def run_neuron_step(
    protocol: StepProtocol,
    *,
    quiet: bool = True,
    d_lambda: float | None = None,
) -> dict[str, np.ndarray]:
    soma = build_combe_neuron_model(quiet=quiet, d_lambda=d_lambda)

    cvode = h.CVode()
    cvode.active(0)
    h.dt = protocol.dt
    h.tstop = protocol.tstop

    clamp = h.IClamp(soma(0.5))
    clamp.delay = protocol.delay
    clamp.dur = protocol.duration
    clamp.amp = protocol.amplitude

    time = h.Vector().record(h._ref_t)
    voltage = h.Vector().record(soma(0.5)._ref_v)
    current = h.Vector().record(clamp._ref_i)

    h.finitialize(protocol.v_init)
    h.fcurrent()
    h.continuerun(protocol.tstop)

    time_np = np.asarray(time, dtype=float)
    voltage_np = np.asarray(voltage, dtype=float)
    current_np = np.asarray(current, dtype=float)

    if current_np.size != time_np.size:
        current_np = current_at_time_points(time_np, current_np)

    return {
        "time": time_np,
        "voltage": voltage_np,
        "current": current_np,
    }

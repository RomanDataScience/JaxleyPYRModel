# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
from pathlib import Path
import re
import csv

import jax.numpy as jnp
import jaxley as jx
import jaxley.solver_gate as solver_gate
import numpy as np
from jaxley.channels import Leak
from jaxley.morphology import distance_direct


def _save_exp_compatible(x, max_value: float = 20.0):
    """Compatibility for Jaxley 0.13 with JAX versions whose clip uses max=."""
    return jnp.exp(jnp.clip(x, max=max_value))


solver_gate.save_exp = _save_exp_compatible

FLOAT_RE = re.compile(rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
MODEL_DIR = Path(__file__).resolve().parent
SEGMENTED_TRACES_DIR = MODEL_DIR.parent / "Experimental_currentClamp_Analysis" / "Segmented_Traces"
SEGMENT_ALIASES = {
    "depolarizing_pulse": "depolarizing_step",
    "hyperpolarizing_step": "hyperpolarizing_pulse",
}

from jaxley_models.l5pc.channels import (
    SKE2,
    CaHVA,
    CaLVA,
    CaNernstReversal,
    CaPump,
    H,
    KPst,
    KTst,
    M,
    NaTaT,
    NaTs2T,
    SKv3_1,
)

params = {}
params["apical"] = {
    "apical_NaTs2T_gNaTs2T": 0.026145,
    "apical_SKv3_1_gSKv3_1": 0.004226,
    "apical_M_gM": 0.000143,
}
params["soma"] = {
    "somatic_NaTs2T_gNaTs2T": 0.983955,
    "somatic_SKv3_1_gSKv3_1": 0.303472,
    "somatic_SKE2_gSKE2": 0.008407,
    "somatic_CaPump_gamma": 0.000609,
    "somatic_CaPump_decay": 210.485291,
    "somatic_CaHVA_gCaHVA": 0.000994,
    "somatic_CaLVA_gCaLVA": 0.000333,
}
params["axon"] = {
    "axonal_NaTaT_gNaTaT": 3.137968,
    "axonal_KPst_gKPst": 0.973538,
    "axonal_KTst_gKTst": 0.089259,
    "axonal_SKE2_gSKE2": 0.007104,
    "axonal_SKv3_1_gSKv3_1": 1.021945,
    "axonal_CaHVA_gCaHVA": 0.00099,
    "axonal_CaLVA_gCaLVA": 0.008752,
    "axonal_CaPump_gamma": 0.00291,
    "axonal_CaPump_decay": 287.19873,
}
bounds = {
    "apical_NaTs2T_gNaTs2T": [0, 0.04],
    "apical_SKv3_1_gSKv3_1": [0, 0.04],
    "apical_M_gM": [0, 0.001],
    "somatic_NaTs2T_gNaTs2T": [0.0, 1.0],
    "somatic_SKv3_1_gSKv3_1": [0.25, 1],
    "somatic_SKE2_gSKE2": [0, 0.1],
    "somatic_CaPump_gamma": [0.0005, 0.01],
    "somatic_CaPump_decay": [20, 1_000],
    "somatic_CaHVA_gCaHVA": [0, 0.001],
    "somatic_CaLVA_gCaLVA": [0, 0.01],
    "axonal_NaTaT_gNaTaT": [0.0, 4.0],
    "axonal_KPst_gKPst": [0.0, 1.0],
    "axonal_KTst_gKTst": [0.0, 0.1],
    "axonal_SKE2_gSKE2": [0.0, 0.1],
    "axonal_SKv3_1_gSKv3_1": [0.0, 2.0],
    "axonal_CaHVA_gCaHVA": [0, 0.001],
    "axonal_CaLVA_gCaLVA": [0, 0.01],
    "axonal_CaPump_gamma": [0.0005, 0.05],
    "axonal_CaPump_decay": [20, 1_000],
}

def update_number_compartments(cell):
    # Reasonable default values for most models.
    frequency = 100.0
    d_lambda = 0.1  # Larger -> more coarse-grained.

    for branch in cell.branches:
        diameter = 2 * branch.nodes["radius"].to_numpy()[0]
        c_m = branch.nodes["capacitance"].to_numpy()[0]
        r_a = branch.nodes["axial_resistivity"].to_numpy()[0]
        l = branch.nodes["length"].to_numpy()[0]

        lambda_f = 1e5 * np.sqrt(diameter / (4 * np.pi * frequency * c_m * r_a))
        ncomp = int((l / (d_lambda * lambda_f) + 0.9) / 2) * 2 + 1
        branch.set_ncomp(ncomp, initialize=False)
    
    # After the loop, you have to run `cell.initialize()` because we passed
    # `set_ncomp(..., initialize=False)` for speeding up the loop over branches.
    return cell



def L5PC():
    base_path = os.path.dirname(__file__)
    cell = jx.read_swc(
        os.path.join(base_path, "CELL.SWC"),
        ncomp=1,
        assign_groups=True
    )

    cell = update_number_compartments(cell)
    cell.initialize()
    ########## APICAL ##########
    cell.apical.set("capacitance", 2.0)
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))

    # The H-conductance depends on the distance from the soma.
    cell.compute_compartment_centers()
    direct_dists = distance_direct(cell.soma.branch(0).comp(0), cell)
    cell.nodes["dist_from_soma"] = direct_dists
    gH_conductance = (-0.8696 + 2.087 * np.exp(cell.apical.nodes["dist_from_soma"] * 0.0031)) * 8e-5
    cell.apical.set("apical_H_gH", gH_conductance)

    ########## SOMA ##########
    cell.soma.insert(NaTs2T().change_name("somatic_NaTs2T"))
    cell.soma.insert(SKv3_1().change_name("somatic_SKv3_1"))
    cell.soma.insert(SKE2().change_name("somatic_SKE2"))
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.insert(CaPump().change_name("somatic_CaPump"))
    cell.soma.insert(CaHVA().change_name("somatic_CaHVA"))
    cell.soma.insert(CaLVA().change_name("somatic_CaLVA"))
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)

    ########## BASAL ##########
    cell.basal.insert(H().change_name("basal_H"))
    cell.basal.set("basal_H_gH", 8e-5)

    # ########## AXON ##########
    cell.insert(CaNernstReversal())
    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)

    cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    cell.axon.insert(KTst().change_name("axonal_KTst"))
    cell.axon.insert(CaPump().change_name("axonal_CaPump"))
    cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    cell.axon.insert(KPst().change_name("axonal_KPst"))
    cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))
    cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))

    ########## WHOLE CELL  ##########
    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)

    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)
    cell.set("v", -65.0)

    for group in ["apical", "soma", "axon"]:
        group_params = params[group]
        for key, value in group_params.items():
            cell.select(cell.nodes[cell.nodes[group]].index).set(key, value)

    return cell

def add_test_stimuli(cell, dt = 0.025, t_max = 100.0):
    time_vec = jnp.arange(0, t_max+2*dt, dt)

    cell.delete_stimuli()
    cell.delete_recordings()

    i_delay = 5.0  # ms
    i_dur = 90.0  # ms
    i_amp = 1.8  # nA
    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    cell.soma.branch(0).loc(0.5).stimulate(current)
    cell.soma.branch(0).loc(0.5).record()

    cell.set("v", -72.0)
    cell.init_states()
    
    return cell, time_vec

def _load_numeric_trace(path):
    values = [float(match) for match in FLOAT_RE.findall(Path(path).read_bytes())]
    if not values:
        raise ValueError(f"No numeric samples found in {path}")
    return np.asarray(values, dtype=float)

def segmented_current_path(
    *,
    cell_name,
    trace_name,
    segment_name="depolarizing_step",
    segmented_dir=SEGMENTED_TRACES_DIR,
):
    segment_name = SEGMENT_ALIASES.get(segment_name, segment_name)
    metadata_path = Path(segmented_dir) / "segment_metadata.csv"
    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["cell"] == cell_name
                    and row["trace"] == trace_name
                    and row["segment"] == segment_name
                ):
                    current_path = Path(row["current_output"])
                    if current_path.exists():
                        return current_path
                    break

    current_path = (
        Path(segmented_dir)
        / str(cell_name)
        / str(segment_name)
        / f"{trace_name}_{segment_name}_i.txt"
    )
    if not current_path.exists():
        raise FileNotFoundError(
            f"No segmented current trace found for {cell_name}/{trace_name}/{segment_name}: "
            f"{current_path}"
        )
    return current_path

def add_segmented_stimuli(
    cell,
    *,
    cell_name,
    trace_name,
    segment_name="depolarizing_step",
    segmented_dir=SEGMENTED_TRACES_DIR,
    experimental_dt=0.05,
    delta_t=0.025,
    current_scale_to_nA=1e-3,
    v_init=-72.0,
):
    """Attach a segmented experimental current trace as the soma stimulus.

    The segmented current file is resolved from `Segmented_Traces/segment_metadata.csv`
    using `cell_name`, `trace_name`, and `segment_name`.

    ⚠️ IMPORTANT!
    If you change `jx.integrate(..., delta_t=0.025)`, pass the same `delta_t`
    here. The n-th entry of the stimulus is applied at the n-th simulation step,
    regardless of dt.
    """
    if experimental_dt <= 0.0:
        raise ValueError("experimental_dt must be positive")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive")

    current_path = segmented_current_path(
        cell_name=cell_name,
        trace_name=trace_name,
        segment_name=segment_name,
        segmented_dir=segmented_dir,
    )
    stimulus = jnp.asarray(_load_numeric_trace(current_path), dtype=jnp.float64)
    stimulus = stimulus * float(current_scale_to_nA)

    target_steps = max(1, int(round(stimulus.size * float(experimental_dt) / float(delta_t))))
    if target_steps != stimulus.size:
        source_indices = jnp.floor(
            jnp.arange(target_steps, dtype=jnp.float64) * float(delta_t) / float(experimental_dt)
        ).astype(jnp.int32)
        source_indices = jnp.minimum(source_indices, stimulus.size - 1)
        stimulus = stimulus[source_indices]

    time_vec = jnp.arange(stimulus.size + 1, dtype=jnp.float64) * float(delta_t)

    cell.delete_stimuli()
    cell.delete_recordings()
    cell.soma.branch(0).loc(0.5).stimulate(stimulus)
    cell.soma.branch(0).loc(0.5).record()

    cell.set("v", v_init)
    cell.init_states()

    return cell, time_vec, stimulus

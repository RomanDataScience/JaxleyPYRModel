from __future__ import annotations

import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from jax import config

# Preserve the historical float64 default for direct imports while allowing the
# refactored runtime bootstrap to select precision and CPU/GPU before import.
if "JAX_ENABLE_X64" not in os.environ:
    config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxley as jx
import jaxley.solver_gate as solver_gate
import numpy as np
import pandas as pd
from jaxley.channels import Leak
from jaxley.morphology import distance_pathwise
from jaxley.utils.morph_attributes import morph_attrs_from_xyzr, split_xyzr_into_equal_length_segments


warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channels_converted.channels_jaxley import (  # noqa: E402
    Cal,
    Cal4,
    CalH,
    Car,
    Cat,
    D3,
    H,
    Icand,
    Kad,
    Kap,
    Kca,
    Kd,
    Kir,
    Km,
    Kv2like,
    MyKca,
    Na3Dend,
    Nap,
    Nav16A,
    Nax,
    enable_cal4_diffusion,
)


MODEL_DIR = Path(__file__).resolve().parent
SEGMENTED_TRACES_DIR = MODEL_DIR.parent / "Experimental_currentClamp_Analysis" / "Segmented_Traces"
SEGMENT_ALIASES = {
    "depolarizing_pulse": "depolarizing_step",
    "hyperpolarizing_step": "hyperpolarizing_pulse",
}


def _save_exp_compatible(x, max_value: float = 20.0):
    return jnp.exp(jnp.clip(x, max=max_value))


solver_gate.save_exp = _save_exp_compatible


@dataclass(frozen=True)
class CombeParameters:
    RmSoma: float = 149999.0
    RaSoma: float = 42.562
    RmTuft: float = 45373.4
    RaTuft: float = 35.0
    DistHalfRm: float = 151.741
    DistHalfRa: float = 90.8296
    SlopeRm: float = 13.8656
    SlopeRa: float = 7.76766
    soma_hbar: float = 0.00003
    KirGbar: float = 0.00020307 * 5.0
    Epas: float = -71.9879
    CmSoma: float = 1.0
    SpineFactorBasal: float = 3.5
    SpineFactorTuft: float = 3.5
    soma_caL: float = 0.00006
    soma_car: float = 0.00003
    gsomacar: float = 0.00008
    soma_caLH: float = 0.0
    soma_caT: float = 0.0003
    soma_km: float = 0.0
    mykca_init: float = 0.0
    soma_kca: float = 0.0
    AXNa: float = 3.5
    gkdrsoma: float = 0.0
    gkdrdend: float = 0.0
    psoma: float = 0.00075
    slowsoma: float = 0.15
    slownotsoma: float = 0.1
    sinfsoma: float = 1.35
    soma_kap: float = 7.0 * 0.0005 * 4.0 * 2.75
    axon_kap: float = 7.0 * 0.0005 * 4.0 * 4.0
    basal_kap: float = 0.0025036
    soma_kad: float = 7.0 * 0.0005 * 4.0 * 2.75
    gna: float = 0.035
    axongkdr: float = 0.011
    gnadend: float = 0.015 * 1.5
    gkdrapical: float = 0.01 * 0.05
    gkv2soma: float = 0.00264 * 5.0
    gkv2: float = 0.00198 * 10.0
    gkv2axon: float = 0.00198 * 10.0
    gkv2scale: float = 0.3
    scale_Na_conduct: float = 14.0
    distalv: float = 0.0
    proximalv: float = 6.0
    icangbar: float = 0.06 * 0.75
    icand_can: float = 0.0
    nap_gnabar: float = 0.0
    gip3: float = 1.85
    kd_deactivation_tau_scale: float = 1.0
    nat_fast_inactivation_tau_scale: float = 1.0
    nat_slow_recovery_tau_scale: float = 1.0
    h_tau_scale: float = 1.0


COMBE_PARAMS = CombeParameters()

CONDUCTANCE_PARAMETER_KEYS = (
    "soma_hbar",
    "KirGbar",
    "soma_caL",
    "soma_car",
    "gsomacar",
    "soma_caLH",
    "soma_caT",
    "soma_km",
    "mykca_init",
    "soma_kca",
    "AXNa",
    "gkdrsoma",
    "gkdrdend",
    "soma_kap",
    "axon_kap",
    "basal_kap",
    "soma_kad",
    "gna",
    "axongkdr",
    "gnadend",
    "gkdrapical",
    "gkv2soma",
    "gkv2",
    "gkv2axon",
    "gkv2scale",
    "scale_Na_conduct",
    "icangbar",
    "nap_gnabar",
)
PASSIVE_PARAMETER_KEYS = (
    "RmSoma",
    "RaSoma",
    "RmTuft",
    "RaTuft",
    "DistHalfRm",
    "DistHalfRa",
    "SlopeRm",
    "SlopeRa",
    "Epas",
    "CmSoma",
    "SpineFactorBasal",
    "SpineFactorTuft",
)
KINETIC_PARAMETER_KEYS = (
    "kd_deactivation_tau_scale",
    "nat_fast_inactivation_tau_scale",
    "nat_slow_recovery_tau_scale",
    "h_tau_scale",
)

params = {
    "conductances": {key: getattr(COMBE_PARAMS, key) for key in CONDUCTANCE_PARAMETER_KEYS},
    "passive": {key: getattr(COMBE_PARAMS, key) for key in PASSIVE_PARAMETER_KEYS},
    "kinetics": {key: getattr(COMBE_PARAMS, key) for key in KINETIC_PARAMETER_KEYS},
}

bounds = {
    "RmSoma": [50_000.0, 300_000.0],
    "RaSoma": [20.0, 150.0],
    "RmTuft": [10_000.0, 150_000.0],
    "RaTuft": [20.0, 150.0],
    "DistHalfRm": [20.0, 500.0],
    "DistHalfRa": [20.0, 300.0],
    "SlopeRm": [1.0, 80.0],
    "SlopeRa": [1.0, 80.0],
    "Epas": [-90.0, -50.0],
    "CmSoma": [0.3, 5.0],
    "SpineFactorBasal": [1.0, 6.0],
    "SpineFactorTuft": [1.0, 6.0],
    "soma_hbar": [0.0, 0.0003],
    "KirGbar": [0.0, 0.005],
    "soma_caL": [0.0, 0.0006],
    "soma_car": [0.0, 0.0003],
    "gsomacar": [0.0, 0.0008],
    "soma_caLH": [0.0, 0.001],
    "soma_caT": [0.0, 0.003],
    "soma_km": [0.0, 0.01],
    "mykca_init": [0.0, 0.01],
    "soma_kca": [0.0, 0.01],
    "AXNa": [0.1, 10.0],
    "gkdrsoma": [0.0, 0.02],
    "gkdrdend": [0.0, 0.02],
    "soma_kap": [0.0, 0.2],
    "axon_kap": [0.0, 0.2],
    "basal_kap": [0.0, 0.05],
    "soma_kad": [0.0, 0.2],
    "gna": [0.0, 0.1],
    "axongkdr": [0.0, 0.05],
    "gnadend": [0.0, 0.1],
    "gkdrapical": [0.0, 0.01],
    "gkv2soma": [0.0, 0.1],
    "gkv2": [0.0, 0.1],
    "gkv2axon": [0.0, 0.1],
    "gkv2scale": [0.0, 2.0],
    "scale_Na_conduct": [1.0, 30.0],
    "icangbar": [0.0, 0.2],
    "nap_gnabar": [0.0, 0.001],
    "kd_deactivation_tau_scale": [0.25, 4.0],
    "nat_fast_inactivation_tau_scale": [0.5, 2.0],
    "nat_slow_recovery_tau_scale": [0.5, 2.0],
    "h_tau_scale": [0.5, 2.0],
}


def morphology_path() -> Path:
    for name in ("CELL.SWC", "Cell.SWC"):
        candidate = MODEL_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No CELL.SWC found in {MODEL_DIR}")


def _hoc_group_name(sec_name: str) -> str:
    base = sec_name.split("[", maxsplit=1)[0]
    if base == "soma":
        return "soma"
    if base == "axon":
        return "axon"
    if base == "dend":
        return "basal"
    return "apical"


def _section_xyzr(sec, h) -> np.ndarray:
    return np.asarray(
        [
            [
                float(h.x3d(i, sec=sec)),
                float(h.y3d(i, sec=sec)),
                float(h.z3d(i, sec=sec)),
                float(h.diam3d(i, sec=sec)) / 2.0,
            ]
            for i in range(int(h.n3d(sec=sec)))
        ],
        dtype=float,
    )


def _ordered_hoc_sections(root_sec) -> tuple[list, list[int]]:
    sections = []
    parents = []

    def visit(sec, parent_index: int) -> None:
        section_index = len(sections)
        sections.append(sec)
        parents.append(parent_index)
        for child in sec.children():
            visit(child, section_index)

    visit(root_sec, -1)
    return sections, parents


HOC_CHANNEL_CLASSES = {
    "d3": D3,
    "cal4": Cal4,
    "icand": Icand,
    "na16a": Nav16A,
    "kd": Kd,
    "Kv2like": Kv2like,
    "nap": Nap,
    "h": H,
    "kap": Kap,
    "kad": Kad,
    "km": Km,
    "cal": Cal,
    "cat": Cat,
    "car": Car,
    "calH": CalH,
    "kca": Kca,
    "mykca": MyKca,
    "kir": Kir,
    "nax": Nax,
    "na3dend": Na3Dend,
}

HOC_ION_PARAM_MAP = {
    "eNa": "ena",
    "eK": "ek",
    "eCa": "eca",
}


def _hoc_channel_param_map() -> dict[str, tuple[str, str]]:
    mapping = {}
    for hoc_mech, channel_cls in HOC_CHANNEL_CLASSES.items():
        prefix = f"{hoc_mech}_"
        for target in channel_cls(hoc_mech).channel_params:
            if target in HOC_ION_PARAM_MAP or target == "celsius":
                continue
            if target.startswith(prefix):
                mapping[target] = (hoc_mech, target.removeprefix(prefix))
    return mapping


HOC_CHANNEL_PARAM_MAP = _hoc_channel_param_map()


def _hoc_mechanism_value(seg, hoc_mech: str, param_name: str, h) -> float:
    try:
        if hasattr(seg, hoc_mech):
            return float(getattr(getattr(seg, hoc_mech), param_name))
    except Exception:
        pass

    try:
        return float(getattr(h, f"{param_name}_{hoc_mech}"))
    except Exception:
        return np.nan


def _hoc_segment_value(seg, attr_name: str) -> float:
    try:
        return float(getattr(seg, attr_name))
    except Exception:
        return np.nan


def build_hoc_section_cell(d_lambda: float = 0.3):
    """Build a Jaxley morphology with one branch per Combe HOC section.

    A standard SWC cannot represent NEURON's zero-length logical section
    connections when child pt3d coordinates do not coincide with parent endpoints.
    This builder uses the HOC topology directly, then copies final per-segment
    geometry, passive properties, and HOC mechanism parameters.
    """
    from channels_converted.modelComparison.neuron_model import build_combe_neuron_model
    from neuron import h

    soma = build_combe_neuron_model(quiet=True, d_lambda=d_lambda)
    sections, parents = _ordered_hoc_sections(soma)

    branches = []
    xyzr = []
    section_attrs = []
    section_groups = []

    for sec in sections:
        nseg = int(sec.nseg)
        branches.append(jx.Branch(ncomp=nseg))

        sec_xyzr = _section_xyzr(sec, h)
        xyzr.append(sec_xyzr)

        comp_xyzr = split_xyzr_into_equal_length_segments(sec_xyzr, nseg)
        morph = np.asarray(
            [morph_attrs_from_xyzr(comp, 0.0, nseg) for comp in comp_xyzr],
            dtype=float,
        )
        section_attrs.append(morph)
        section_groups.append(_hoc_group_name(sec.name()))

    cell = jx.Cell(branches, parents=parents, xyzr=xyzr)
    group_indices = {group: [] for group in ("soma", "axon", "basal", "apical")}

    start = 0
    for section_index, (sec, morph, group) in enumerate(
        zip(sections, section_attrs, section_groups)
    ):
        nseg = int(sec.nseg)
        indices = np.arange(start, start + nseg)
        start += nseg

        cell.nodes.loc[indices, "radius"] = morph[:, 0]
        cell.nodes.loc[indices, "area"] = morph[:, 1]
        cell.nodes.loc[indices, "volume"] = morph[:, 2]
        cell.nodes.loc[indices, "resistive_load_in"] = morph[:, 3]
        cell.nodes.loc[indices, "resistive_load_out"] = morph[:, 4]
        cell.nodes.loc[indices, "length"] = float(sec.L) / nseg
        cell.nodes.loc[indices, "axial_resistivity"] = float(sec.Ra)
        cell.nodes.loc[indices, "capacitance"] = [float(seg.cm) for seg in sec]
        cell.nodes.loc[indices, "hoc_g_pas"] = [float(seg.pas.g) for seg in sec]
        cell.nodes.loc[indices, "hoc_e_pas"] = [float(seg.pas.e) for seg in sec]
        cell.nodes.loc[indices, "hoc_section_index"] = section_index
        # Most Combe distributions are assigned while the HOC section still has
        # nseg=1. HOC's final write is therefore evaluated at x=1 and remains
        # sectionwise constant after later nseg changes. Keep that exact coordinate
        # rather than reconstructing it from Jaxley's center-to-center distances.
        cell.nodes.loc[indices, "hoc_assignment_distance_um"] = float(
            h.distance(1.0, sec=sec)
        )
        cell.nodes.loc[indices, "hoc_celsius"] = float(h.celsius)
        for target, attr_name in HOC_ION_PARAM_MAP.items():
            cell.nodes.loc[indices, f"hoc_{target}"] = [
                _hoc_segment_value(seg, attr_name) for seg in sec
            ]
        for target, (hoc_mech, param_name) in HOC_CHANNEL_PARAM_MAP.items():
            cell.nodes.loc[indices, f"hoc_{target}"] = [
                _hoc_mechanism_value(seg, hoc_mech, param_name, h) for seg in sec
            ]
        group_indices[group].extend(indices.tolist())

    cell.initialize()
    for group, indices in group_indices.items():
        cell.select(np.asarray(indices, dtype=int)).add_to_group(group)
    return cell


def apply_hoc_passive_profile(cell):
    if "hoc_g_pas" not in cell.nodes.columns:
        return cell

    cell.set("capacitance", cell.nodes["capacitance"].to_numpy(dtype=float))
    cell.set("axial_resistivity", cell.nodes["axial_resistivity"].to_numpy(dtype=float))
    if "Leak_gLeak" in cell.nodes.columns:
        cell.set("Leak_gLeak", cell.nodes["hoc_g_pas"].to_numpy(dtype=float))
        cell.set("Leak_eLeak", cell.nodes["hoc_e_pas"].to_numpy(dtype=float))
    return cell


def apply_hoc_channel_profile(cell):
    targets = [*HOC_CHANNEL_PARAM_MAP, *HOC_ION_PARAM_MAP, "celsius"]
    for target in targets:
        hoc_col = f"hoc_{target}"
        if target not in cell.nodes.columns or hoc_col not in cell.nodes.columns:
            continue

        values = cell.nodes[hoc_col].to_numpy(dtype=float)
        current = cell.nodes[target].to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(current)
        if not np.any(mask):
            continue

        indices = cell.nodes.index[mask].to_numpy()
        cell.select(indices).set(target, values[mask])
    return cell


def my_exp(x):
    return np.where(x < -50.0, 0.0, np.exp(np.minimum(x, 50.0)))


def passive_sigmoid(distance, soma_value, tuft_value, half_distance, slope):
    return tuft_value + (soma_value - tuft_value) / (
        1.0 + my_exp((distance - half_distance) / slope)
    )


def update_number_compartments(cell, d_lambda=0.1, frequency=100.0):
    """Apply the d-lambda rule at ``frequency`` Hz (100 Hz by default)."""

    for branch in cell.branches:
        diameter = 2.0 * branch.nodes["radius"].to_numpy(dtype=float)[0]
        capacitance = branch.nodes["capacitance"].to_numpy(dtype=float)[0]
        axial_resistivity = branch.nodes["axial_resistivity"].to_numpy(dtype=float)[0]
        length = branch.nodes["length"].to_numpy(dtype=float)[0]

        lambda_f = 1e5 * np.sqrt(
            diameter / (4.0 * np.pi * frequency * capacitance * axial_resistivity)
        )
        ncomp = int((length / (d_lambda * lambda_f) + 0.9) / 2.0) * 2 + 5
        branch.set_ncomp(max(1, ncomp), initialize=False)

    return cell


def set_distances_from_soma(cell):
    cell.compute_compartment_centers()
    root = cell.soma.branch(0).comp(0)
    root_offset = float(root.nodes["length"].to_numpy(dtype=float)[0]) / 2.0
    distances = distance_pathwise(root, cell)
    cell.nodes["dist_from_soma"] = np.asarray(distances, dtype=float) + root_offset
    return cell


def group_mask(cell, group: str) -> np.ndarray:
    return cell.nodes[group].to_numpy(dtype=bool)


def set_on(cell, mask: np.ndarray, key: str, value):
    indices = cell.nodes.index[mask].to_numpy()
    if np.isscalar(value):
        cell.select(indices).set(key, float(value))
    else:
        cell.select(indices).set(key, np.asarray(value, dtype=float)[mask])


def insert_combe_channels(cell):
    cell.insert(D3("d3"))
    cell.insert(Leak("Leak"))
    cell.insert(Cal4("cal4"))

    for group in (cell.soma, cell.apical):
        group.insert(Icand("icand"))
        group.insert(Nav16A("na16a"))
        group.insert(Kd("kd"))
        group.insert(Kv2like("Kv2like"))
        group.insert(H("h"))
        group.insert(Kap("kap"))
        group.insert(Km("km"))
        group.insert(Kca("kca"))
        group.insert(MyKca("mykca"))

    cell.soma.insert(Nap("nap"))
    cell.soma.insert(Cal("cal"))
    cell.soma.insert(Cat("cat"))
    cell.soma.insert(Car("car"))

    cell.apical.insert(Car("car"))
    cell.apical.insert(CalH("calH"))
    cell.apical.insert(Cat("cat"))
    cell.apical.insert(Kad("kad"))
    cell.apical.insert(Kir("kir"))

    cell.axon.insert(Nax("nax"))
    cell.axon.insert(Kd("kd"))
    cell.axon.insert(Km("km"))
    cell.axon.insert(Kap("kap"))
    cell.axon.insert(Kv2like("Kv2like"))

    cell.basal.insert(Na3Dend("na3dend"))
    cell.basal.insert(Nap("nap"))
    cell.basal.insert(Kap("kap"))
    cell.basal.insert(H("h"))
    cell.basal.insert(Kd("kd"))
    cell.basal.insert(Kv2like("Kv2like"))
    cell.basal.insert(Kir("kir"))

    return cell


def set_passive_properties(cell, p: CombeParameters = COMBE_PARAMS):
    if p == COMBE_PARAMS and "hoc_g_pas" in cell.nodes.columns:
        return apply_hoc_passive_profile(cell)

    n = len(cell.nodes)
    dist = cell.nodes["dist_from_soma"].to_numpy(dtype=float)
    soma = group_mask(cell, "soma")
    axon = group_mask(cell, "axon")
    basal = group_mask(cell, "basal")
    apical = group_mask(cell, "apical")

    capacitance = np.full(n, p.CmSoma, dtype=float)
    g_leak = np.full(n, 1.0 / passive_sigmoid(0.0, p.RmSoma, p.RmTuft, p.DistHalfRm, p.SlopeRm))
    axial = np.full(n, passive_sigmoid(0.0, p.RaSoma, p.RaTuft, p.DistHalfRa, p.SlopeRa))

    apical_dist = np.minimum(dist, 394.0)
    apical_spines = np.where(
        dist <= 100.0,
        1.0,
        np.where(
            dist > 394.0,
            p.SpineFactorTuft,
            2.0 + (dist - 100.0) * (p.SpineFactorTuft - 2.0) / 294.0,
        ),
    )
    apical_rm = passive_sigmoid(apical_dist, p.RmSoma, p.RmTuft, p.DistHalfRm, p.SlopeRm)
    apical_ra = passive_sigmoid(apical_dist, p.RaSoma, p.RaTuft, p.DistHalfRa, p.SlopeRa)
    capacitance[apical] = apical_spines[apical] * p.CmSoma
    g_leak[apical] = apical_spines[apical] / apical_rm[apical]
    axial[apical] = apical_ra[apical]

    basal_spines = np.where(dist <= 40.0, 1.0, p.SpineFactorBasal)
    capacitance[basal] = basal_spines[basal] * p.CmSoma
    g_leak[basal] = basal_spines[basal] / passive_sigmoid(
        0.0, p.RmSoma, p.RmTuft, p.DistHalfRm, p.SlopeRm
    )
    axial[basal] = passive_sigmoid(0.0, p.RaSoma, p.RaTuft, p.DistHalfRa, p.SlopeRa)

    capacitance[soma | axon] = p.CmSoma
    g_leak[soma | axon] = 1.0 / passive_sigmoid(
        0.0, p.RmSoma, p.RmTuft, p.DistHalfRm, p.SlopeRm
    )
    axial[soma | axon] = passive_sigmoid(0.0, p.RaSoma, p.RaTuft, p.DistHalfRa, p.SlopeRa)

    cell.set("capacitance", capacitance)
    cell.set("axial_resistivity", axial)
    if "Leak_gLeak" in cell.nodes.columns:
        cell.set("Leak_gLeak", g_leak)
        cell.set("Leak_eLeak", p.Epas)
    return cell


def set_common_reversals(cell):
    for key, value in {
        "eNa": 50.0,
        "eK": -80.0,
        "eCa": 140.0,
        "CaCon_i": 50e-6,
        "CaCon_e": 2.0,
        "celsius": 34.0,
    }.items():
        if key in cell.nodes.columns:
            cell.set(key, value)
    return cell


def set_cal4_profile(cell, p: CombeParameters = COMBE_PARAMS):
    dist = cell.nodes["dist_from_soma"].to_numpy(dtype=float)
    alpha = p.gip3 * (0.75 + 0.25 * my_exp(-dist / 100.0))
    cell.set("cal4_ip3i", 0.16e-3)
    cell.set("cal4_alpha", alpha)
    return cell


def set_soma_channels(cell, p: CombeParameters = COMBE_PARAMS):
    soma = group_mask(cell, "soma")
    set_on(cell, soma, "icand_gbar", p.icangbar)
    set_on(cell, soma, "icand_can", p.icand_can)
    set_on(cell, soma, "na16a_gbar", p.gna * p.scale_Na_conduct)
    set_on(cell, soma, "na16a_dist", p.sinfsoma)
    set_on(cell, soma, "na16a_persist", p.psoma)
    set_on(cell, soma, "na16a_slowdown", p.slowsoma)
    set_on(cell, soma, "na16a_C1O1v2", p.proximalv)
    set_on(
        cell,
        soma,
        "na16a_fast_inactivation_tau_scale",
        p.nat_fast_inactivation_tau_scale,
    )
    set_on(
        cell,
        soma,
        "na16a_slow_recovery_tau_scale",
        p.nat_slow_recovery_tau_scale,
    )
    set_on(cell, soma, "kd_gbar", p.gkdrsoma)
    set_on(
        cell,
        soma,
        "kd_deactivation_tau_scale",
        p.kd_deactivation_tau_scale,
    )
    set_on(cell, soma, "Kv2like_gbar", p.gkv2soma)
    set_on(cell, soma, "nap_gnabar", p.nap_gnabar)
    set_on(cell, soma, "nap_K", 4.5)
    set_on(cell, soma, "nap_vhalf", -60.4)
    set_on(cell, soma, "h_gbar", p.soma_hbar)
    set_on(cell, soma, "h_K", 8.8)
    set_on(cell, soma, "h_vhalf", -82.0)
    set_on(cell, soma, "h_tau_scale", p.h_tau_scale)
    set_on(cell, soma, "kap_gkabar", p.soma_kap)
    set_on(cell, soma, "km_gbar", p.soma_km)
    set_on(cell, soma, "cal_gcalbar", 0.1 * p.soma_caL)
    set_on(cell, soma, "cat_gcatbar", p.soma_caT)
    set_on(cell, soma, "car_gcabar", p.gsomacar)
    set_on(cell, soma, "kca_cac", 0.00075)
    set_on(cell, soma, "kca_gbar", 0.5 * p.soma_kca)
    set_on(cell, soma, "mykca_gkbar", 5.5 * p.mykca_init)
    return cell


def apical_na16a_dist(distance):
    y = 30.0 + 45.0 * (1.0 - my_exp(-distance / 126.0))
    return np.select(
        [y <= 44.6, y <= 58.2, y <= 65.79],
        [(y - 11.0) / 14.0, (y - 27.0) / 7.3, (y - 44.0) / 3.3],
        default=(y - 44.0) / 3.3,
    )


def apical_na16a_c1o1v2(distance, p: CombeParameters = COMBE_PARAMS):
    y = 30.0 + 45.0 * (1.0 - my_exp(-distance / 126.0))
    proximal = p.proximalv - p.proximalv * distance / 200.0
    return np.where(y > 65.79, p.distalv, proximal)


def set_apical_channels(cell, p: CombeParameters = COMBE_PARAMS):
    apical = group_mask(cell, "apical")
    dist = cell.nodes["dist_from_soma"].to_numpy(dtype=float)
    capped_h_dist = np.minimum(dist, 500.0)
    ndist = np.minimum(capped_h_dist, 300.0)

    set_on(cell, apical, "icand_gbar", p.icangbar)
    set_on(cell, apical, "icand_can", p.icand_can)
    set_on(cell, apical, "car_gcabar", np.full_like(dist, 0.1 * p.soma_car))
    set_on(cell, apical, "calH_gcalbar", np.where(dist > 50.0, 2.0 * p.soma_caLH, 0.1 * p.soma_caLH))
    set_on(cell, apical, "cat_gcatbar", np.where(dist < 100.0, 0.0, 4.0 * p.soma_caT * dist / 350.0))
    set_on(cell, apical, "kca_cac", 0.00075)
    set_on(cell, apical, "kca_gbar", np.where((dist < 200.0) & (dist > 50.0), 5.0 * p.soma_kca, 0.5 * p.soma_kca))
    set_on(cell, apical, "mykca_gkbar", np.where((dist < 200.0) & (dist > 50.0), 2.0 * p.mykca_init, 0.5 * p.mykca_init))
    set_on(cell, apical, "h_gbar", p.soma_hbar * (1.0 + (6.0 / 5.0) * capped_h_dist / 100.0))
    set_on(cell, apical, "h_vhalf", np.where(capped_h_dist > 100.0, -81.0 - 8.0 * (ndist - 100.0) / 200.0, -81.0))
    set_on(cell, apical, "h_tau_scale", p.h_tau_scale)
    set_on(cell, apical, "kap_gkabar", np.where(capped_h_dist > 100.0, 0.0, p.soma_kap * (1.0 + capped_h_dist / 100.0)))
    set_on(cell, apical, "kad_gkabar", np.where(capped_h_dist > 100.0, p.soma_kad * (1.0 + capped_h_dist / 100.0), 0.0))
    set_on(cell, apical, "Kv2like_gbar", np.where(capped_h_dist > 100.0, p.gkv2 * p.gkv2scale, p.gkv2))
    set_on(cell, apical, "na16a_gbar", p.gnadend * p.scale_Na_conduct)
    set_on(cell, apical, "na16a_persist", p.psoma)
    set_on(cell, apical, "na16a_slowdown", p.slownotsoma)
    set_on(cell, apical, "na16a_dist", apical_na16a_dist(dist))
    set_on(cell, apical, "na16a_C1O1v2", apical_na16a_c1o1v2(dist, p))
    set_on(
        cell,
        apical,
        "na16a_fast_inactivation_tau_scale",
        p.nat_fast_inactivation_tau_scale,
    )
    set_on(
        cell,
        apical,
        "na16a_slow_recovery_tau_scale",
        p.nat_slow_recovery_tau_scale,
    )
    set_on(cell, apical, "kd_gbar", p.gkdrapical)
    set_on(
        cell,
        apical,
        "kd_deactivation_tau_scale",
        p.kd_deactivation_tau_scale,
    )
    set_on(cell, apical, "km_gbar", p.soma_km)
    set_on(cell, apical, "kir_gbar", np.where(dist > 100.0, p.KirGbar, p.KirGbar * dist / 100.0))
    set_on(cell, apical, "kir_ek", -95.0)
    return cell


def set_axon_channels(cell, p: CombeParameters = COMBE_PARAMS):
    axon = group_mask(cell, "axon")
    set_on(cell, axon, "nax_gbar", p.gna * p.AXNa)
    set_on(
        cell,
        axon,
        "nax_fast_inactivation_tau_scale",
        p.nat_fast_inactivation_tau_scale,
    )
    set_on(cell, axon, "kd_gbar", p.axongkdr)
    set_on(
        cell,
        axon,
        "kd_deactivation_tau_scale",
        p.kd_deactivation_tau_scale,
    )
    set_on(cell, axon, "km_gbar", 3.0 * p.soma_km)
    set_on(cell, axon, "kap_gkabar", p.axon_kap)
    set_on(cell, axon, "Kv2like_gbar", p.gkv2axon)
    return cell


def set_basal_channels(cell, p: CombeParameters = COMBE_PARAMS):
    basal = group_mask(cell, "basal")
    dist = cell.nodes["dist_from_soma"].to_numpy(dtype=float)
    set_on(cell, basal, "na3dend_gbar", p.gnadend)
    set_on(
        cell,
        basal,
        "na3dend_fast_inactivation_tau_scale",
        p.nat_fast_inactivation_tau_scale,
    )
    set_on(cell, basal, "nap_gnabar", p.nap_gnabar)
    set_on(cell, basal, "nap_K", 4.5)
    set_on(cell, basal, "nap_vhalf", -60.4)
    set_on(cell, basal, "kap_gkabar", p.basal_kap)
    set_on(cell, basal, "h_gbar", p.soma_hbar)
    set_on(cell, basal, "h_tau_scale", p.h_tau_scale)
    set_on(cell, basal, "kd_gbar", p.gkdrdend)
    set_on(
        cell,
        basal,
        "kd_deactivation_tau_scale",
        p.kd_deactivation_tau_scale,
    )
    set_on(cell, basal, "Kv2like_gbar", p.gkv2 * p.gkv2scale)
    set_on(cell, basal, "kir_gbar", np.where(dist > 40.0, p.KirGbar, p.KirGbar * dist / 40.0))
    set_on(cell, basal, "kir_ek", -95.0)
    return cell


def _my_exp_jax(x):
    return jnp.where(x < -50.0, 0.0, jnp.exp(jnp.minimum(x, 50.0)))


def _passive_sigmoid_jax(distance, soma_value, tuft_value, half_distance, slope):
    return tuft_value + (soma_value - tuft_value) / (
        1.0 + _my_exp_jax((distance - half_distance) / slope)
    )


EXACT_HOC_UPDATE_MODE = "exact_hoc_frozen_grid"
RULE_UPDATE_MODE = "rule_based_final_centers"
SUPPORTED_FIT_PARAMETER_KEYS = frozenset(
    (*CONDUCTANCE_PARAMETER_KEYS, *PASSIVE_PARAMETER_KEYS, *KINETIC_PARAMETER_KEYS)
)


def _fit_values(keys, values, reference_values=None):
    out = dict(reference_values or asdict(COMBE_PARAMS))
    for index, key in enumerate(keys):
        out[key] = values[index]
    return out


def _apical_na16a_dist_jax(distance):
    y = 30.0 + 45.0 * (1.0 - _my_exp_jax(-distance / 126.0))
    return jnp.select(
        [y <= 44.6, y <= 58.2, y <= 65.79],
        [(y - 11.0) / 14.0, (y - 27.0) / 7.3, (y - 44.0) / 3.3],
        default=(y - 44.0) / 3.3,
    )


def _apical_na16a_c1o1v2_jax(distance, p):
    y = 30.0 + 45.0 * (1.0 - _my_exp_jax(-distance / 126.0))
    proximal = p["proximalv"] - p["proximalv"] * distance / 200.0
    return jnp.where(y > 65.79, p["distalv"], proximal)


def _reference_parameters(cell):
    return {
        **asdict(COMBE_PARAMS),
        **dict(getattr(cell, "_combe_reference_parameters", {})),
    }


def _parameter_update_mode(cell):
    return getattr(cell, "_combe_parameter_update_mode", RULE_UPDATE_MODE)


def _profile_distances(view, update_mode):
    distance_key = (
        "hoc_assignment_distance_um"
        if update_mode == EXACT_HOC_UPDATE_MODE
        and "hoc_assignment_distance_um" in view.nodes.columns
        else "dist_from_soma"
    )
    return jnp.asarray(view.nodes[distance_key].to_numpy(dtype=float))


def _parameter_baseline(view, key):
    values = view.nodes[key]
    return jnp.asarray(values[values.notna()].to_numpy(dtype=float))


def _anchor_to_reference(view, key, fitted, reference):
    """Apply a fitted rule without discarding the imported HOC profile.

    The HOC assignments were evaluated once per section and copied when nseg
    changed. On the frozen fitting grid, the exact corresponding operation is
    therefore the imported baseline plus the full endpoint-rule delta. This
    works for zero defaults, coupled products, and nonlinear passive rules.
    """
    baseline = _parameter_baseline(view, key)
    fitted = jnp.broadcast_to(jnp.asarray(fitted), baseline.shape)
    reference = jnp.broadcast_to(jnp.asarray(reference), baseline.shape)
    return baseline + (fitted - reference)


def _fit_profile(view, key, value, *dependencies):
    return view, key, value, frozenset(dependencies)


def _write_selected_profiles(
    fitted_profiles,
    reference_profiles,
    selected_keys,
    update_mode,
    state,
):
    for fitted_spec, reference_spec in zip(fitted_profiles, reference_profiles):
        view, key, fitted, dependencies = fitted_spec
        _, reference_key, reference, reference_dependencies = reference_spec
        if key != reference_key or dependencies != reference_dependencies:
            raise RuntimeError("Mismatched Combe fitted/reference profile definitions.")
        if dependencies.isdisjoint(selected_keys):
            continue

        value = (
            _anchor_to_reference(view, key, fitted, reference)
            if update_mode == EXACT_HOC_UPDATE_MODE
            else fitted
        )
        state = view.data_set(key, value, state)
    return state


def _passive_fit_profiles(cell, p, update_mode):
    soma_rm = _passive_sigmoid_jax(
        0.0, p["RmSoma"], p["RmTuft"], p["DistHalfRm"], p["SlopeRm"]
    )
    soma_ra = _passive_sigmoid_jax(
        0.0, p["RaSoma"], p["RaTuft"], p["DistHalfRa"], p["SlopeRa"]
    )

    apical_distances = _profile_distances(cell.apical, update_mode)
    basal_distances = _profile_distances(cell.basal, update_mode)
    apical_dist = jnp.minimum(apical_distances, 394.0)
    apical_spines = jnp.where(
        apical_distances <= 100.0,
        1.0,
        jnp.where(
            apical_distances > 394.0,
            p["SpineFactorTuft"],
            2.0
            + (apical_distances - 100.0)
            * (p["SpineFactorTuft"] - 2.0)
            / 294.0,
        ),
    )
    apical_rm = _passive_sigmoid_jax(
        apical_dist, p["RmSoma"], p["RmTuft"], p["DistHalfRm"], p["SlopeRm"]
    )
    apical_ra = _passive_sigmoid_jax(
        apical_dist, p["RaSoma"], p["RaTuft"], p["DistHalfRa"], p["SlopeRa"]
    )
    basal_spines = jnp.where(
        basal_distances <= 40.0, 1.0, p["SpineFactorBasal"]
    )

    rm_dependencies = ("RmSoma", "RmTuft", "DistHalfRm", "SlopeRm")
    ra_dependencies = ("RaSoma", "RaTuft", "DistHalfRa", "SlopeRa")
    return (
        _fit_profile(cell.soma, "capacitance", p["CmSoma"], "CmSoma"),
        _fit_profile(cell.axon, "capacitance", p["CmSoma"], "CmSoma"),
        _fit_profile(
            cell.basal,
            "capacitance",
            basal_spines * p["CmSoma"],
            "CmSoma",
            "SpineFactorBasal",
        ),
        _fit_profile(
            cell.apical,
            "capacitance",
            apical_spines * p["CmSoma"],
            "CmSoma",
            "SpineFactorTuft",
        ),
        _fit_profile(
            cell.soma, "axial_resistivity", soma_ra, *ra_dependencies
        ),
        _fit_profile(
            cell.axon, "axial_resistivity", soma_ra, *ra_dependencies
        ),
        _fit_profile(
            cell.basal, "axial_resistivity", soma_ra, *ra_dependencies
        ),
        _fit_profile(
            cell.apical, "axial_resistivity", apical_ra, *ra_dependencies
        ),
        _fit_profile(cell.soma, "Leak_gLeak", 1.0 / soma_rm, *rm_dependencies),
        _fit_profile(cell.axon, "Leak_gLeak", 1.0 / soma_rm, *rm_dependencies),
        _fit_profile(
            cell.basal,
            "Leak_gLeak",
            basal_spines / soma_rm,
            *(rm_dependencies + ("SpineFactorBasal",)),
        ),
        _fit_profile(
            cell.apical,
            "Leak_gLeak",
            apical_spines / apical_rm,
            *(rm_dependencies + ("SpineFactorTuft",)),
        ),
        _fit_profile(cell, "Leak_eLeak", p["Epas"], "Epas"),
    )


def _conductance_fit_profiles(cell, p, update_mode):
    apical_dist = _profile_distances(cell.apical, update_mode)
    basal_dist = _profile_distances(cell.basal, update_mode)
    capped_h_dist = jnp.minimum(apical_dist, 500.0)

    return (
        _fit_profile(cell.soma, "icand_gbar", p["icangbar"], "icangbar"),
        _fit_profile(
            cell.soma,
            "na16a_gbar",
            p["gna"] * p["scale_Na_conduct"],
            "gna",
            "scale_Na_conduct",
        ),
        _fit_profile(cell.soma, "kd_gbar", p["gkdrsoma"], "gkdrsoma"),
        _fit_profile(
            cell.soma, "Kv2like_gbar", p["gkv2soma"], "gkv2soma"
        ),
        _fit_profile(cell.soma, "nap_gnabar", p["nap_gnabar"], "nap_gnabar"),
        _fit_profile(cell.soma, "h_gbar", p["soma_hbar"], "soma_hbar"),
        _fit_profile(cell.soma, "kap_gkabar", p["soma_kap"], "soma_kap"),
        _fit_profile(cell.soma, "km_gbar", p["soma_km"], "soma_km"),
        _fit_profile(
            cell.soma, "cal_gcalbar", 0.1 * p["soma_caL"], "soma_caL"
        ),
        _fit_profile(cell.soma, "cat_gcatbar", p["soma_caT"], "soma_caT"),
        _fit_profile(cell.soma, "car_gcabar", p["gsomacar"], "gsomacar"),
        _fit_profile(
            cell.soma, "kca_gbar", 0.5 * p["soma_kca"], "soma_kca"
        ),
        _fit_profile(
            cell.soma,
            "mykca_gkbar",
            5.5 * p["mykca_init"],
            "mykca_init",
        ),
        _fit_profile(cell.apical, "icand_gbar", p["icangbar"], "icangbar"),
        _fit_profile(
            cell.apical, "car_gcabar", 0.1 * p["soma_car"], "soma_car"
        ),
        _fit_profile(
            cell.apical,
            "calH_gcalbar",
            jnp.where(
                apical_dist > 50.0,
                2.0 * p["soma_caLH"],
                0.1 * p["soma_caLH"],
            ),
            "soma_caLH",
        ),
        _fit_profile(
            cell.apical,
            "cat_gcatbar",
            jnp.where(
                apical_dist < 100.0,
                0.0,
                4.0 * p["soma_caT"] * apical_dist / 350.0,
            ),
            "soma_caT",
        ),
        _fit_profile(
            cell.apical,
            "kca_gbar",
            jnp.where(
                (apical_dist < 200.0) & (apical_dist > 50.0),
                5.0 * p["soma_kca"],
                0.5 * p["soma_kca"],
            ),
            "soma_kca",
        ),
        _fit_profile(
            cell.apical,
            "mykca_gkbar",
            jnp.where(
                (apical_dist < 200.0) & (apical_dist > 50.0),
                2.0 * p["mykca_init"],
                0.5 * p["mykca_init"],
            ),
            "mykca_init",
        ),
        _fit_profile(
            cell.apical,
            "h_gbar",
            p["soma_hbar"] * (1.0 + (6.0 / 5.0) * capped_h_dist / 100.0),
            "soma_hbar",
        ),
        _fit_profile(
            cell.apical,
            "kap_gkabar",
            jnp.where(
                capped_h_dist > 100.0,
                0.0,
                p["soma_kap"] * (1.0 + capped_h_dist / 100.0),
            ),
            "soma_kap",
        ),
        _fit_profile(
            cell.apical,
            "kad_gkabar",
            jnp.where(
                capped_h_dist > 100.0,
                p["soma_kad"] * (1.0 + capped_h_dist / 100.0),
                0.0,
            ),
            "soma_kad",
        ),
        _fit_profile(
            cell.apical,
            "Kv2like_gbar",
            jnp.where(
                capped_h_dist > 100.0,
                p["gkv2"] * p["gkv2scale"],
                p["gkv2"],
            ),
            "gkv2",
            "gkv2scale",
        ),
        _fit_profile(
            cell.apical,
            "na16a_gbar",
            p["gnadend"] * p["scale_Na_conduct"],
            "gnadend",
            "scale_Na_conduct",
        ),
        _fit_profile(cell.apical, "kd_gbar", p["gkdrapical"], "gkdrapical"),
        _fit_profile(cell.apical, "km_gbar", p["soma_km"], "soma_km"),
        _fit_profile(
            cell.apical,
            "kir_gbar",
            jnp.where(
                apical_dist > 100.0,
                p["KirGbar"],
                p["KirGbar"] * apical_dist / 100.0,
            ),
            "KirGbar",
        ),
        _fit_profile(
            cell.axon, "nax_gbar", p["gna"] * p["AXNa"], "gna", "AXNa"
        ),
        _fit_profile(cell.axon, "kd_gbar", p["axongkdr"], "axongkdr"),
        _fit_profile(cell.axon, "km_gbar", 3.0 * p["soma_km"], "soma_km"),
        _fit_profile(cell.axon, "kap_gkabar", p["axon_kap"], "axon_kap"),
        _fit_profile(
            cell.axon, "Kv2like_gbar", p["gkv2axon"], "gkv2axon"
        ),
        _fit_profile(cell.basal, "na3dend_gbar", p["gnadend"], "gnadend"),
        _fit_profile(cell.basal, "nap_gnabar", p["nap_gnabar"], "nap_gnabar"),
        _fit_profile(cell.basal, "kap_gkabar", p["basal_kap"], "basal_kap"),
        _fit_profile(cell.basal, "h_gbar", p["soma_hbar"], "soma_hbar"),
        _fit_profile(cell.basal, "kd_gbar", p["gkdrdend"], "gkdrdend"),
        _fit_profile(
            cell.basal,
            "Kv2like_gbar",
            p["gkv2"] * p["gkv2scale"],
            "gkv2",
            "gkv2scale",
        ),
        _fit_profile(
            cell.basal,
            "kir_gbar",
            jnp.where(
                basal_dist > 40.0,
                p["KirGbar"],
                p["KirGbar"] * basal_dist / 40.0,
            ),
            "KirGbar",
        ),
    )


def _kinetic_fit_profiles(cell, p):
    """Return shared kinetic-scale profiles for each active channel placement."""

    return (
        _fit_profile(
            cell.soma,
            "kd_deactivation_tau_scale",
            p["kd_deactivation_tau_scale"],
            "kd_deactivation_tau_scale",
        ),
        _fit_profile(
            cell.apical,
            "kd_deactivation_tau_scale",
            p["kd_deactivation_tau_scale"],
            "kd_deactivation_tau_scale",
        ),
        _fit_profile(
            cell.axon,
            "kd_deactivation_tau_scale",
            p["kd_deactivation_tau_scale"],
            "kd_deactivation_tau_scale",
        ),
        _fit_profile(
            cell.basal,
            "kd_deactivation_tau_scale",
            p["kd_deactivation_tau_scale"],
            "kd_deactivation_tau_scale",
        ),
        _fit_profile(
            cell.soma,
            "na16a_fast_inactivation_tau_scale",
            p["nat_fast_inactivation_tau_scale"],
            "nat_fast_inactivation_tau_scale",
        ),
        _fit_profile(
            cell.apical,
            "na16a_fast_inactivation_tau_scale",
            p["nat_fast_inactivation_tau_scale"],
            "nat_fast_inactivation_tau_scale",
        ),
        _fit_profile(
            cell.axon,
            "nax_fast_inactivation_tau_scale",
            p["nat_fast_inactivation_tau_scale"],
            "nat_fast_inactivation_tau_scale",
        ),
        _fit_profile(
            cell.basal,
            "na3dend_fast_inactivation_tau_scale",
            p["nat_fast_inactivation_tau_scale"],
            "nat_fast_inactivation_tau_scale",
        ),
        _fit_profile(
            cell.soma,
            "na16a_slow_recovery_tau_scale",
            p["nat_slow_recovery_tau_scale"],
            "nat_slow_recovery_tau_scale",
        ),
        _fit_profile(
            cell.apical,
            "na16a_slow_recovery_tau_scale",
            p["nat_slow_recovery_tau_scale"],
            "nat_slow_recovery_tau_scale",
        ),
        _fit_profile(
            cell.soma,
            "h_tau_scale",
            p["h_tau_scale"],
            "h_tau_scale",
        ),
        _fit_profile(
            cell.apical,
            "h_tau_scale",
            p["h_tau_scale"],
            "h_tau_scale",
        ),
        _fit_profile(
            cell.basal,
            "h_tau_scale",
            p["h_tau_scale"],
            "h_tau_scale",
        ),
    )


def set_fitted_passive_parameters(
    cell,
    p,
    state=None,
    *,
    selected_keys=PASSIVE_PARAMETER_KEYS,
    reference=None,
    update_mode=None,
):
    selected_keys = frozenset(selected_keys)
    reference = reference or _reference_parameters(cell)
    update_mode = update_mode or _parameter_update_mode(cell)
    return _write_selected_profiles(
        _passive_fit_profiles(cell, p, update_mode),
        _passive_fit_profiles(cell, reference, update_mode),
        selected_keys,
        update_mode,
        state,
    )


def set_fitted_conductance_parameters(
    cell,
    p,
    state=None,
    *,
    selected_keys=CONDUCTANCE_PARAMETER_KEYS,
    reference=None,
    update_mode=None,
):
    selected_keys = frozenset(selected_keys)
    reference = reference or _reference_parameters(cell)
    update_mode = update_mode or _parameter_update_mode(cell)
    return _write_selected_profiles(
        _conductance_fit_profiles(cell, p, update_mode),
        _conductance_fit_profiles(cell, reference, update_mode),
        selected_keys,
        update_mode,
        state,
    )


def set_fitted_kinetic_parameters(
    cell,
    p,
    state=None,
    *,
    selected_keys=KINETIC_PARAMETER_KEYS,
    reference=None,
    update_mode=None,
):
    selected_keys = frozenset(selected_keys)
    reference = reference or _reference_parameters(cell)
    update_mode = update_mode or _parameter_update_mode(cell)
    return _write_selected_profiles(
        _kinetic_fit_profiles(cell, p),
        _kinetic_fit_profiles(cell, reference),
        selected_keys,
        update_mode,
        state,
    )


def set_fitted_parameters(cell, keys, values, state=None):
    keys = tuple(keys)
    if len(keys) != len(values):
        raise ValueError(
            f"Expected {len(keys)} Combe fit values for {len(keys)} keys, "
            f"received {len(values)}."
        )
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"Duplicate Combe fit parameter keys: {duplicates}")
    unsupported = sorted(set(keys) - SUPPORTED_FIT_PARAMETER_KEYS)
    if unsupported:
        raise KeyError(f"Unsupported Combe fit parameter keys: {unsupported}")
    if not keys:
        return state

    reference = _reference_parameters(cell)
    p = _fit_values(keys, values, reference)
    update_mode = _parameter_update_mode(cell)
    selected_keys = frozenset(keys)
    state = set_fitted_passive_parameters(
        cell,
        p,
        state,
        selected_keys=selected_keys,
        reference=reference,
        update_mode=update_mode,
    )
    state = set_fitted_conductance_parameters(
        cell,
        p,
        state,
        selected_keys=selected_keys,
        reference=reference,
        update_mode=update_mode,
    )
    state = set_fitted_kinetic_parameters(
        cell,
        p,
        state,
        selected_keys=selected_keys,
        reference=reference,
        update_mode=update_mode,
    )
    return state


def set_combe_channels(cell, p: CombeParameters = COMBE_PARAMS):
    set_common_reversals(cell)
    set_cal4_profile(cell, p)
    set_soma_channels(cell, p)
    set_apical_channels(cell, p)
    set_axon_channels(cell, p)
    set_basal_channels(cell, p)
    return cell


def Combe2023(
    d_lambda: float = 0.1,
    *,
    enable_calcium_diffusion: bool = True,
    params: CombeParameters = COMBE_PARAMS,
    morphology_source: str = "hoc",
):
    """Build the Combe2023 CCh-driven channel-placement port."""
    if morphology_source == "hoc":
        cell = build_hoc_section_cell(d_lambda=d_lambda)
    elif morphology_source == "swc":
        cell = jx.read_swc(
            str(morphology_path()),
            ncomp=1,
            assign_groups=True,
        )
        set_distances_from_soma(cell)
        set_passive_properties(cell, params)
        update_number_compartments(cell, d_lambda=d_lambda)
        cell.initialize()
    else:
        raise ValueError("morphology_source must be either 'hoc' or 'swc'.")

    set_distances_from_soma(cell)
    insert_combe_channels(cell)
    set_passive_properties(cell, params)
    set_combe_channels(cell, params)
    if params == COMBE_PARAMS and morphology_source == "hoc":
        apply_hoc_channel_profile(cell)

    if enable_calcium_diffusion:
        enable_cal4_diffusion(cell, axial_diffusion=0.22)

    cell.set("v", params.Epas)
    cell._combe_reference_parameters = asdict(params)
    cell._combe_parameter_update_mode = (
        EXACT_HOC_UPDATE_MODE
        if morphology_source == "hoc" and params == COMBE_PARAMS
        else RULE_UPDATE_MODE
    )
    return cell


def add_step_stimuli(
    cell,
    *,
    dt: float = 0.025,
    tstop: float = 500.0,
    delay: float = 100.0,
    duration: float = 300.0,
    amplitude: float = 0.3,
    v_init: float = COMBE_PARAMS.Epas,
):
    time = jnp.arange(0.0, tstop + dt, dt)
    current = jx.step_current(delay, duration, amplitude, dt, tstop)

    cell.delete_stimuli()
    cell.delete_recordings()
    cell.soma.branch(0).loc(0.5).stimulate(current)
    cell.soma.branch(0).loc(0.5).record()
    cell.set("v", v_init)
    cell.init_states()
    return cell, time, current


L5PC_Combe = Combe2023

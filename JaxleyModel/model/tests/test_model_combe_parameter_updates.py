from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JaxleyModel.model.model_Combe import (
    COMBE_PARAMS,
    CONDUCTANCE_PARAMETER_KEYS,
    EXACT_HOC_UPDATE_MODE,
    PASSIVE_PARAMETER_KEYS,
    RULE_UPDATE_MODE,
    Combe2023,
    bounds,
    set_fitted_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TARGETS = {
    "RmSoma": (("Leak_gLeak", group) for group in ("soma", "axon", "basal", "apical")),
    "RmTuft": (("Leak_gLeak", group) for group in ("soma", "axon", "basal", "apical")),
    "DistHalfRm": (
        ("Leak_gLeak", group) for group in ("soma", "axon", "basal", "apical")
    ),
    "SlopeRm": (("Leak_gLeak", group) for group in ("soma", "axon", "basal", "apical")),
    "RaSoma": (
        ("axial_resistivity", group)
        for group in ("soma", "axon", "basal", "apical")
    ),
    "RaTuft": (
        ("axial_resistivity", group)
        for group in ("soma", "axon", "basal", "apical")
    ),
    "DistHalfRa": (
        ("axial_resistivity", group)
        for group in ("soma", "axon", "basal", "apical")
    ),
    "SlopeRa": (
        ("axial_resistivity", group)
        for group in ("soma", "axon", "basal", "apical")
    ),
    "Epas": (("Leak_eLeak", "cell"),),
    "CmSoma": (
        ("capacitance", "soma"),
        ("capacitance", "axon"),
        ("capacitance", "basal"),
        ("capacitance", "apical"),
    ),
    "SpineFactorBasal": (
        ("capacitance", "basal"),
        ("Leak_gLeak", "basal"),
    ),
    "SpineFactorTuft": (
        ("capacitance", "apical"),
        ("Leak_gLeak", "apical"),
    ),
    "soma_hbar": (
        ("h_gbar", "soma"),
        ("h_gbar", "apical"),
        ("h_gbar", "basal"),
    ),
    "KirGbar": (("kir_gbar", "apical"), ("kir_gbar", "basal")),
    "soma_caL": (("cal_gcalbar", "soma"),),
    "soma_car": (("car_gcabar", "apical"),),
    "gsomacar": (("car_gcabar", "soma"),),
    "soma_caLH": (("calH_gcalbar", "apical"),),
    "soma_caT": (("cat_gcatbar", "soma"), ("cat_gcatbar", "apical")),
    "soma_km": (
        ("km_gbar", "soma"),
        ("km_gbar", "apical"),
        ("km_gbar", "axon"),
    ),
    "mykca_init": (("mykca_gkbar", "soma"), ("mykca_gkbar", "apical")),
    "soma_kca": (("kca_gbar", "soma"), ("kca_gbar", "apical")),
    "AXNa": (("nax_gbar", "axon"),),
    "gkdrsoma": (("kd_gbar", "soma"),),
    "gkdrdend": (("kd_gbar", "basal"),),
    "soma_kap": (("kap_gkabar", "soma"), ("kap_gkabar", "apical")),
    "axon_kap": (("kap_gkabar", "axon"),),
    "basal_kap": (("kap_gkabar", "basal"),),
    "soma_kad": (("kad_gkabar", "apical"),),
    "gna": (("na16a_gbar", "soma"), ("nax_gbar", "axon")),
    "axongkdr": (("kd_gbar", "axon"),),
    "gnadend": (("na16a_gbar", "apical"), ("na3dend_gbar", "basal")),
    "gkdrapical": (("kd_gbar", "apical"),),
    "gkv2soma": (("Kv2like_gbar", "soma"),),
    "gkv2": (("Kv2like_gbar", "apical"), ("Kv2like_gbar", "basal")),
    "gkv2axon": (("Kv2like_gbar", "axon"),),
    "gkv2scale": (("Kv2like_gbar", "apical"), ("Kv2like_gbar", "basal")),
    "scale_Na_conduct": (
        ("na16a_gbar", "soma"),
        ("na16a_gbar", "apical"),
    ),
    "icangbar": (("icand_gbar", "soma"), ("icand_gbar", "apical")),
    "nap_gnabar": (("nap_gnabar", "soma"), ("nap_gnabar", "basal")),
}

# Materialize generator-valued entries once so parametrized tests are repeatable.
EXPECTED_TARGETS = {
    key: tuple(targets) for key, targets in EXPECTED_TARGETS.items()
}


@pytest.fixture(scope="module")
def hoc_cell():
    if not (REPO_ROOT / "Combe2023").is_dir():
        pytest.skip("The ignored Combe2023 HOC source is not available.")
    return Combe2023(d_lambda=0.3, enable_calcium_diffusion=False)


def _indices(update):
    return np.asarray(update["indices"]).reshape(-1)


def _baseline(cell, update):
    return cell.nodes.loc[_indices(update), update["key"]].to_numpy(dtype=float)


def _group_for_update(cell, update):
    indices = _indices(update)
    matching = [
        group
        for group in ("soma", "axon", "basal", "apical")
        if cell.nodes.loc[indices, group].to_numpy(dtype=bool).all()
    ]
    return matching[0] if len(matching) == 1 else "cell"


def test_empty_update_is_a_true_noop(hoc_cell):
    original_state = []
    result = set_fitted_parameters(
        hoc_cell,
        (),
        jnp.asarray([], dtype=float),
        original_state,
    )
    assert result is original_state
    assert result == []


def test_reference_values_are_bitwise_identity(hoc_cell):
    keys = CONDUCTANCE_PARAMETER_KEYS + PASSIVE_PARAMETER_KEYS
    values = jnp.asarray([getattr(COMBE_PARAMS, key) for key in keys])

    updates = set_fitted_parameters(hoc_cell, keys, values)

    assert updates
    assert "na16a_dist" not in {update["key"] for update in updates}
    assert "na16a_C1O1v2" not in {update["key"] for update in updates}
    for update in updates:
        assert np.array_equal(np.asarray(update["val"]), _baseline(hoc_cell, update))


@pytest.mark.parametrize(
    "key",
    sorted(EXPECTED_TARGETS),
)
def test_each_knob_only_writes_declared_targets(hoc_cell, key):
    lower, upper = bounds[key]
    value = lower + 0.37 * (upper - lower)
    if value == getattr(COMBE_PARAMS, key):
        value = lower + 0.63 * (upper - lower)

    updates = set_fitted_parameters(hoc_cell, (key,), jnp.asarray([value]))
    actual_targets = Counter(
        (update["key"], _group_for_update(hoc_cell, update)) for update in updates
    )
    assert actual_targets == Counter(EXPECTED_TARGETS[key])
    assert any(
        not np.array_equal(np.asarray(update["val"]), _baseline(hoc_cell, update))
        for update in updates
    )


def test_hoc_endpoint_feature_reproduces_sectionwise_profiles(hoc_cell):
    assert hoc_cell._combe_parameter_update_mode == EXACT_HOC_UPDATE_MODE
    assert "hoc_assignment_distance_um" in hoc_cell.nodes.columns

    apical = hoc_cell.apical.nodes
    distance = np.minimum(
        apical["hoc_assignment_distance_um"].to_numpy(dtype=float),
        500.0,
    )
    expected_h = COMBE_PARAMS.soma_hbar * (
        1.0 + (6.0 / 5.0) * distance / 100.0
    )
    np.testing.assert_allclose(
        apical["h_gbar"].to_numpy(dtype=float),
        expected_h,
        rtol=1e-12,
        atol=1e-14,
    )
    assert apical.groupby("hoc_section_index")["h_gbar"].nunique().max() == 1


def test_joint_passive_update_uses_hoc_endpoint_rules_on_frozen_grid(hoc_cell):
    replacements = {
        "RmSoma": 120_000.0,
        "RaSoma": 80.0,
        "RmTuft": 60_000.0,
        "RaTuft": 55.0,
        "DistHalfRm": 180.0,
        "DistHalfRa": 110.0,
        "SlopeRm": 20.0,
        "SlopeRa": 12.0,
        "Epas": -68.0,
        "CmSoma": 1.5,
        "SpineFactorBasal": 4.0,
        "SpineFactorTuft": 5.0,
    }
    keys = tuple(replacements)
    updates = set_fitted_parameters(
        hoc_cell,
        keys,
        jnp.asarray([replacements[key] for key in keys]),
    )

    for update in updates:
        indices = _indices(update)
        section_ids = hoc_cell.nodes.loc[
            indices, "hoc_section_index"
        ].to_numpy(dtype=int)
        values = np.asarray(update["val"])
        for section_id in np.unique(section_ids):
            assert np.unique(values[section_ids == section_id]).size == 1

    apical_capacitance = next(
        update
        for update in updates
        if update["key"] == "capacitance"
        and _group_for_update(hoc_cell, update) == "apical"
    )
    distance = hoc_cell.nodes.loc[
        _indices(apical_capacitance), "hoc_assignment_distance_um"
    ].to_numpy(dtype=float)
    spine_factor = np.where(
        distance <= 100.0,
        1.0,
        np.where(
            distance > 394.0,
            replacements["SpineFactorTuft"],
            2.0
            + (distance - 100.0)
            * (replacements["SpineFactorTuft"] - 2.0)
            / 294.0,
        ),
    )
    expected = spine_factor * replacements["CmSoma"]
    np.testing.assert_allclose(
        np.asarray(apical_capacitance["val"]),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )


def test_coupled_and_zero_reference_rules(hoc_cell):
    gna = COMBE_PARAMS.gna * 1.2
    na_scale = COMBE_PARAMS.scale_Na_conduct * 0.9
    updates = set_fitted_parameters(
        hoc_cell,
        ("gna", "scale_Na_conduct"),
        jnp.asarray([gna, na_scale]),
    )
    soma_na = next(
        update
        for update in updates
        if update["key"] == "na16a_gbar"
        and _group_for_update(hoc_cell, update) == "soma"
    )
    np.testing.assert_allclose(
        np.asarray(soma_na["val"]),
        gna * na_scale,
        rtol=1e-12,
        atol=1e-14,
    )

    zero_reference = set_fitted_parameters(
        hoc_cell,
        ("soma_kca",),
        jnp.asarray([1.0e-5]),
    )
    assert {update["key"] for update in zero_reference} == {"kca_gbar"}
    assert all(np.asarray(update["val"]).max() > 0.0 for update in zero_reference)


def test_parameter_updates_remain_jittable_and_differentiable(hoc_cell):
    @jax.jit
    def total_h_conductance(value):
        updates = set_fitted_parameters(
            hoc_cell,
            ("soma_hbar",),
            jnp.asarray([value]),
        )
        return sum(jnp.sum(update["val"]) for update in updates)

    value = total_h_conductance(COMBE_PARAMS.soma_hbar)
    gradient = jax.grad(total_h_conductance)(COMBE_PARAMS.soma_hbar)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_rule_mode_uses_final_compartment_centers():
    cell = Combe2023(
        d_lambda=0.3,
        enable_calcium_diffusion=False,
        morphology_source="swc",
    )
    assert cell._combe_parameter_update_mode == RULE_UPDATE_MODE

    new_h = COMBE_PARAMS.soma_hbar * 1.5
    updates = set_fitted_parameters(cell, ("soma_hbar",), jnp.asarray([new_h]))
    apical_h = next(
        update
        for update in updates
        if update["key"] == "h_gbar"
        and _group_for_update(cell, update) == "apical"
    )
    distance = np.minimum(
        cell.nodes.loc[_indices(apical_h), "dist_from_soma"].to_numpy(dtype=float),
        500.0,
    )
    expected = new_h * (1.0 + (6.0 / 5.0) * distance / 100.0)
    np.testing.assert_allclose(
        np.asarray(apical_h["val"]),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("keys", "values", "error"),
    (
        (("soma_hbar",), (), ValueError),
        (("soma_hbar", "soma_hbar"), (1.0, 2.0), ValueError),
        (("not_a_parameter",), (1.0,), KeyError),
    ),
)
def test_invalid_parameter_requests_fail_clearly(hoc_cell, keys, values, error):
    with pytest.raises(error):
        set_fitted_parameters(hoc_cell, keys, jnp.asarray(values))

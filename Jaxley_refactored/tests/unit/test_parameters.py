from pathlib import Path

import numpy as np

from jaxley_refactored.config import load_config
from jaxley_refactored.parameters import (
    Parameterizer,
    ProjectedBoxSpace,
    combe2023_catalog,
)


PROJECT = Path(__file__).resolve().parents[2]
KINETIC_NAMES = (
    "kd_deactivation_tau_scale",
    "nat_fast_inactivation_tau_scale",
    "nat_slow_recovery_tau_scale",
    "h_tau_scale",
)


class CapturingBackend:
    def parameter_state(self, cell, keys, values, state=None):
        return tuple(keys), np.asarray(values), state


def test_catalog_preserves_the_legacy_40_and_appends_four_kinetic_scales():
    catalog = combe2023_catalog()
    legacy = catalog.select(include_tags=("conductance", "passive"))
    selected = catalog.select(
        include_tags=("conductance", "passive", "kinetics")
    )

    assert len(legacy) == 40
    assert legacy[-1].name == "SpineFactorTuft"
    assert len(selected) == 44
    assert tuple(spec.name for spec in selected[-4:]) == KINETIC_NAMES


def test_kinetic_metadata_and_persistent_sodium_remain_explicit():
    catalog = combe2023_catalog()
    expected_bounds = {
        "kd_deactivation_tau_scale": (0.25, 4.0),
        "nat_fast_inactivation_tau_scale": (0.5, 2.0),
        "nat_slow_recovery_tau_scale": (0.5, 2.0),
        "h_tau_scale": (0.5, 2.0),
    }
    for name, bounds in expected_bounds.items():
        spec = catalog.get(name)
        assert spec.default == 1.0
        assert spec.bounds == bounds
        assert spec.tags == ("kinetics",)
        assert spec.units == "dimensionless"

    persistent_sodium = catalog.get("nap_gnabar")
    assert persistent_sodium.default == 0.0
    assert persistent_sodium.bounds == (0.0, 0.001)
    assert persistent_sodium.tags == ("conductance",)
    assert persistent_sodium.targets == (
        "soma.nap_gnabar",
        "basal.nap_gnabar",
    )


def test_all_lsu_local_and_hybrid_configs_select_the_same_44_parameters():
    catalog = combe2023_catalog()
    paths = (
        PROJECT / "configs/losses/LSU_1.yaml",
        PROJECT / "configs/losses/LSU_1_wide_bounds.yaml",
        PROJECT / "configs/losses/LSU_1_wide_bounds_adam.yaml",
        PROJECT / "configs/search/LSU_1_cma_adam.yaml",
    )
    expected = None
    for path in paths:
        config = load_config(path)
        specs = catalog.select(
            include_tags=config.model.parameters.include_tags,
            include=config.model.parameters.include,
            exclude=config.model.parameters.exclude,
        )
        names = tuple(spec.name for spec in specs)
        assert len(names) == 44
        assert names[-4:] == KINETIC_NAMES
        assert "nap_gnabar" in names
        expected = names if expected is None else expected
        assert names == expected

    hybrid = load_config(PROJECT / "configs/search/LSU_1_cma_adam.yaml")
    assert hybrid.search.global_search.parameter_names == ()


def test_projected_box_preserves_exact_zero_and_bounds():
    specs = combe2023_catalog().select(include=("soma_caLH",))
    space = ProjectedBoxSpace.from_specs(specs)

    assert float(space.normalize([0.0])[0]) == 0.0
    np.testing.assert_allclose(space.project([-1.0, 2.0]), [0.0, 1.0])


def test_parameter_bounds_can_be_expanded_without_changing_defaults():
    catalog = combe2023_catalog()
    conductance = catalog.get("soma_hbar").with_expanded_bounds(2.0)
    resistance = catalog.get("RmSoma").with_expanded_bounds(2.0)
    reversal = catalog.get("Epas").with_expanded_bounds(2.0)

    assert conductance.bounds == (0.0, 0.0006)
    assert resistance.bounds == (25_000.0, 600_000.0)
    np.testing.assert_allclose(reversal.bounds, (-108.0121, -28.0121))
    assert conductance.default == catalog.get("soma_hbar").default
    assert resistance.default == catalog.get("RmSoma").default
    assert reversal.default == catalog.get("Epas").default


def test_parameterizer_appends_fixed_distribution_coefficients():
    catalog = combe2023_catalog()
    fitted = catalog.select(include=("soma_hbar",))
    fixed = catalog.select(include=("DistHalfRm",))
    parameterizer = Parameterizer(
        cell=object(),
        specs=fitted,
        backend=CapturingBackend(),
        fixed_specs=fixed,
        fixed_values=(175.0,),
    )

    keys, values, state = parameterizer.state(np.asarray([4e-5]), {"base": True})
    assert keys == ("soma_hbar", "DistHalfRm")
    np.testing.assert_allclose(values, [4e-5, 175.0])
    assert state == {"base": True}

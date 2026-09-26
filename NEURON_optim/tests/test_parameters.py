import numpy as np

from neuron_optim.parameters import (
    ALL_KEYS,
    CONDUCTANCE,
    KINETIC,
    PASSIVE,
    complete_parameter_mapping,
    local_normalized_bounds,
    local_relative_bounds,
    make_parameter_space,
    passive_model_values,
)


def test_default_parameter_space_is_bounded_and_56_dimensional():
    space = make_parameter_space()
    assert len(ALL_KEYS) == 56
    assert len(space.keys) == 56
    assert np.all(space.lower < space.upper)
    normalized = space.normalize(space.reference)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    np.testing.assert_allclose(space.physical(normalized), space.reference)


def test_passive_model_mapping_disables_active_currents():
    space = make_parameter_space(include=PASSIVE)
    values = passive_model_values(dict(zip(space.keys, space.reference, strict=True)))

    assert set(space.keys) == set(PASSIVE)
    assert all(values[key] == 0.0 for key in CONDUCTANCE)
    assert all(values[key] == 1.0 for key in KINETIC)
    assert values["Epas"] == space.reference[PASSIVE.index("Epas")]


def test_local_bounds_and_complete_mapping_keep_fixed_parameters():
    space = make_parameter_space(include=("gna", "nav16_I1O1b1"))
    centers = {"gna": 0.08, "nav16_I1O1b1": 0.005}
    lower, upper = local_normalized_bounds(centers, space.keys, 0.15)
    local = make_parameter_space(
        include=space.keys, lower_overrides=lower, upper_overrides=upper
    )
    normalized = local.normalize([centers[key] for key in local.keys])
    mapping = complete_parameter_mapping(local, normalized, {"Epas": -70.0})

    assert set(mapping) >= set(ALL_KEYS)
    assert mapping["Epas"] == -70.0
    assert mapping["gna"] == centers["gna"]
    assert mapping["nav16_I1O1b1"] == centers["nav16_I1O1b1"]
    assert np.all(local.lower < local.upper)


def test_local_relative_bounds_keep_zero_centers_variable():
    lower, upper = local_relative_bounds(
        {"gkdrsoma": 0.0, "nav16_C1O1v2": -25.0},
        ("gkdrsoma", "nav16_C1O1v2"),
        0.15,
        lower_limits={"gkdrsoma": 0.0, "nav16_C1O1v2": -32.0},
        upper_limits={"gkdrsoma": 0.01, "nav16_C1O1v2": -20.0},
    )
    assert lower["gkdrsoma"] == 0.0
    assert upper["gkdrsoma"] > 0.0
    np.testing.assert_allclose(
        [lower["nav16_C1O1v2"], upper["nav16_C1O1v2"]],
        [-28.75, -21.25],
    )


def test_nav16_patch_scales_every_intended_rate(tmp_path):
    from neuron_optim.mechanisms import _patch_sources, _repo_root

    _patch_sources(_repo_root() / "channels_converted" / "mod", tmp_path)
    text = (tmp_path / "Nav16_a.mod").read_text()
    for rate in ("I1C1_a", "C1I1_a"):
        line = next(line for line in text.splitlines() if line.strip().startswith(f"{rate} ="))
        assert line.rstrip().endswith("/ fast_inactivation_tau_scale")
    for rate in ("I1I2_a", "I2I1_a"):
        line = next(line for line in text.splitlines() if line.strip().startswith(f"{rate} ="))
        assert line.rstrip().endswith("/ slow_recovery_tau_scale")

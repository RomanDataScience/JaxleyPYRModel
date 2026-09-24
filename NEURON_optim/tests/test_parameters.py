import numpy as np

from neuron_optim.parameters import (
    ALL_KEYS,
    CONDUCTANCE,
    KINETIC,
    PASSIVE,
    complete_parameter_mapping,
    local_normalized_bounds,
    make_parameter_space,
    passive_model_values,
)


def test_default_parameter_space_is_bounded_and_51_dimensional():
    space = make_parameter_space()
    assert len(ALL_KEYS) == 53
    assert len(space.keys) == 53
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
    assert values["persist"] == 0.0
    assert values["Epas"] == space.reference[PASSIVE.index("Epas")]


def test_local_bounds_and_complete_mapping_keep_fixed_parameters():
    space = make_parameter_space(include=("gna", "persist"))
    centers = {"gna": 0.08, "persist": 0.006}
    lower, upper = local_normalized_bounds(centers, space.keys, 0.15)
    local = make_parameter_space(
        include=space.keys, lower_overrides=lower, upper_overrides=upper
    )
    normalized = local.normalize([centers[key] for key in local.keys])
    mapping = complete_parameter_mapping(local, normalized, {"Epas": -70.0})

    assert set(mapping) >= set(ALL_KEYS)
    assert mapping["Epas"] == -70.0
    assert mapping["gna"] == centers["gna"]
    assert mapping["persist"] == centers["persist"]
    assert np.all(local.lower < local.upper)

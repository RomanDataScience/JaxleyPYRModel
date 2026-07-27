import numpy as np

from jaxley_refactored.parameters import (
    Parameterizer,
    ProjectedBoxSpace,
    combe2023_catalog,
)


class CapturingBackend:
    def parameter_state(self, cell, keys, values, state=None):
        return tuple(keys), np.asarray(values), state


def test_catalog_contains_the_legacy_40_parameters():
    catalog = combe2023_catalog()
    selected = catalog.select(include_tags=("conductance", "passive"))

    assert len(selected) == 40
    assert selected[-1].name == "SpineFactorTuft"


def test_projected_box_preserves_exact_zero_and_bounds():
    specs = combe2023_catalog().select(include=("soma_caLH",))
    space = ProjectedBoxSpace.from_specs(specs)

    assert float(space.normalize([0.0])[0]) == 0.0
    np.testing.assert_allclose(space.project([-1.0, 2.0]), [0.0, 1.0])


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

import numpy as np

from neuron_optim import stages
from neuron_optim.nav_window import open_probability, window_ratio
from neuron_optim.parameters import DEFAULTS


def test_default_model_has_unit_ratio_and_small_resting_open_probability():
    assert window_ratio(DEFAULTS) == 1.0
    assert 0.0 < open_probability(-60.0, DEFAULTS) < 5e-3


def test_ratio_tracks_recovery_rate_and_conductance():
    fast_recovery = {**DEFAULTS, "nav16_I1O1b1": 0.5}  # old upper bound
    assert window_ratio(fast_recovery) > 2.0
    doubled_dend = {**DEFAULTS, "gnadend": 2.0 * DEFAULTS["gnadend"]}
    np.testing.assert_allclose(window_ratio(doubled_dend), 2.0)


def test_rejection_is_graded_and_skips_sodium_free_stages():
    options = {"nav_window": {"max_ratio": 2.0, "penalty": 100.0}}
    assert stages._nav_window_rejection(dict(DEFAULTS), options) is None
    mild = stages._nav_window_rejection({**DEFAULTS, "gnadend": 3.0 * DEFAULTS["gnadend"]}, options)
    severe = stages._nav_window_rejection({**DEFAULTS, "gnadend": 30.0 * DEFAULTS["gnadend"]}, options)
    assert 100.0 < mild[0] < severe[0]
    assert severe[1]["error"] == "NavWindowTooLarge"
    sodium_off = {**DEFAULTS, "gnadend": 1.0, "gna": 0.0, "gna12": 0.0, "gnaaxon": 0.0}
    sodium_off["gnadend"] = 0.0
    assert stages._nav_window_rejection(sodium_off, options) is None
    assert stages._nav_window_rejection({**DEFAULTS, "gnadend": 1.0}, {}) is None


def test_rejected_candidate_is_not_simulated():
    class Boom:
        def simulate_many(self, *_args, **_kwargs):
            raise AssertionError("should not simulate")

    loss, payload, error = stages._simulate_and_score(
        "depol_full", {**DEFAULTS, "gnadend": 1.0}, Boom(), [], [], None,
        {"nav_window": {"max_ratio": 2.0}},
    )
    assert payload is None and error["error"] == "NavWindowTooLarge" and loss > 2.0e4

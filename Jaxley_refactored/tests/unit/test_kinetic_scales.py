import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_refactored.compatibility.model_combe import legacy_module


@pytest.fixture(scope="module")
def channels():
    legacy = legacy_module()
    return {
        "kd": legacy.Kd("kd"),
        "h": legacy.H("h"),
        "nax": legacy.Nax("nax"),
        "na3dend": legacy.Na3Dend("na3dend"),
        "na16a": legacy.Nav16A("na16a"),
    }


def _rates(channel, voltage, **updates):
    params = dict(channel.channel_params)
    params.update(updates)
    return tuple(channel.rates(jnp.asarray(voltage), params))


@pytest.mark.parametrize("scale", (0.25, 1.0, 4.0))
def test_kd_scale_changes_only_the_shared_activation_deactivation_tau(
    channels, scale
):
    baseline = _rates(channels["kd"], jnp.asarray([-80.0, 20.0]))
    scaled = _rates(
        channels["kd"],
        jnp.asarray([-80.0, 20.0]),
        kd_deactivation_tau_scale=scale,
    )

    np.testing.assert_allclose(scaled[0], baseline[0])
    np.testing.assert_allclose(scaled[1], 0.6 * scale)
    np.testing.assert_allclose(scaled[2], baseline[2])
    np.testing.assert_allclose(scaled[3], baseline[3])


@pytest.mark.parametrize("scale", (0.5, 1.0, 2.0))
def test_h_scale_changes_tau_after_floor_without_changing_steady_state(
    channels, scale
):
    voltage = jnp.asarray([-120.0, -65.0, 20.0])
    baseline = _rates(channels["h"], voltage)
    scaled = _rates(channels["h"], voltage, h_tau_scale=scale)

    np.testing.assert_allclose(scaled[0], baseline[0])
    np.testing.assert_allclose(scaled[1], baseline[1] * scale)


@pytest.mark.parametrize(
    ("channel_name", "parameter_name"),
    (
        ("nax", "nax_fast_inactivation_tau_scale"),
        ("na3dend", "na3dend_fast_inactivation_tau_scale"),
    ),
)
def test_hh_sodium_fast_scale_changes_only_inactivation_tau(
    channels, channel_name, parameter_name
):
    voltage = jnp.asarray([-80.0, -20.0, 20.0])
    baseline = _rates(channels[channel_name], voltage)
    scaled = _rates(channels[channel_name], voltage, **{parameter_name: 2.0})

    np.testing.assert_allclose(scaled[0], baseline[0])
    np.testing.assert_allclose(scaled[1], baseline[1])
    np.testing.assert_allclose(scaled[2], baseline[2])
    np.testing.assert_allclose(scaled[3], baseline[3] * 2.0)
    for index in range(4, len(baseline)):
        np.testing.assert_allclose(scaled[index], baseline[index])


def test_nav16_fast_scale_changes_only_paired_fast_inactivation_rates(channels):
    baseline = _rates(
        channels["na16a"],
        -35.0,
        na16a_persist=0.2,
        na16a_dist=1.0,
    )
    scaled = _rates(
        channels["na16a"],
        -35.0,
        na16a_persist=0.2,
        na16a_dist=1.0,
        na16a_fast_inactivation_tau_scale=2.0,
    )

    np.testing.assert_allclose(scaled[:2], baseline[:2])
    np.testing.assert_allclose(scaled[2:6], np.asarray(baseline[2:6]) / 2.0)
    np.testing.assert_allclose(scaled[6:], baseline[6:])


def test_nav16_slow_scale_changes_entry_and_recovery_without_changing_ratio(
    channels,
):
    baseline = _rates(channels["na16a"], -35.0, na16a_dist=1.0)
    scaled = _rates(
        channels["na16a"],
        -35.0,
        na16a_dist=1.0,
        na16a_slow_recovery_tau_scale=2.0,
    )

    np.testing.assert_allclose(scaled[:6], baseline[:6])
    np.testing.assert_allclose(scaled[6:], np.asarray(baseline[6:]) / 2.0)
    np.testing.assert_allclose(
        scaled[6] / scaled[7],
        baseline[6] / baseline[7],
    )


@pytest.mark.parametrize(
    ("fast_scale", "slow_scale"),
    ((0.5, 0.5), (2.0, 2.0)),
)
def test_nav16_probability_update_is_finite_at_scale_bounds(
    channels, fast_scale, slow_scale
):
    channel = channels["na16a"]
    params = dict(channel.channel_params)
    params.update(
        {
            "na16a_dist": 1.0,
            "na16a_fast_inactivation_tau_scale": fast_scale,
            "na16a_slow_recovery_tau_scale": slow_scale,
        }
    )
    states = {
        "na16a_C1": jnp.asarray(0.70),
        "na16a_O1": jnp.asarray(0.10),
        "na16a_I1": jnp.asarray(0.15),
        "na16a_I2": jnp.asarray(0.05),
    }

    updated = channel.update_states(states, 0.05, jnp.asarray(-20.0), params)
    probabilities = jnp.stack(tuple(updated.values()))

    assert np.isfinite(np.asarray(probabilities)).all()
    assert bool(jnp.all(probabilities >= 0.0))
    np.testing.assert_allclose(jnp.sum(probabilities), 1.0, atol=1e-12)


def test_each_kinetic_scale_has_a_finite_nonzero_rate_gradient(channels):
    def kd_value(scale):
        return _rates(
            channels["kd"], -65.0, kd_deactivation_tau_scale=scale
        )[1]

    def h_value(scale):
        return _rates(channels["h"], -65.0, h_tau_scale=scale)[1]

    def fast_na_value(scale):
        rates = _rates(
            channels["na16a"],
            -35.0,
            na16a_persist=0.2,
            na16a_dist=1.0,
            na16a_fast_inactivation_tau_scale=scale,
        )
        return sum(rates[2:6])

    def slow_na_value(scale):
        rates = _rates(
            channels["na16a"],
            -35.0,
            na16a_dist=1.0,
            na16a_slow_recovery_tau_scale=scale,
        )
        return rates[6] + rates[7]

    gradients = jnp.asarray(
        [
            jax.grad(kd_value)(1.0),
            jax.grad(fast_na_value)(1.0),
            jax.grad(slow_na_value)(1.0),
            jax.grad(h_value)(1.0),
        ]
    )
    assert np.isfinite(np.asarray(gradients)).all()
    assert bool(jnp.all(jnp.abs(gradients) > 0.0))

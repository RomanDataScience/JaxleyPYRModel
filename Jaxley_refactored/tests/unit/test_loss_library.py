from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_refactored.config import load_config
from jaxley_refactored.config.schema import LossComponentSpec, LossPenaltySpec
from jaxley_refactored.data import (
    TraceRecord,
    bucket_records,
    weight_records,
)
from jaxley_refactored.fitting.losses import (
    BucketObjective,
    apply_multiplicative_penalties,
    component_denominators,
    default_loss_registry,
    observed_interspike_masks,
    soft_dblo_error,
    soft_upward_crossing_count,
)
from jaxley_refactored.fitting.trainer import Trainer


PROJECT = Path(__file__).resolve().parents[2]


def test_example_loss_configs_are_valid_and_have_unique_components():
    expected = {
        "voltage_mse.yaml": ("waveform_mse",),
        "pseudo_huber.yaml": ("waveform_huber",),
        "huber_derivative_passive.yaml": (
            "waveform_huber",
            "waveform_derivative",
            "resting_voltage",
            "hyperpolarizing_steady_state",
        ),
        "LSU_1.yaml": (
            "hyperpolarizing_waveform_mse",
            "depolarizing_firing_rate",
            "depolarizing_dblo",
            "depolarizing_spike_waveform",
            "depolarizing_spike_derivative",
            "depolarizing_spike_height",
            "depolarizing_recovery_waveform",
            "depolarizing_recovery_derivative",
            "depolarizing_ahp_depth",
        ),
    }
    for filename, labels in expected.items():
        config = load_config(PROJECT / "configs/losses" / filename)
        assert tuple(component.label for component in config.fit.components) == labels
    lsu = load_config(PROJECT / "configs/losses/LSU_1.yaml")
    assert not lsu.fit.renormalize_protocol_filtered_components
    assert len(lsu.fit.penalties) == 1
    assert lsu.fit.penalties[0].label == "outside_step_spikes"
    assert lsu.fit.penalties[0].factor_per_spike == 1.1
    assert len(lsu.fit.components) == 9
    component = lsu.fit.components[0]
    assert component.label == "hyperpolarizing_waveform_mse"
    assert component.kind == "voltage_mse"
    assert component.weight == 1.0
    assert component.protocols == ("hyperpolarizing_pulse",)
    assert component.window == "score"
    assert component.scale == 5.0
    firing_rate = lsu.fit.components[1]
    assert firing_rate.label == "depolarizing_firing_rate"
    assert firing_rate.kind == "soft_firing_rate_error"
    assert firing_rate.weight == 4.0
    assert firing_rate.protocols == ("depolarizing_step",)
    assert firing_rate.window == "stimulus"
    assert firing_rate.threshold_mV == -20.0
    assert firing_rate.temperature_mV == 2.0
    assert firing_rate.scale == 5.0
    components = {item.label: item for item in lsu.fit.components}
    assert components["depolarizing_dblo"].weight == 2.0
    assert components["depolarizing_spike_waveform"].weight == 0.25
    assert components["depolarizing_spike_derivative"].weight == 0.25
    assert components["depolarizing_spike_height"].weight == 0.32
    assert components["depolarizing_recovery_waveform"].weight == 0.5
    assert components["depolarizing_recovery_derivative"].weight == 0.2
    assert components["depolarizing_ahp_depth"].weight == 0.10


def test_every_lsu_variant_inherits_the_same_reweighted_objective():
    paths = (
        PROJECT / "configs/losses/LSU_1.yaml",
        PROJECT / "configs/losses/LSU_1_wide_bounds.yaml",
        PROJECT / "configs/losses/LSU_1_wide_bounds_adam.yaml",
        PROJECT / "configs/search/LSU_1_cma_adam.yaml",
        PROJECT / "configs/search/LSU_1_cma_adam_smoke.yaml",
    )
    configs = tuple(load_config(path) for path in paths)
    reference = configs[0].fit
    assert all(
        config.dataset.simulation_post_ms == 500.0
        and config.dataset.simulation_post_ms_by_protocol
        == {"hyperpolarizing_pulse": 100.0}
        and config.dataset.score_post_ms == 500.0
        for config in configs
    )

    for config in configs[1:]:
        assert config.fit.components == reference.components
        assert config.fit.penalties == reference.penalties
        assert config.fit.protocol_weights == reference.protocol_weights
        assert (
            config.fit.renormalize_protocol_filtered_components
            == reference.renormalize_protocol_filtered_components
        )


def test_registered_waveform_losses_are_finite_and_differentiable():
    registry = default_loss_registry()
    observed = jnp.asarray([[0.0, 1.0, 2.0, 3.0]])
    mask = jnp.asarray([[True, True, True, True]])

    for name in (
        "voltage_mse",
        "voltage_mae",
        "pseudo_huber",
        "normalized_voltage_mse",
        "derivative_mse",
        "correlation_loss",
        "resting_voltage_error",
        "steady_state_error",
        "soft_firing_rate_error",
        "subthreshold_mean_error",
        "soft_dblo_error",
        "soft_minimum_voltage_error",
        "soft_maximum_voltage_error",
    ):
        term = registry.get(name)

        def scalar(predicted):
            return jnp.sum(
                term(
                    predicted,
                    observed,
                    mask,
                    dt_ms=0.05,
                    scale=2.0,
                    delta=1.0,
                    threshold_mV=-20.0,
                    temperature_mV=2.0,
                    baseline_mask=mask,
                    interspike_masks=mask[:, None, :],
                    interspike_valid=jnp.ones((1, 1), dtype=bool),
                )
            )

        value, gradient = jax.value_and_grad(scalar)(observed + 0.5)
        assert np.isfinite(float(value))
        assert np.isfinite(np.asarray(gradient)).all()


def test_pseudo_huber_is_less_sensitive_to_a_large_outlier_than_mse():
    registry = default_loss_registry()
    observed = jnp.zeros((1, 3))
    predicted = jnp.asarray([[0.0, 0.0, 100.0]])
    mask = jnp.ones_like(observed, dtype=bool)

    mse = registry.get("voltage_mse")(
        predicted, observed, mask, dt_ms=0.05, scale=1.0, delta=1.0
    )
    huber = registry.get("pseudo_huber")(
        predicted, observed, mask, dt_ms=0.05, scale=1.0, delta=1.0
    )
    assert float(huber[0]) < float(mse[0])


def test_soft_firing_rate_error_prefers_matching_spike_count():
    registry = default_loss_registry()
    term = registry.get("soft_firing_rate_error")
    observed = jnp.asarray([[-65.0, -65.0, 20.0, -65.0, -65.0, 20.0, -65.0]])
    matching = observed
    missing_spike = jnp.asarray([[-65.0, -65.0, 20.0, -65.0, -65.0, -65.0, -65.0]])
    mask = jnp.ones_like(observed, dtype=bool)
    kwargs = {
        "dt_ms": 1.0,
        "scale": 5.0,
        "threshold_mV": -20.0,
        "temperature_mV": 2.0,
    }

    matching_loss = term(matching, observed, mask, **kwargs)
    missing_loss = term(missing_spike, observed, mask, **kwargs)

    assert float(matching_loss[0]) < float(missing_loss[0])


def test_soft_firing_rate_does_not_count_a_stationary_near_threshold_plateau():
    registry = default_loss_registry()
    term = registry.get("soft_firing_rate_error")
    observed = jnp.asarray(
        [[-65.0, 20.0, -65.0, -65.0, 20.0, -65.0, -65.0]]
    )
    plateau = jnp.full_like(observed, -30.0)
    mask = jnp.ones_like(observed, dtype=bool)
    kwargs = {
        "dt_ms": 1.0,
        "scale": 5.0,
        "threshold_mV": -20.0,
        "temperature_mV": 2.0,
    }

    matching_loss = term(observed, observed, mask, **kwargs)
    plateau_loss = term(plateau, observed, mask, **kwargs)

    np.testing.assert_allclose(matching_loss, [0.0], atol=1e-12)
    assert float(plateau_loss[0]) > 0.0

    def scalar(offset):
        rising = plateau.at[0, 3].add(offset)
        return jnp.sum(term(rising, observed, mask, **kwargs))

    value, gradient = jax.value_and_grad(scalar)(jnp.asarray(2.0))
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert float(gradient) != 0.0


def test_observed_interspike_masks_follow_peak_to_next_threshold_definition():
    observed = np.asarray(
        [
            [
                -65.0,
                -65.0,
                5.0,
                30.0,
                -50.0,
                0.0,
                25.0,
                -55.0,
                10.0,
                20.0,
                -65.0,
                5.0,
            ],
            [-65.0] * 12,
        ]
    )
    stimulus = np.zeros_like(observed, dtype=bool)
    stimulus[:, 2:10] = True

    masks, valid = observed_interspike_masks(
        observed, stimulus, threshold_mV=-20.0
    )

    assert masks.shape == (2, 2, 12)
    np.testing.assert_array_equal(valid[0], [True, True])
    np.testing.assert_array_equal(valid[1], [False, False])
    np.testing.assert_array_equal(np.flatnonzero(masks[0, 0]), [3, 4])
    np.testing.assert_array_equal(np.flatnonzero(masks[0, 1]), [6, 7])
    assert not masks[..., 10:].any()


def test_soft_dblo_matches_mean_interspike_minimum_minus_rest_and_is_smooth():
    observed = jnp.asarray(
        [[-70.0, -70.0, 20.0, -50.0, 20.0, -40.0, -65.0, -65.0]]
    )
    predicted = observed.at[0, 3].set(-45.0).at[0, 5].set(-35.0)
    stimulus = jnp.asarray(
        [[False, False, True, True, True, True, True, True]]
    )
    baseline = jnp.asarray(
        [[True, True, False, False, False, False, False, False]]
    )
    intervals = jnp.zeros((1, 2, observed.shape[-1]), dtype=bool)
    intervals = intervals.at[0, 0, 3].set(True).at[0, 1, 5].set(True)
    valid = jnp.asarray([[True, True]])
    kwargs = {
        "baseline_mask": baseline,
        "interspike_masks": intervals,
        "interspike_valid": valid,
        "scale": 5.0,
        "temperature_mV": 1.0,
    }

    exact = soft_dblo_error(observed, observed, stimulus, **kwargs)
    shifted_troughs = soft_dblo_error(predicted, observed, stimulus, **kwargs)
    common_offset = soft_dblo_error(predicted + 7.0, observed, stimulus, **kwargs)

    np.testing.assert_allclose(exact, [0.0], atol=1e-12)
    np.testing.assert_allclose(shifted_troughs, [1.0], rtol=1e-6)
    np.testing.assert_allclose(common_offset, shifted_troughs, rtol=1e-6)

    def scalar(voltage):
        return jnp.sum(soft_dblo_error(voltage, observed, stimulus, **kwargs))

    value, gradient = jax.jit(jax.value_and_grad(scalar))(predicted)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()
    assert not np.allclose(np.asarray(gradient), 0.0)
    np.testing.assert_allclose(np.sum(gradient), 0.0, atol=1e-6)


def test_soft_dblo_is_finite_zero_without_two_observed_spikes():
    observed = jnp.asarray([[-70.0, -70.0, -60.0, -55.0]])
    predicted = observed + 10.0
    stimulus = jnp.asarray([[False, False, True, True]])
    baseline = jnp.asarray([[True, True, False, False]])
    intervals = jnp.zeros((1, 1, 4), dtype=bool)
    valid = jnp.asarray([[False]])

    def scalar(voltage):
        return jnp.sum(
            soft_dblo_error(
                voltage,
                observed,
                stimulus,
                baseline_mask=baseline,
                interspike_masks=intervals,
                interspike_valid=valid,
                scale=5.0,
                temperature_mV=1.0,
            )
        )

    value, gradient = jax.value_and_grad(scalar)(predicted)
    assert float(value) == 0.0
    np.testing.assert_array_equal(gradient, jnp.zeros_like(predicted))


def test_outside_spike_multiplier_is_continuous_and_counts_crossing_destination():
    low = -65.0
    high = 20.0
    outside = jnp.asarray(
        [[True, True, True, False, False, False, False, True, True, True]]
    )
    flat = jnp.full((1, 10), low)
    near_threshold_plateau = jnp.full((1, 10), -20.0)
    inside = flat.at[0, 3].set(high)
    one_outside = flat.at[0, 7].set(high)
    two_outside = one_outside.at[0, 1].set(high)
    kwargs = {"threshold_mV": -20.0, "temperature_mV": 0.1}

    flat_count = soft_upward_crossing_count(flat, outside, **kwargs)[0]
    plateau_count = soft_upward_crossing_count(
        near_threshold_plateau, outside, **kwargs
    )[0]
    inside_count = soft_upward_crossing_count(inside, outside, **kwargs)[0]
    one_count = soft_upward_crossing_count(one_outside, outside, **kwargs)[0]
    two_count = soft_upward_crossing_count(two_outside, outside, **kwargs)[0]

    assert np.isclose(float(flat_count), 0.0, atol=1e-8)
    assert np.isclose(float(plateau_count), 0.0, atol=1e-8)
    assert np.isclose(float(inside_count), 0.0, atol=1e-8)
    assert np.isclose(float(one_count), 1.0, atol=1e-6)
    assert np.isclose(float(two_count), 2.0, atol=1e-6)

    penalty = LossPenaltySpec(
        kind="soft_outside_stimulus_spike_multiplier",
        label="outside",
        factor_per_spike=1.2,
    )
    _, one_multiplier, _ = apply_multiplicative_penalties(
        jnp.asarray(10.0), {"outside": one_count}, (penalty,)
    )
    _, two_multiplier, _ = apply_multiplicative_penalties(
        jnp.asarray(10.0), {"outside": two_count}, (penalty,)
    )
    assert np.isclose(float(one_multiplier), 1.2, rtol=1e-6)
    assert np.isclose(float(two_multiplier), 1.44, rtol=1e-6)

    def penalized(peak):
        voltage = jnp.asarray([[low, peak, low]])
        mask = jnp.ones_like(voltage, dtype=bool)
        count = soft_upward_crossing_count(voltage, mask, **kwargs)[0]
        value, _, _ = apply_multiplicative_penalties(
            jnp.asarray(2.0), {"outside": count}, (penalty,)
        )
        return value

    value, gradient = jax.value_and_grad(penalized)(jnp.asarray(-20.0))
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert float(gradient) > 0.0


def test_outside_spike_multiplier_has_a_finite_numerical_ceiling():
    penalty = LossPenaltySpec(
        kind="soft_outside_stimulus_spike_multiplier",
        label="outside",
        factor_per_spike=1.2,
        maximum_multiplier=1e6,
    )
    value, multiplier, slopes = apply_multiplicative_penalties(
        jnp.asarray(3.0),
        {"outside": jnp.asarray(10_000.0)},
        (penalty,),
    )

    assert np.isfinite(float(value))
    assert float(multiplier) <= 1e6 * (1.0 + 1e-6)
    assert float(slopes["outside"]) == 0.0


def test_global_penalty_chain_coefficients_match_direct_gradient():
    penalty = LossPenaltySpec(
        kind="soft_outside_stimulus_spike_multiplier",
        label="outside",
        factor_per_spike=1.2,
    )

    def base_1(value):
        return value**2

    def base_2(value):
        return 2.0 * value**2

    def count_1(value):
        return jax.nn.sigmoid(value)

    def count_2(value):
        return jax.nn.sigmoid(2.0 * value)

    def direct(value):
        base = base_1(value) + base_2(value)
        count = count_1(value) + count_2(value)
        result, _, _ = apply_multiplicative_penalties(
            base, {"outside": count}, (penalty,)
        )
        return result

    value = jnp.asarray(0.7)
    base = base_1(value) + base_2(value)
    count = count_1(value) + count_2(value)
    _, multiplier, slopes = apply_multiplicative_penalties(
        base, {"outside": count}, (penalty,)
    )
    base_coefficient = multiplier
    count_coefficient = base * multiplier * slopes["outside"]
    bucketwise_gradient = (
        jax.grad(
            lambda item: (
                base_coefficient * base_1(item)
                + count_coefficient * count_1(item)
            )
        )(value)
        + jax.grad(
            lambda item: (
                base_coefficient * base_2(item)
                + count_coefficient * count_2(item)
            )
        )(value)
    )

    np.testing.assert_allclose(
        bucketwise_gradient,
        jax.grad(direct)(value),
        rtol=1e-6,
        atol=1e-6,
    )


def test_trainer_applies_one_global_multiplier_across_shape_buckets():
    penalty = LossPenaltySpec(
        kind="soft_outside_stimulus_spike_multiplier",
        label="outside",
        factor_per_spike=1.2,
    )
    component = LossComponentSpec(kind="voltage_mse", label="base")

    def prepared(key, base_function, count_function):
        def evaluate(value):
            base = base_function(value)
            count = count_function(value)
            return base, (
                jnp.asarray([[value]]),
                {"base": base},
                base,
                {"outside": count},
            )

        def penalty_gradient(value, base_coefficient, count_coefficients):
            return jax.grad(
                lambda item: (
                    base_coefficient * base_function(item)
                    + count_coefficients[0] * count_function(item)
                )
            )(value)

        return SimpleNamespace(
            bucket=SimpleNamespace(key=key),
            evaluate=evaluate,
            penalty_gradient=penalty_gradient,
        )

    first = prepared(
        (0.05, 3),
        lambda value: value**2,
        lambda value: jax.nn.sigmoid(value),
    )
    second = prepared(
        (0.05, 4),
        lambda value: 2.0 * value**2,
        lambda value: jax.nn.sigmoid(2.0 * value),
    )
    trainer = object.__new__(Trainer)
    trainer.fit = SimpleNamespace(
        penalties=(penalty,),
        components=(component,),
    )
    trainer._prepared = (first, second)

    value = jnp.asarray(0.7)
    evaluation = trainer._evaluate_with_penalties(value, gradient=True)
    total_loss, gradient = evaluation[:2]

    def direct(item):
        base = item**2 + 2.0 * item**2
        count = jax.nn.sigmoid(item) + jax.nn.sigmoid(2.0 * item)
        result, _, _ = apply_multiplicative_penalties(
            base, {"outside": count}, (penalty,)
        )
        return result

    np.testing.assert_allclose(total_loss, direct(value), rtol=1e-6)
    np.testing.assert_allclose(gradient, jax.grad(direct)(value), rtol=1e-6)
    assert np.isclose(sum(evaluation[2].values()), float(total_loss))
    assert np.isclose(
        evaluation[4]["base"],
        evaluation[6]["base_loss"] * evaluation[6]["loss_multiplier"],
    )


def test_protocol_filtered_component_is_normalized_across_shape_buckets():
    def record(protocol, size, weight):
        time = np.arange(size, dtype=float) * 0.05
        return TraceRecord(
            cell_id="cell",
            trace_id=protocol,
            protocol=protocol,
            voltage_mV=np.zeros(size),
            current_nA=np.zeros(size),
            time_ms=time,
            score_mask=np.ones(size, dtype=bool),
            dt_ms=0.05,
            initial_voltage_mV=0.0,
            weight=weight,
            metadata={"epoch_start_ms": 0.05, "epoch_stop_ms": 0.1},
            checksums={},
        )

    buckets = bucket_records(
        (
            record("depolarizing_step", 3, 0.5),
            record("hyperpolarizing_pulse", 4, 0.5),
        )
    )
    component = LossComponentSpec(
        kind="voltage_mse",
        label="hyper_only",
        protocols=("hyperpolarizing_pulse",),
        window="full_trace",
    )
    denominators = component_denominators((component,), buckets)
    registry = default_loss_registry()
    total = 0.0
    for bucket in buckets:
        objective = BucketObjective(
            (component,), bucket, denominators, registry
        )
        amplitude = 2.0 if bucket.records[0].protocol == "hyperpolarizing_pulse" else 1.0
        predicted = jnp.full_like(jnp.asarray(bucket.observed_mV), amplitude)
        value, contributions = objective(
            predicted, jnp.asarray(bucket.observed_mV)
        )
        assert "hyper_only" in contributions
        total += float(value)

    assert np.isclose(total, 4.0)

    global_denominators = component_denominators(
        (component,),
        buckets,
        renormalize_protocol_filtered=False,
    )
    global_total = 0.0
    for bucket in buckets:
        objective = BucketObjective(
            (component,), bucket, global_denominators, registry
        )
        amplitude = (
            2.0
            if bucket.records[0].protocol == "hyperpolarizing_pulse"
            else 1.0
        )
        predicted = jnp.full_like(jnp.asarray(bucket.observed_mV), amplitude)
        value, _ = objective(predicted, jnp.asarray(bucket.observed_mV))
        global_total += float(value)

    assert np.isclose(global_total, 2.0)


def test_lsu_global_protocol_weights_are_not_cancelled_by_component_filters():
    def record(trace_id, protocol):
        size = 3
        return TraceRecord(
            cell_id="cell",
            trace_id=trace_id,
            protocol=protocol,
            voltage_mV=np.zeros(size),
            current_nA=np.zeros(size),
            time_ms=np.arange(size, dtype=float),
            score_mask=np.ones(size, dtype=bool),
            dt_ms=1.0,
            initial_voltage_mV=-65.0,
            weight=0.0,
            metadata={"epoch_start_ms": 1.0, "epoch_stop_ms": 1.0},
            checksums={},
        )

    weighted = weight_records(
        (
            record("d1", "depolarizing_step"),
            record("d3", "depolarizing_step"),
            record("h1", "hyperpolarizing_pulse"),
            record("h3", "hyperpolarizing_pulse"),
        ),
        aggregation="protocol_mean",
        protocol_weights={
            "depolarizing_step": 0.7,
            "hyperpolarizing_pulse": 0.3,
        },
    )

    by_protocol = {
        protocol: [item.weight for item in weighted if item.protocol == protocol]
        for protocol in ("depolarizing_step", "hyperpolarizing_pulse")
    }
    np.testing.assert_allclose(by_protocol["depolarizing_step"], [0.35, 0.35])
    np.testing.assert_allclose(by_protocol["hyperpolarizing_pulse"], [0.15, 0.15])

    lsu = load_config(PROJECT / "configs/losses/LSU_1.yaml")
    denominators = component_denominators(
        lsu.fit.components,
        bucket_records(weighted),
        renormalize_protocol_filtered=(
            lsu.fit.renormalize_protocol_filtered_components
        ),
    )
    assert set(denominators.values()) == {1.0}

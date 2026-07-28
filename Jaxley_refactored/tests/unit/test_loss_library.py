from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_refactored.config import load_config
from jaxley_refactored.config.schema import LossComponentSpec
from jaxley_refactored.data import TraceRecord, bucket_records
from jaxley_refactored.fitting.losses import (
    BucketObjective,
    component_denominators,
    default_loss_registry,
)


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
            "depolarizing_voltage_plateau",
            "depolarizing_spike_waveform",
            "depolarizing_spike_derivative",
            "depolarizing_recovery_waveform",
            "depolarizing_recovery_derivative",
            "depolarizing_ahp_depth",
        ),
    }
    for filename, labels in expected.items():
        config = load_config(PROJECT / "configs/losses" / filename)
        assert tuple(component.label for component in config.fit.components) == labels


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
        "soft_minimum_voltage_error",
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

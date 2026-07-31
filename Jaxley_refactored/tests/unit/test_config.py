from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from jaxley_refactored.config import ConfigError, load_config
from jaxley_refactored.config.hashing import stable_hash
from jaxley_refactored.config.schema import AppConfig


PROJECT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT / "configs"
FULL_CONFIG = CONFIG_ROOT / "LSU_1_cma_adam.yaml"
HYPER_CONFIG = CONFIG_ROOT / "hyperpolarizing_only_cma_adam.yaml"


def test_config_directory_contains_only_two_standalone_yaml_files():
    config_files = {path for path in CONFIG_ROOT.rglob("*") if path.is_file()}

    assert config_files == {FULL_CONFIG, HYPER_CONFIG}
    for path in config_files:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        assert isinstance(raw, dict)
        assert "extends" not in raw


def test_hybrid_lsu_config_is_standalone_and_explicit():
    path = FULL_CONFIG
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    assert "extends" not in raw
    assert set(raw) == {
        "schema_version",
        "model",
        "protocol",
        "dataset",
        "fit",
        "runtime",
        "search",
        "output",
    }
    assert raw["dataset"]["selection"]["trace_indices"] == [1, 3]
    assert raw["dataset"]["selection"]["validation_trace_indices"] == [2, 4]
    assert raw["dataset"]["resampling"]["target_dt_ms"] == 0.1
    assert raw["dataset"]["simulation_window"] == {
        "post_stimulus_ms": 500.0,
        "post_stimulus_ms_by_protocol": {
            "hyperpolarizing_pulse": 100.0,
        },
    }
    assert len(raw["fit"]["objective"]["components"]) == 19
    assert len(raw["fit"]["objective"]["penalties"]) == 2
    assert raw["fit"]["checkpoint"]["reject_incompatible_hashes"] is True
    assert raw["output"]["provenance"]["include_resolved_config"] is True

    config = load_config(path)
    assert config.dataset.trace_indices == (1, 3)
    assert config.dataset.validation_trace_indices == (2, 4)
    assert config.model.parameters.bound_expansion_factor == 1.0
    assert config.fit.initialization.mode == "jittered_reference"
    assert config.fit.initialization.scale == 0.10
    assert config.fit.optimizer.line_search.enabled is True
    assert config.runtime.backend == "cpu"
    assert config.runtime.precision == "float64"
    assert config.runtime.solver == "bwd_euler"
    assert config.runtime.voltage_solver == "jaxley.dhs"
    assert config.search.strategy == "hybrid"
    assert config.search.global_search.population_size == 40
    assert config.search.global_search.generations == 100
    assert config.search.global_search.parent_fraction == 0.30
    assert config.search.global_search.elites == 10
    assert config.search.local_exploration.epochs == 50
    assert config.search.local_exploration.backtracking is False
    assert config.search.local_refinement.epochs == 150
    assert config.search.local_refinement.backtracking is True
    assert config.output.root == PROJECT / "runs"


@pytest.mark.parametrize("path", (FULL_CONFIG, HYPER_CONFIG))
def test_supported_configs_resolve_paths_and_are_stable(path):
    config = load_config(path)

    assert config.dataset.cell_id == "m20240527cd"
    assert config.dataset.root.is_absolute()
    assert config.dataset.manifest.is_absolute()
    assert config.output.root.is_absolute()
    assert config.model.morphology.provider == "hoc_live"
    assert config.model.distributions.preset == "combe2023_cch_driven"
    assert stable_hash(asdict(config)) == stable_hash(asdict(config))


def test_optional_post_stimulus_simulation_window_is_validated():
    default = AppConfig.from_mapping({})
    assert default.dataset.simulation_post_ms is None
    assert default.dataset.simulation_post_ms_by_protocol == {}

    configured = AppConfig.from_mapping(
        {
            "dataset": {
                "simulation_window": {
                    "post_stimulus_ms": 500.0,
                    "post_stimulus_ms_by_protocol": {
                        "hyperpolarizing_pulse": 150.0,
                    },
                }
            }
        }
    )
    assert configured.dataset.simulation_post_ms == 500.0
    assert configured.dataset.simulation_post_ms_by_protocol == {
        "hyperpolarizing_pulse": 150.0,
    }
    assert stable_hash(asdict(configured)) != stable_hash(asdict(default))

    with pytest.raises(ConfigError, match="post_stimulus_ms"):
        AppConfig.from_mapping(
            {
                "dataset": {
                    "simulation_window": {
                        "post_stimulus_ms": 0.0,
                    }
                }
            }
        )

    with pytest.raises(ConfigError, match="hyperpolarizing_pulse"):
        AppConfig.from_mapping(
            {
                "dataset": {
                    "simulation_window": {
                        "post_stimulus_ms_by_protocol": {
                            "hyperpolarizing_pulse": -1.0,
                        }
                    }
                }
            }
        )


def test_unknown_root_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        AppConfig.from_mapping({"schema_version": 1, "typo": True})


def test_backtracking_optimizer_configuration_is_validated():
    config = AppConfig.from_mapping(
        {
            "fit": {
                "optimizer": {
                    "learning_rate": 0.01,
                    "line_search": {
                        "enabled": True,
                        "minimum_learning_rate": 0.0001,
                        "maximum_learning_rate": 0.1,
                        "maximum_trials": 6,
                    },
                }
            }
        }
    )

    assert config.fit.optimizer.line_search.enabled
    assert (
        config.fit.optimizer.line_search.minimum_learning_rate
        <= config.fit.optimizer.learning_rate
        <= config.fit.optimizer.line_search.maximum_learning_rate
    )
    assert config.fit.optimizer.line_search.maximum_trials > 0

    with pytest.raises(ConfigError, match="reduction_factor"):
        AppConfig.from_mapping(
            {
                "fit": {
                    "optimizer": {
                        "line_search": {
                            "enabled": True,
                            "reduction_factor": 1.0,
                        }
                    }
                }
            }
        )


def test_cma_parent_fraction_defaults_and_validation():
    default = AppConfig.from_mapping(
        {"search": {"strategy": "hybrid"}}
    )
    assert default.search.global_search.parent_fraction == 0.5

    configured = AppConfig.from_mapping(
        {
            "search": {
                "strategy": "hybrid",
                "global": {"parent_fraction": 0.30},
            }
        }
    )
    assert configured.search.global_search.parent_fraction == 0.30

    for invalid in (0.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ConfigError, match="parent_fraction"):
            AppConfig.from_mapping(
                {
                    "search": {
                        "strategy": "hybrid",
                        "global": {"parent_fraction": invalid},
                    }
                }
            )


def test_hyperpolarizing_only_config_has_isolated_data_loss_and_output():
    config = load_config(HYPER_CONFIG)

    assert config.dataset.segments == ("hyperpolarizing_pulse",)
    assert config.dataset.trace_indices == (1, 3)
    assert config.dataset.simulation_post_ms == 100.0
    assert config.dataset.score_pre_ms == 50.0
    assert config.dataset.score_post_ms == 100.0
    assert config.fit.protocol_weights == {
        "depolarizing_step": 0.0,
        "hyperpolarizing_pulse": 1.0,
    }
    assert config.fit.penalties == ()
    assert len(config.fit.components) == 3
    trough, waveform, derivative = config.fit.components
    assert trough.kind == "soft_trough_depth_error"
    assert trough.label == "hyperpolarizing_trough_depth"
    assert trough.protocols == ("hyperpolarizing_pulse",)
    assert trough.window == "stimulus"
    assert trough.weight == 4.0
    assert trough.scale == 1.0
    assert trough.temperature_mV == 0.5
    assert waveform.kind == "voltage_mse"
    assert waveform.label == "hyperpolarizing_waveform_mse"
    assert waveform.protocols == ("hyperpolarizing_pulse",)
    assert waveform.window == "score"
    assert waveform.weight == 1.0
    assert waveform.scale == 1.0
    assert derivative.kind == "derivative_mse"
    assert derivative.label == "hyperpolarizing_derivative_mse"
    assert derivative.protocols == ("hyperpolarizing_pulse",)
    assert derivative.window == "score"
    assert derivative.weight == 1.0
    assert derivative.scale == 1.0
    assert config.output.root == PROJECT / "runs_hyper"


def test_outside_spike_penalty_configuration_is_validated():
    default_config = AppConfig.from_mapping(
        {
            "fit": {
                "objective": {
                    "penalties": [
                        {"kind": "soft_outside_stimulus_spike_multiplier"}
                    ]
                }
            }
        }
    )
    assert default_config.fit.penalties[0].factor_per_spike == 1.1

    config = AppConfig.from_mapping(
        {
            "fit": {
                "objective": {
                    "penalties": [
                        {
                            "kind": "soft_outside_stimulus_spike_multiplier",
                            "factor_per_spike": 1.2,
                            "maximum_multiplier": 1e12,
                        }
                    ]
                }
            }
        }
    )
    assert config.fit.penalties[0].factor_per_spike == 1.2

    with pytest.raises(ConfigError, match="factor_per_spike"):
        AppConfig.from_mapping(
            {
                "fit": {
                    "objective": {
                        "penalties": [
                            {
                                "kind": "soft_outside_stimulus_spike_multiplier",
                                "factor_per_spike": 1.0,
                            }
                        ]
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "field",
    (
        "kernel_tau_ms",
        "width_scale_ms",
        "upstroke_scale_mV_per_ms",
        "repolarization_scale_mV_per_ms",
        "slope_temperature_mV_per_ms",
        "amplitude_gate_mV",
        "amplitude_gate_temperature_mV",
    ),
)
@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_feature_loss_positive_fields_must_be_finite(field, value):
    with pytest.raises(ConfigError, match=field):
        AppConfig.from_mapping(
            {
                "fit": {
                    "objective": {
                        "components": [
                            {
                                "kind": "soft_spike_width_slope_error",
                                field: value,
                            }
                        ]
                    }
                }
            }
        )

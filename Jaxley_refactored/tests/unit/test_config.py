from dataclasses import asdict
from pathlib import Path

import pytest

from jaxley_refactored.config import ConfigError, load_config
from jaxley_refactored.config.hashing import stable_hash
from jaxley_refactored.config.schema import AppConfig


PROJECT = Path(__file__).resolve().parents[2]


def test_default_config_resolves_paths_and_is_stable():
    config = load_config(PROJECT / "configs/runtimes/cpu_x64.yaml")

    assert config.dataset.cell_id == "m20240527cd"
    assert config.dataset.root.is_absolute()
    assert config.model.morphology.provider == "hoc_live"
    assert config.model.distributions.preset == "combe2023_cch_driven"
    assert stable_hash(asdict(config)) == stable_hash(asdict(config))


def test_unknown_root_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        AppConfig.from_mapping({"schema_version": 1, "typo": True})


def test_backtracking_optimizer_configuration_is_validated():
    config = load_config(PROJECT / "configs/optimizers/adam_backtracking.yaml")

    assert config.fit.optimizer.line_search.enabled
    assert config.fit.optimizer.learning_rate == 0.01
    assert config.fit.optimizer.line_search.maximum_trials == 6

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

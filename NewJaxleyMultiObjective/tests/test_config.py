from newjaxley_multiobjective.config import FitnessWindowConfig, ObjectiveConfig


def test_fitness_windows_are_protocol_specific():
    config = FitnessWindowConfig.from_mapping(
        {
            "depolarizing": {"pre_ms": 200.0, "post_ms": 600.0},
            "hyperpolarizing": {"pre_ms": 200.0, "post_ms": 200.0},
        }
    )

    assert config.for_protocol("depolarizing_step") == (200.0, 600.0)
    assert config.for_protocol("hyperpolarizing_pulse") == (200.0, 200.0)


def test_ap_count_only_mode_has_one_objective():
    config = ObjectiveConfig.from_mapping(
        {"mode": "ap_count_only", "features": ["ap_count"]}
    )

    assert config.labels == ("ap_count",)

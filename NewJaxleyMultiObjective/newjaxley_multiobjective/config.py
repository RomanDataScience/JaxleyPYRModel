"""Configuration for the small MOCMA pipeline.

The existing refactored application owns the model/data configuration. This
module adds only the multi-objective settings and loads that application
configuration without importing JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


OBJECTIVE_LABELS = (
    "resting_membrane_potential",
    "average_fahp",
    "time_mahp",
    "amplitude_mahp",
    "ap_count",
    "last_stabilized_isi",
    "last_ap_half_width",
    "peak_input_resistance",
    "steady_state_input_resistance",
    "sag",
    "trajectory_overlap",
)

DEFAULT_SCALES = {
    "resting_membrane_potential": 2.0,
    "average_fahp": 2.0,
    "time_mahp": 20.0,
    "amplitude_mahp": 2.0,
    "ap_count": 1.0,
    "last_stabilized_isi": 10.0,
    "last_ap_half_width": 0.5,
    "peak_input_resistance": 20.0,
    "steady_state_input_resistance": 20.0,
    "sag": 0.1,
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_ints(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain positive integers") from error
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class FeatureConfig:
    threshold_mV: float = -20.0
    refractory_ms: float = 2.0
    smoothing_ms: float = 0.2
    baseline_window_ms: float = 30.0
    fahp_search_ms: float = 20.0
    post_step_analysis_ms: float = 700.0
    hyper_peak_window_ms: float = 20.0
    hyper_steady_state_fraction: float = 0.2

    @classmethod
    def from_mapping(cls, value: Any) -> "FeatureConfig":
        data = _mapping(value, "multi_objective.feature_detection")
        fraction = float(data.get("hyper_steady_state_fraction", 0.2))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("hyper_steady_state_fraction must be in (0, 1]")
        return cls(
            threshold_mV=float(data.get("threshold_mV", -20.0)),
            refractory_ms=_positive(data.get("refractory_ms", 2.0), "refractory_ms"),
            smoothing_ms=float(data.get("smoothing_ms", 0.2)),
            baseline_window_ms=_positive(data.get("baseline_window_ms", 30.0), "baseline_window_ms"),
            fahp_search_ms=_positive(data.get("fahp_search_ms", 20.0), "fahp_search_ms"),
            post_step_analysis_ms=_positive(data.get("post_step_analysis_ms", 700.0), "post_step_analysis_ms"),
            hyper_peak_window_ms=_positive(data.get("hyper_peak_window_ms", 20.0), "hyper_peak_window_ms"),
            hyper_steady_state_fraction=fraction,
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    labels: tuple[str, ...] = OBJECTIVE_LABELS
    scales: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SCALES))
    invalid_feature_penalty: float = 1.0e6
    overlap_sigma_mV: float = 3.0
    overlap_window_weights: Mapping[str, float] | None = None
    tie_tolerance: float = 1.0e-12

    def __post_init__(self):
        if self.overlap_window_weights is None:
            object.__setattr__(
                self,
                "overlap_window_weights",
                {"baseline": 1.0, "stimulus": 1.0, "recovery": 1.0, "first_100ms_spiking": 2.0},
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "ObjectiveConfig":
        data = _mapping(value, "multi_objective.objectives")
        labels = tuple(data.get("features", OBJECTIVE_LABELS))
        if labels != OBJECTIVE_LABELS:
            raise ValueError(
                "The objective feature order must contain exactly the eleven "
                "supported labels: " + ", ".join(OBJECTIVE_LABELS)
            )
        scales = dict(DEFAULT_SCALES)
        scales.update({str(k): _positive(v, f"objective scale {k}") for k, v in _mapping(data.get("scales"), "objectives.scales").items()})
        weights = dict(
            {"baseline": 1.0, "stimulus": 1.0, "recovery": 1.0, "first_100ms_spiking": 2.0}
        )
        weights.update({str(k): _positive(v, f"overlap window weight {k}") for k, v in _mapping(data.get("overlap_window_weights"), "objectives.overlap_window_weights").items()})
        return cls(
            labels=labels,
            scales=scales,
            invalid_feature_penalty=_positive(data.get("invalid_feature_penalty", 1.0e6), "invalid_feature_penalty"),
            overlap_sigma_mV=_positive(data.get("overlap_sigma_mV", 3.0), "overlap_sigma_mV"),
            overlap_window_weights=weights,
            tie_tolerance=float(data.get("tie_tolerance", 1.0e-12)),
        )


@dataclass(frozen=True)
class FitnessWindowConfig:
    """Time-course window around each current stimulus, in milliseconds."""

    depolarizing_pre_ms: float = 200.0
    depolarizing_post_ms: float = 600.0
    hyperpolarizing_pre_ms: float = 200.0
    hyperpolarizing_post_ms: float = 200.0

    @classmethod
    def from_mapping(cls, value: Any) -> "FitnessWindowConfig":
        data = _mapping(value, "multi_objective.fitness_windows")
        depolarizing = _mapping(data.get("depolarizing"), "fitness_windows.depolarizing")
        hyperpolarizing = _mapping(data.get("hyperpolarizing"), "fitness_windows.hyperpolarizing")
        return cls(
            depolarizing_pre_ms=_positive(depolarizing.get("pre_ms", 200.0), "depolarizing pre_ms"),
            depolarizing_post_ms=_positive(depolarizing.get("post_ms", 600.0), "depolarizing post_ms"),
            hyperpolarizing_pre_ms=_positive(hyperpolarizing.get("pre_ms", 200.0), "hyperpolarizing pre_ms"),
            hyperpolarizing_post_ms=_positive(hyperpolarizing.get("post_ms", 200.0), "hyperpolarizing post_ms"),
        )

    def for_protocol(self, protocol: str) -> tuple[float, float]:
        if protocol == "depolarizing_step":
            return self.depolarizing_pre_ms, self.depolarizing_post_ms
        return self.hyperpolarizing_pre_ms, self.hyperpolarizing_post_ms


@dataclass(frozen=True)
class PipelineConfig:
    source_path: Path
    base_config_path: Path
    app_config: Any
    feature: FeatureConfig
    objectives: ObjectiveConfig
    fitness_windows: FitnessWindowConfig
    optimization_trace_indices: tuple[int, ...]
    test_trace_indices: tuple[int, ...]
    seed: int
    population_size: int
    trials: int
    study_name: str
    storage: str
    resume: bool
    heartbeat_interval_s: int
    grace_period_s: int
    output_root: Path
    all_traces: bool


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base_text = raw.get("base_config")
    if not base_text:
        raise ValueError("Pipeline config requires base_config")
    base_path = Path(base_text)
    if not base_path.is_absolute():
        base_path = (source.parent / base_path).resolve()

    # Importing the refactored config is intentionally delayed until this
    # function is called; it is JAX-free and can run before runtime bootstrap.
    from jaxley_refactored.config import load_config

    app_config = load_config(base_path)
    multi = _mapping(raw.get("multi_objective"), "multi_objective")
    sampler = str(multi.get("sampler", "mocma"))
    if sampler != "mocma":
        raise ValueError("Only the mocma sampler is supported")
    seed = int(multi.get("seed", app_config.runtime.seed))
    population_size = int(multi.get("population_size", 0))
    trials = int(multi.get("trials", 0))
    if seed < 0 or population_size < 2 or trials <= 0:
        raise ValueError("seed must be nonnegative, population_size >= 2, trials > 0")

    from dataclasses import replace

    trace_selection = _mapping(multi.get("trace_selection"), "multi_objective.trace_selection")
    optimization_trace_indices = _positive_ints(
        trace_selection.get("optimization_indices", (2, 4)),
        "trace_selection.optimization_indices",
    )
    test_trace_indices = _positive_ints(
        trace_selection.get("test_indices", (1, 3)),
        "trace_selection.test_indices",
    )
    if set(optimization_trace_indices) & set(test_trace_indices):
        raise ValueError("Optimization and test trace indices must be disjoint")

    fitness_windows = FitnessWindowConfig.from_mapping(multi.get("fitness_windows"))
    dataset = app_config.dataset
    if "cell_id" in multi:
        dataset = replace(dataset, cell_id=str(multi["cell_id"]))
    dataset = replace(
        dataset,
        traces=(),
        trace_indices=optimization_trace_indices,
        validation_trace_indices=test_trace_indices,
        score_pre_ms=min(
            fitness_windows.depolarizing_pre_ms,
            fitness_windows.hyperpolarizing_pre_ms,
        ),
        score_post_ms=max(
            fitness_windows.depolarizing_post_ms,
            fitness_windows.hyperpolarizing_post_ms,
        ),
        # Keep the complete segmented recording. Some hyperpolarizing
        # segments contain only 100 ms after the pulse, so the evaluator can
        # clip the requested 200 ms window to what is actually available.
        simulation_post_ms=None,
        simulation_post_ms_by_protocol={},
    )
    runtime = replace(app_config.runtime, seed=seed)
    output_value = raw.get("output_root", "runs")
    output_root = Path(output_value)
    if not output_root.is_absolute():
        output_root = (source.parent / output_root).resolve()
    app_config = replace(app_config, dataset=dataset, runtime=runtime)

    storage = multi.get("storage", "auto")
    if storage is None:
        storage = "auto"
    heartbeat = int(multi.get("heartbeat_interval_s", 60))
    grace_period = int(multi.get("grace_period_s", max(heartbeat * 5, 300)))
    if heartbeat <= 0 or grace_period <= 0:
        raise ValueError("heartbeat_interval_s and grace_period_s must be positive")
    return PipelineConfig(
        source_path=source,
        base_config_path=base_path,
        app_config=app_config,
        feature=FeatureConfig.from_mapping(multi.get("feature_detection")),
        objectives=ObjectiveConfig.from_mapping(multi.get("objectives")),
        fitness_windows=fitness_windows,
        optimization_trace_indices=optimization_trace_indices,
        test_trace_indices=test_trace_indices,
        seed=seed,
        population_size=population_size,
        trials=trials,
        study_name=str(multi.get("study_name", "mocma_features")),
        storage=str(storage),
        resume=bool(multi.get("resume", True)),
        heartbeat_interval_s=heartbeat,
        grace_period_s=grace_period,
        output_root=output_root,
        all_traces=False,
    )

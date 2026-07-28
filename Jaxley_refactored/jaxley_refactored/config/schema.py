"""Frozen application configuration with dependency-free validation.

The project intentionally keeps configuration validation independent of JAX.
That allows the CLI bootstrap to select a backend before importing JAX and
gives cluster jobs useful errors even on nodes without accelerator libraries.

Each class owns validation for one concern. New providers or optimizer options
can be added without changing unrelated model or data code (open/closed
principle), while callers depend on these small value objects rather than YAML
dictionaries (dependency inversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when configuration cannot be interpreted unambiguously."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping.")
    return value


def _strict(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {where} key(s): {', '.join(unknown)}")


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ConfigError(f"{where} must be a string or list of strings.")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ConfigError(f"{where} cannot contain empty names.")
    return result


def _positive_ints(value: Any, where: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{where} must be a list of positive integers.")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ConfigError(f"{where} must contain only positive integers.")
        try:
            integer = int(item)
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"{where} must contain only positive integers."
            ) from error
        if integer != item or integer <= 0:
            raise ConfigError(f"{where} must contain only positive integers.")
        result.append(integer)
    if len(result) != len(set(result)):
        raise ConfigError(f"{where} cannot contain duplicate indices.")
    return tuple(result)


def _positive(value: Any, where: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ConfigError(f"{where} must be positive.")
    return result


def _at_least_one(value: Any, where: str) -> float:
    result = float(value)
    if result < 1.0:
        raise ConfigError(f"{where} must be at least 1.")
    return result


@dataclass(frozen=True)
class MorphologySpec:
    """Static morphology choice; changes require rebuilding the model."""

    provider: str = "hoc_live"
    path: Path | None = None
    d_lambda: float = 0.3
    frequency_hz: float = 100.0
    required_groups: tuple[str, ...] = ("soma", "axon", "basal", "apical")
    reject_unclassified: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "MorphologySpec":
        data = _mapping(value, "model.morphology")
        _strict(
            data,
            {
                "provider",
                "path",
                "source_provenance",
                "grouping",
                "root",
                "discretization",
            },
            "model.morphology",
        )
        provider = str(data.get("provider", "hoc_live"))
        if provider not in {"hoc_live", "hoc_artifact", "exact_hoc_artifact", "swc"}:
            raise ConfigError(f"Unsupported morphology provider: {provider}")
        if provider == "exact_hoc_artifact":
            provider = "hoc_artifact"

        grouping = _mapping(data.get("grouping"), "model.morphology.grouping")
        _strict(
            grouping,
            {"strategy", "required_groups", "reject_unclassified"},
            "model.morphology.grouping",
        )
        discretization = _mapping(
            data.get("discretization"), "model.morphology.discretization"
        )
        _strict(
            discretization,
            {"strategy", "d_lambda", "frequency_hz"},
            "model.morphology.discretization",
        )
        strategy = discretization.get("strategy", "d_lambda")
        if strategy != "d_lambda":
            raise ConfigError("Only d_lambda discretization is currently supported.")

        path = data.get("path")
        return cls(
            provider=provider,
            path=Path(path) if path else None,
            d_lambda=_positive(
                discretization.get("d_lambda", 0.3),
                "model.morphology.discretization.d_lambda",
            ),
            frequency_hz=_positive(
                discretization.get("frequency_hz", 100.0),
                "model.morphology.discretization.frequency_hz",
            ),
            required_groups=_strings(
                grouping.get(
                    "required_groups", ("soma", "axon", "basal", "apical")
                ),
                "model.morphology.grouping.required_groups",
            ),
            reject_unclassified=bool(
                grouping.get("reject_unclassified", True)
            ),
        )


@dataclass(frozen=True)
class MechanismSpec:
    """Static mechanism selection and calcium-diffusion policy."""

    include: tuple[str, ...] = ("all_from_preset",)
    exclude: tuple[str, ...] = ()
    calcium_diffusion: bool = True
    calcium_axial_diffusion: float = 0.22

    @classmethod
    def from_mapping(cls, value: Any) -> "MechanismSpec":
        data = _mapping(value, "model.mechanisms")
        _strict(
            data,
            {"preset", "include", "exclude", "overrides", "calcium_diffusion"},
            "model.mechanisms",
        )
        diffusion = _mapping(
            data.get("calcium_diffusion"), "model.mechanisms.calcium_diffusion"
        )
        _strict(
            diffusion,
            {"enabled", "state", "axial_diffusion"},
            "model.mechanisms.calcium_diffusion",
        )
        include = _strings(
            data.get("include", "all_from_preset"), "model.mechanisms.include"
        )
        return cls(
            include=include,
            exclude=_strings(data.get("exclude"), "model.mechanisms.exclude"),
            calcium_diffusion=bool(diffusion.get("enabled", True)),
            calcium_axial_diffusion=float(diffusion.get("axial_diffusion", 0.22)),
        )


@dataclass(frozen=True)
class ParameterSelection:
    """How the parameter catalog is filtered for a fit."""

    include_tags: tuple[str, ...] = ("conductance", "passive")
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    value_overrides: Mapping[str, float] = field(default_factory=dict)
    bound_expansion_factor: float = 1.0

    @classmethod
    def from_mapping(cls, value: Any) -> "ParameterSelection":
        data = _mapping(value, "model.parameters")
        _strict(
            data,
            {"catalog", "value_overrides", "bound_expansion_factor", "fit"},
            "model.parameters",
        )
        fit = _mapping(data.get("fit"), "model.parameters.fit")
        _strict(fit, {"include_tags", "include", "exclude"}, "model.parameters.fit")
        overrides = {
            str(key): float(item)
            for key, item in _mapping(
                data.get("value_overrides"), "model.parameters.value_overrides"
            ).items()
        }
        return cls(
            include_tags=_strings(
                fit.get("include_tags", ("conductance", "passive")),
                "model.parameters.fit.include_tags",
            ),
            include=_strings(fit.get("include"), "model.parameters.fit.include"),
            exclude=_strings(fit.get("exclude"), "model.parameters.fit.exclude"),
            value_overrides=overrides,
            bound_expansion_factor=_at_least_one(
                data.get("bound_expansion_factor", 1.0),
                "model.parameters.bound_expansion_factor",
            ),
        )


@dataclass(frozen=True)
class DistributionSpec:
    """Spatial profile family and coefficient overrides.

    The profile *family* is static and therefore part of the model signature.
    Numeric overrides are resolved through the parameter catalog, allowing the
    same coefficients to be fixed for simulation or selected for fitting.
    """

    preset: str = "combe2023_cch_driven"
    overrides: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any) -> "DistributionSpec":
        data = _mapping(value, "model.distributions")
        _strict(data, {"preset", "overrides"}, "model.distributions")
        preset = str(data.get("preset", "combe2023_cch_driven"))
        if preset != "combe2023_cch_driven":
            raise ConfigError(f"Unsupported distribution preset: {preset}")
        return cls(
            preset=preset,
            overrides={
                str(key): float(item)
                for key, item in _mapping(
                    data.get("overrides"), "model.distributions.overrides"
                ).items()
            },
        )


@dataclass(frozen=True)
class ModelSpec:
    """Complete static model recipe."""

    model_id: str
    morphology: MorphologySpec
    profile_mode: str
    mechanisms: MechanismSpec
    distributions: DistributionSpec
    parameters: ParameterSelection

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelSpec":
        data = _mapping(value, "model")
        _strict(
            data,
            {
                "id",
                "morphology",
                "profile",
                "mechanisms",
                "distributions",
                "parameters",
            },
            "model",
        )
        profile = _mapping(data.get("profile"), "model.profile")
        _strict(
            profile,
            {
                "mode",
                "id",
                "assignment_distance_feature",
                "topology_during_fit",
                "fit_parameterization",
            },
            "model.profile",
        )
        mode = str(profile.get("mode", "exact_hoc_frozen_grid"))
        if mode not in {"exact_hoc_frozen_grid", "rule_based_final_centers"}:
            raise ConfigError(f"Unsupported model.profile.mode: {mode}")
        return cls(
            model_id=str(data.get("id", "combe2023")),
            morphology=MorphologySpec.from_mapping(data.get("morphology")),
            profile_mode=mode,
            mechanisms=MechanismSpec.from_mapping(data.get("mechanisms")),
            distributions=DistributionSpec.from_mapping(data.get("distributions")),
            parameters=ParameterSelection.from_mapping(data.get("parameters")),
        )


@dataclass(frozen=True)
class SiteSpec:
    group: str = "soma"
    branch: int = 0
    location: float = 0.5
    state: str = "v"

    @classmethod
    def from_mapping(cls, value: Any, where: str) -> "SiteSpec":
        data = _mapping(value, where)
        _strict(data, {"group", "branch", "location", "state"}, where)
        location = float(data.get("location", 0.5))
        if not 0.0 <= location <= 1.0:
            raise ConfigError(f"{where}.location must be in [0, 1].")
        return cls(
            group=str(data.get("group", "soma")),
            branch=int(data.get("branch", 0)),
            location=location,
            state=str(data.get("state", "v")),
        )


@dataclass(frozen=True)
class ProtocolSpec:
    injection_site: SiteSpec
    recording_site: SiteSpec
    initial_state_mode: str = "observed_first_sample"
    fixed_voltage_mV: float = -71.9879
    alignment: str = "prefix"

    @classmethod
    def from_mapping(cls, value: Any) -> "ProtocolSpec":
        data = _mapping(value, "protocol")
        _strict(
            data,
            {"injection_site", "recording_sites", "initial_state", "alignment"},
            "protocol",
        )
        recordings = data.get("recording_sites") or [{}]
        if not isinstance(recordings, Sequence) or len(recordings) != 1:
            raise ConfigError("Exactly one recording site is currently supported.")
        initial = _mapping(data.get("initial_state"), "protocol.initial_state")
        _strict(
            initial,
            {"mode", "fixed_voltage_mV"},
            "protocol.initial_state",
        )
        mode = str(initial.get("mode", "observed_first_sample"))
        if mode not in {"observed_first_sample", "fixed"}:
            raise ConfigError(f"Unsupported initial-state mode: {mode}")
        alignment = str(data.get("alignment", "prefix"))
        if alignment not in {"prefix", "drop_initial"}:
            raise ConfigError(f"Unsupported sample alignment: {alignment}")
        return cls(
            injection_site=SiteSpec.from_mapping(
                data.get("injection_site"), "protocol.injection_site"
            ),
            recording_site=SiteSpec.from_mapping(
                recordings[0], "protocol.recording_sites[0]"
            ),
            initial_state_mode=mode,
            fixed_voltage_mV=float(initial.get("fixed_voltage_mV", -71.9879)),
            alignment=alignment,
        )


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    manifest: Path
    cell_id: str = "m20240527cd"
    traces: tuple[str, ...] = ("*",)
    trace_indices: tuple[int, ...] = ()
    segments: tuple[str, ...] = (
        "depolarizing_step",
        "hyperpolarizing_pulse",
    )
    target_dt_ms: float = 0.05
    current_scale_to_nA: float = 1e-3
    score_pre_ms: float = 100.0
    score_post_ms: float = 800.0

    @classmethod
    def from_mapping(cls, value: Any) -> "DatasetSpec":
        data = _mapping(value, "dataset")
        _strict(
            data,
            {
                "provider",
                "root",
                "manifest",
                "cell_id",
                "selection",
                "units",
                "validation",
                "resampling",
                "score_windows",
            },
            "dataset",
        )
        if data.get("provider", "segmented_current_clamp") != "segmented_current_clamp":
            raise ConfigError("Only segmented_current_clamp datasets are supported.")
        selection = _mapping(data.get("selection"), "dataset.selection")
        _strict(
            selection,
            {"traces", "trace_indices", "segments"},
            "dataset.selection",
        )
        units = _mapping(data.get("units"), "dataset.units")
        resampling = _mapping(data.get("resampling"), "dataset.resampling")
        windows = _mapping(data.get("score_windows"), "dataset.score_windows")
        current_unit = str(units.get("current_on_disk", "pA"))
        scales = {"pA": 1e-3, "nA": 1.0}
        if current_unit not in scales:
            raise ConfigError("dataset.units.current_on_disk must be pA or nA.")
        root = Path(data.get("root", ""))
        manifest = Path(data.get("manifest", "segment_metadata.csv"))
        traces = _strings(
            selection.get("traces", () if "trace_indices" in selection else ("*",)),
            "dataset.selection.traces",
        )
        trace_indices = _positive_ints(
            selection.get("trace_indices"), "dataset.selection.trace_indices"
        )
        if trace_indices and traces:
            raise ConfigError(
                "dataset.selection.traces and trace_indices are mutually exclusive."
            )
        return cls(
            root=root,
            manifest=manifest,
            cell_id=str(data.get("cell_id", "m20240527cd")),
            traces=traces,
            trace_indices=trace_indices,
            segments=_strings(
                selection.get(
                    "segments",
                    ("depolarizing_step", "hyperpolarizing_pulse"),
                ),
                "dataset.selection.segments",
            ),
            target_dt_ms=_positive(
                resampling.get("target_dt_ms", 0.05),
                "dataset.resampling.target_dt_ms",
            ),
            current_scale_to_nA=scales[current_unit],
            score_pre_ms=float(windows.get("pre_ms", 100.0)),
            score_post_ms=float(windows.get("post_ms", 800.0)),
        )


@dataclass(frozen=True)
class LineSearchSpec:
    """Policy for accepting an Adam direction only when it improves the loss."""

    enabled: bool = False
    reduction_factor: float = 0.5
    growth_factor: float = 1.2
    minimum_learning_rate: float = 1e-5
    maximum_learning_rate: float = 5e-2
    maximum_trials: int = 6

    @classmethod
    def from_mapping(cls, value: Any, learning_rate: float) -> "LineSearchSpec":
        data = _mapping(value, "fit.optimizer.line_search")
        _strict(
            data,
            {
                "enabled",
                "reduction_factor",
                "growth_factor",
                "minimum_learning_rate",
                "maximum_learning_rate",
                "maximum_trials",
                "require_loss_decrease",
            },
            "fit.optimizer.line_search",
        )
        reduction = float(data.get("reduction_factor", 0.5))
        growth = float(data.get("growth_factor", 1.2))
        minimum = _positive(
            data.get("minimum_learning_rate", 1e-5),
            "fit.optimizer.line_search.minimum_learning_rate",
        )
        maximum = _positive(
            data.get("maximum_learning_rate", max(5e-2, learning_rate)),
            "fit.optimizer.line_search.maximum_learning_rate",
        )
        trials = int(data.get("maximum_trials", 6))
        if not 0.0 < reduction < 1.0:
            raise ConfigError(
                "fit.optimizer.line_search.reduction_factor must be in (0, 1)."
            )
        if growth < 1.0:
            raise ConfigError(
                "fit.optimizer.line_search.growth_factor must be at least 1."
            )
        if minimum > maximum:
            raise ConfigError(
                "fit.optimizer.line_search.minimum_learning_rate cannot exceed "
                "maximum_learning_rate."
            )
        if trials <= 0:
            raise ConfigError(
                "fit.optimizer.line_search.maximum_trials must be positive."
            )
        if data.get("require_loss_decrease", True) is not True:
            raise ConfigError(
                "Backtracking currently requires require_loss_decrease: true."
            )
        return cls(
            enabled=bool(data.get("enabled", False)),
            reduction_factor=reduction,
            growth_factor=growth,
            minimum_learning_rate=minimum,
            maximum_learning_rate=maximum,
            maximum_trials=trials,
        )


@dataclass(frozen=True)
class OptimizerSpec:
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 10.0
    epochs: int = 50
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    line_search: LineSearchSpec = field(default_factory=LineSearchSpec)

    @classmethod
    def from_mapping(cls, value: Any) -> "OptimizerSpec":
        data = _mapping(value, "fit.optimizer")
        _strict(
            data,
            {
                "name",
                "learning_rate",
                "gradient_clip_norm",
                "epochs",
                "beta1",
                "beta2",
                "epsilon",
                "line_search",
            },
            "fit.optimizer",
        )
        if data.get("name", "adam") != "adam":
            raise ConfigError("Only the Adam optimizer is currently supported.")
        epochs = int(data.get("epochs", 50))
        if epochs <= 0:
            raise ConfigError("fit.optimizer.epochs must be positive.")
        learning_rate = _positive(
            data.get("learning_rate", 1e-3), "fit.optimizer.learning_rate"
        )
        return cls(
            learning_rate=learning_rate,
            gradient_clip_norm=_positive(
                data.get("gradient_clip_norm", 10.0),
                "fit.optimizer.gradient_clip_norm",
            ),
            epochs=epochs,
            beta1=float(data.get("beta1", 0.9)),
            beta2=float(data.get("beta2", 0.999)),
            epsilon=_positive(data.get("epsilon", 1e-8), "fit.optimizer.epsilon"),
            line_search=LineSearchSpec.from_mapping(
                data.get("line_search"), learning_rate
            ),
        )


_LOSS_KINDS = {
    "masked_voltage_mse",
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
    "soft_maximum_voltage_error",
}
_LOSS_WINDOWS = {"score", "full_trace", "baseline", "stimulus", "recovery", "stimulus_end"}


@dataclass(frozen=True)
class LossComponentSpec:
    """One statically configured, differentiable objective component."""

    kind: str
    weight: float = 1.0
    window: str = "score"
    protocols: tuple[str, ...] = ()
    scale: float = 1.0
    delta: float = 1.0
    threshold_mV: float = -20.0
    temperature_mV: float = 2.0
    label: str = ""

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "LossComponentSpec":
        where = f"fit.objective.components[{index}]"
        data = _mapping(value, where)
        _strict(
            data,
            {
                "kind",
                "weight",
                "window",
                "protocols",
                "scale",
                "scale_mV",
                "scale_mV_per_ms",
                "scale_hz",
                "delta",
                "threshold_mV",
                "temperature_mV",
                "label",
            },
            where,
        )
        kind = str(data.get("kind", "masked_voltage_mse"))
        if kind not in _LOSS_KINDS:
            raise ConfigError(f"Unsupported loss component kind: {kind}")
        window = str(data.get("window", "score"))
        if window not in _LOSS_WINDOWS:
            raise ConfigError(f"Unsupported loss window: {window}")
        scale_keys = [
            key
            for key in ("scale", "scale_mV", "scale_mV_per_ms", "scale_hz")
            if key in data
        ]
        if len(scale_keys) > 1:
            raise ConfigError(f"{where} defines multiple scale fields.")
        scale = data.get(scale_keys[0], 1.0) if scale_keys else 1.0
        weight = float(data.get("weight", 1.0))
        if weight < 0.0:
            raise ConfigError(f"{where}.weight cannot be negative.")
        return cls(
            kind=kind,
            weight=weight,
            window=window,
            protocols=_strings(data.get("protocols"), f"{where}.protocols"),
            scale=_positive(scale, f"{where}.scale"),
            delta=_positive(data.get("delta", 1.0), f"{where}.delta"),
            threshold_mV=float(data.get("threshold_mV", -20.0)),
            temperature_mV=_positive(
                data.get("temperature_mV", 2.0), f"{where}.temperature_mV"
            ),
            label=str(data.get("label", kind)),
        )


@dataclass(frozen=True)
class FitSpec:
    aggregation: str
    protocol_weights: Mapping[str, float]
    components: tuple[LossComponentSpec, ...]
    optimizer: OptimizerSpec
    batching_strategy: str = "vmap"
    pad_to_longest: bool = False
    checkpoint_every_epochs: int = 1

    @classmethod
    def from_mapping(cls, value: Any) -> "FitSpec":
        data = _mapping(value, "fit")
        _strict(
            data,
            {
                "objective",
                "parameter_transform",
                "optimizer",
                "batching",
                "checkpoint",
            },
            "fit",
        )
        objective = _mapping(data.get("objective"), "fit.objective")
        _strict(
            objective,
            {"components", "aggregation", "protocol_weights", "report_metrics"},
            "fit.objective",
        )
        raw_components = objective.get("components") or [
            {"kind": "masked_voltage_mse", "weight": 1.0}
        ]
        if not isinstance(raw_components, Sequence) or isinstance(raw_components, str):
            raise ConfigError("fit.objective.components must be a list.")
        components = tuple(
            LossComponentSpec.from_mapping(component, index)
            for index, component in enumerate(raw_components)
        )
        if not components or sum(component.weight for component in components) <= 0:
            raise ConfigError("At least one loss component must have positive weight.")
        labels = [component.label for component in components]
        if len(labels) != len(set(labels)):
            raise ConfigError("Loss component labels must be unique.")
        aggregation = str(objective.get("aggregation", "protocol_mean"))
        if aggregation not in {"protocol_mean", "trace_mean", "sample_mean"}:
            raise ConfigError(f"Unsupported objective aggregation: {aggregation}")
        weights = {
            str(key): float(item)
            for key, item in _mapping(
                objective.get("protocol_weights"), "fit.objective.protocol_weights"
            ).items()
        } or {"depolarizing_step": 0.5, "hyperpolarizing_pulse": 0.5}
        if any(weight < 0.0 for weight in weights.values()) or sum(weights.values()) <= 0:
            raise ConfigError("Protocol weights must be non-negative with positive sum.")
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}

        batching = _mapping(data.get("batching"), "fit.batching")
        strategy = str(batching.get("strategy", "vmap"))
        if strategy == "auto":
            strategy = "vmap"
        if strategy not in {"vmap", "serial"}:
            raise ConfigError("fit.batching.strategy must be vmap, serial, or auto.")
        checkpoint = _mapping(data.get("checkpoint"), "fit.checkpoint")
        every = int(checkpoint.get("every_epochs", 1))
        if every <= 0:
            raise ConfigError("fit.checkpoint.every_epochs must be positive.")
        return cls(
            aggregation=aggregation,
            protocol_weights=weights,
            components=components,
            optimizer=OptimizerSpec.from_mapping(data.get("optimizer")),
            batching_strategy=strategy,
            pad_to_longest=bool(batching.get("pad_to_longest", False)),
            checkpoint_every_epochs=every,
        )


@dataclass(frozen=True)
class RuntimeSpec:
    backend: str = "auto"
    precision: str = "float64"
    seed: int = 0
    jit: bool = True
    solver: str = "bwd_euler"
    voltage_solver: str = "jaxley.dhs"
    checkpoint_levels: int = 2
    preallocate: bool = False
    memory_fraction: float = 0.8
    compilation_cache: Path | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeSpec":
        data = _mapping(value, "runtime")
        _strict(
            data,
            {
                "backend",
                "precision",
                "seed",
                "jit",
                "solver",
                "rematerialization",
                "compilation_cache",
                "memory",
                "distributed",
            },
            "runtime",
        )
        backend = str(data.get("backend", "auto"))
        if backend not in {"auto", "cpu", "gpu"}:
            raise ConfigError("runtime.backend must be auto, cpu, or gpu.")
        precision = str(data.get("precision", "float64"))
        if precision not in {"float32", "float64"}:
            raise ConfigError("runtime.precision must be float32 or float64.")
        solver = _mapping(data.get("solver"), "runtime.solver")
        remat = _mapping(data.get("rematerialization"), "runtime.rematerialization")
        cache = _mapping(data.get("compilation_cache"), "runtime.compilation_cache")
        memory = _mapping(data.get("memory"), "runtime.memory")
        cache_path = cache.get("directory") if cache.get("enabled", True) else None
        fraction = float(memory.get("fraction", 0.8))
        if not 0.0 < fraction <= 1.0:
            raise ConfigError("runtime.memory.fraction must be in (0, 1].")
        return cls(
            backend=backend,
            precision=precision,
            seed=int(data.get("seed", 0)),
            jit=bool(data.get("jit", True)),
            solver=str(solver.get("ode", "bwd_euler")),
            voltage_solver=str(solver.get("voltage", "jaxley.dhs")),
            checkpoint_levels=int(remat.get("levels", 2)),
            preallocate=bool(memory.get("preallocate", False)),
            memory_fraction=fraction,
            compilation_cache=Path(cache_path) if cache_path else None,
        )


@dataclass(frozen=True)
class OutputSpec:
    root: Path = Path("runs")
    run_name: str = "auto"
    plot_every_epochs: int = 1
    evaluate_every_epochs: int = 5

    @classmethod
    def from_mapping(cls, value: Any) -> "OutputSpec":
        data = _mapping(value, "output")
        _strict(
            data,
            {
                "root",
                "run_name",
                "plot_every_epochs",
                "evaluate_every_epochs",
                "save_predictions",
                "provenance",
            },
            "output",
        )
        plot_every = int(data.get("plot_every_epochs", 1))
        if plot_every < 0:
            raise ConfigError("output.plot_every_epochs cannot be negative.")
        return cls(
            root=Path(data.get("root", "runs")),
            run_name=str(data.get("run_name", "auto")),
            plot_every_epochs=plot_every,
            evaluate_every_epochs=int(data.get("evaluate_every_epochs", 5)),
        )


@dataclass(frozen=True)
class AppConfig:
    """Validated root configuration used throughout the application."""

    schema_version: int
    model: ModelSpec
    protocol: ProtocolSpec
    dataset: DatasetSpec
    fit: FitSpec
    runtime: RuntimeSpec
    output: OutputSpec
    source_path: Path | None = None

    @classmethod
    def from_mapping(
        cls, value: Any, *, source_path: Path | None = None
    ) -> "AppConfig":
        data = _mapping(value, "configuration")
        _strict(
            data,
            {
                "schema_version",
                "model",
                "protocol",
                "dataset",
                "fit",
                "runtime",
                "output",
            },
            "configuration",
        )
        version = int(data.get("schema_version", 1))
        if version != 1:
            raise ConfigError(f"Unsupported schema_version: {version}")
        return cls(
            schema_version=version,
            model=ModelSpec.from_mapping(data.get("model")),
            protocol=ProtocolSpec.from_mapping(data.get("protocol")),
            dataset=DatasetSpec.from_mapping(data.get("dataset")),
            fit=FitSpec.from_mapping(data.get("fit")),
            runtime=RuntimeSpec.from_mapping(data.get("runtime")),
            output=OutputSpec.from_mapping(data.get("output")),
            source_path=source_path,
        )

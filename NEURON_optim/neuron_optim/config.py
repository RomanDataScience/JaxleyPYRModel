"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .parameters import make_parameter_space


@dataclass(frozen=True)
class RunConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def cell(self) -> str:
        return str(self.raw["data"]["cell"])

    @property
    def trace_names(self) -> tuple[str, ...]:
        return tuple(self.raw["data"].get("trace_names", ["v75ctrl", "v76ctrl", "v77ctrl", "v78ctrl"]))

    @property
    def hyperpolarizing_trace_index(self) -> int:
        """Ordinal trace selected by the passive and hyperpolarizing stages."""
        return int(self.raw["data"].get("hyperpolarizing_trace_index", 0))

    @property
    def hyperpolarizing_trace_names(self) -> tuple[str, ...]:
        """Resolve the selected hyperpolarizing trace from the trace list.

        The hyperpolarizing current protocol is shared across the configured
        traces, so the default index is zero. Depolarizing stages continue to
        use all ``trace_names``.
        """
        return (self.trace_names[self.hyperpolarizing_trace_index],)

    @property
    def data_root(self) -> Path:
        path = Path(self.raw["data"]["root"])
        return (self.path.parent / path).resolve() if not path.is_absolute() else path

    @property
    def d_lambda(self) -> float:
        return float(self.raw.get("runtime", {}).get("d_lambda", 0.3))

    @property
    def backend(self) -> str:
        return str(self.raw.get("runtime", {}).get("backend", "neuron")).lower()

    @property
    def simulation_pre_ms(self) -> float:
        return float(self.raw.get("runtime", {}).get("simulation_pre_ms", 500.0))

    @property
    def simulation_post_ms(self) -> float:
        return float(self.raw.get("runtime", {}).get("simulation_post_ms", 600.0))

    @property
    def hyperpolarizing_fitness_pre_ms(self) -> float:
        return float(self.raw.get("runtime", {}).get(
            "hyperpolarizing_fitness_pre_ms", 100.0
        ))

    @property
    def workers(self) -> int:
        return int(self.raw.get("runtime", {}).get("parallel_workers", 6))

    @property
    def study_workers(self) -> int:
        return int(self.raw.get("runtime", {}).get("study_workers", 1))

    @property
    def parameters(self):
        section = self.raw.get("parameters", {})
        return make_parameter_space(section.get("include"), section.get("exclude", ()))

    def stage_parameter_names(self, stage: str) -> tuple[str, ...]:
        section = self.section(stage)
        names = section.get("parameter_names")
        if names is None:
            return self.parameters.keys
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError(f"{stage}.parameter_names must be a non-empty list")
        selected = tuple(str(name) for name in names)
        unknown = sorted(set(selected) - set(self.parameters.keys))
        if unknown:
            raise ValueError(f"{stage}.parameter_names contains unsupported parameters: {unknown}")
        if len(set(selected)) != len(selected):
            raise ValueError(f"{stage}.parameter_names contains duplicates")
        return selected

    def stage_parameter_space(self, stage: str, *, lower_overrides=None,
                              upper_overrides=None):
        section = self.section(stage)
        configured_lower = section.get("lower_bounds", {})
        configured_upper = section.get("upper_bounds", {})
        if not isinstance(configured_lower, dict):
            raise ValueError(f"{stage}.lower_bounds must be a mapping")
        if not isinstance(configured_upper, dict):
            raise ValueError(f"{stage}.upper_bounds must be a mapping")
        lower = dict(configured_lower)
        lower.update(lower_overrides or {})
        upper = dict(configured_upper)
        upper.update(upper_overrides or {})
        return make_parameter_space(
            include=self.stage_parameter_names(stage),
            lower_overrides=lower,
            upper_overrides=upper,
        )

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def hash(self) -> str:
        encoded = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            space = self.parameters
            if len(space.keys) == 0:
                errors.append("parameters cannot be empty")
        except Exception as exc:
            errors.append(str(exc))
        if len(self.trace_names) != 4:
            errors.append("data.trace_names must contain exactly four traces")
        try:
            hyper_index = self.hyperpolarizing_trace_index
            if not 0 <= hyper_index < len(self.trace_names):
                errors.append("data.hyperpolarizing_trace_index must select a trace by valid index")
        except (TypeError, ValueError):
            errors.append("data.hyperpolarizing_trace_index must be an integer")
        if self.simulation_pre_ms < 0:
            errors.append("runtime.simulation_pre_ms must be >= 0")
        if self.simulation_post_ms < 0:
            errors.append("runtime.simulation_post_ms must be >= 0")
        if self.hyperpolarizing_fitness_pre_ms < 0:
            errors.append("runtime.hyperpolarizing_fitness_pre_ms must be >= 0")
        if self.workers < 1:
            errors.append("runtime.parallel_workers must be >= 1")
        if self.study_workers < 1:
            errors.append("runtime.study_workers must be >= 1")
        if self.backend not in {"neuron", "jaxley"}:
            errors.append("runtime.backend must be either 'neuron' or 'jaxley'")
        for name in ("passive", "stage1", "stage2"):
            section = self.section(name)
            if int(section.get("generations", 0)) < 1:
                errors.append(f"{name}.generations must be >= 1")
            if int(section.get("population_size", 0)) < 2:
                errors.append(f"{name}.population_size must be >= 2")
            if not section.get("seeds"):
                errors.append(f"{name}.seeds must not be empty")
            if int(section.get("checkpoint_every", 1)) < 1:
                errors.append(f"{name}.checkpoint_every must be >= 1")
            if name == "stage2":
                try:
                    self.stage_parameter_space(name)
                except Exception as exc:
                    errors.append(str(exc))
        if int(self.raw.get("plotting", {}).get("every", 1)) < 1:
            errors.append("plotting.every must be >= 1")
        if not self.data_root.exists():
            errors.append(f"data root does not exist: {self.data_root}")
        return errors


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return RunConfig(raw, path)

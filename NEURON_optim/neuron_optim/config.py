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
    def workers(self) -> int:
        return int(self.raw.get("runtime", {}).get("parallel_workers", 6))

    @property
    def parameters(self):
        section = self.raw.get("parameters", {})
        return make_parameter_space(section.get("include"), section.get("exclude", ()))

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
        if self.simulation_pre_ms < 0:
            errors.append("runtime.simulation_pre_ms must be >= 0")
        if self.simulation_post_ms < 0:
            errors.append("runtime.simulation_post_ms must be >= 0")
        if self.workers < 1:
            errors.append("runtime.parallel_workers must be >= 1")
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

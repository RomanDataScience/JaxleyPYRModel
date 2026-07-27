"""YAML loading, inheritance, environment expansion, and path resolution."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from .schema import AppConfig, ConfigError


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping.")

    parents = raw.pop("extends", ())
    if isinstance(parents, (str, Path)):
        parents = (parents,)
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(os.path.expandvars(str(parent)))
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, _read_yaml(parent_path, (*stack, path)))
    return _deep_merge(merged, raw)


def _resolved(base: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    expanded_text = os.path.expandvars(str(path))
    unresolved = re.findall(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))", expanded_text)
    if unresolved:
        names = sorted({first or second for first, second in unresolved})
        raise ConfigError(
            f"Undefined environment variable(s) in path: {', '.join(names)}"
        )
    expanded = Path(expanded_text).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML config and resolve all filesystem paths against its directory."""
    source = Path(path).resolve()
    config = AppConfig.from_mapping(_read_yaml(source), source_path=source)
    base = source.parent
    morphology = replace(
        config.model.morphology,
        path=_resolved(base, config.model.morphology.path),
    )
    model = replace(config.model, morphology=morphology)

    dataset_root = _resolved(base, config.dataset.root)
    assert dataset_root is not None
    manifest = config.dataset.manifest
    if not manifest.is_absolute():
        manifest = dataset_root / manifest
    dataset = replace(
        config.dataset,
        root=dataset_root,
        manifest=manifest.resolve(),
    )
    runtime = replace(
        config.runtime,
        compilation_cache=_resolved(base, config.runtime.compilation_cache),
    )
    output_root = _resolved(base, config.output.root)
    assert output_root is not None
    output = replace(config.output, root=output_root)
    return replace(
        config,
        model=model,
        dataset=dataset,
        runtime=runtime,
        output=output,
    )

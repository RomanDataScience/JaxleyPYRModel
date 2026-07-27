"""Apply environment settings that must precede importing JAX."""

from __future__ import annotations

import os

from jaxley_refactored.config.schema import RuntimeSpec


def configure_environment(spec: RuntimeSpec) -> None:
    if spec.backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif spec.backend == "gpu":
        # Do not spell the implementation backend here: JAX calls accelerator
        # devices "gpu", but JAX_PLATFORMS expects vendor backends such as
        # "cuda" or "rocm". Let the installed jaxlib discover either one and
        # fail explicitly in ``validate_device`` if no GPU is visible.
        os.environ.pop("JAX_PLATFORMS", None)
    os.environ["JAX_ENABLE_X64"] = (
        "true" if spec.precision == "float64" else "false"
    )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "true" if spec.preallocate else "false"
    )
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(spec.memory_fraction)
    if spec.compilation_cache is not None:
        spec.compilation_cache.mkdir(parents=True, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(spec.compilation_cache)

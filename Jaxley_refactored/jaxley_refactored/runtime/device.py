"""Fail-fast backend validation after JAX import."""

from __future__ import annotations

from jaxley_refactored.config.schema import RuntimeSpec


def validate_device(spec: RuntimeSpec) -> dict:
    import jax

    devices = jax.devices()
    platforms = {device.platform for device in devices}
    if spec.backend == "gpu" and "gpu" not in platforms:
        raise RuntimeError(
            f"GPU requested but JAX exposed only {sorted(platforms)} devices."
        )
    if spec.backend == "cpu" and platforms != {"cpu"}:
        raise RuntimeError(f"CPU requested but JAX exposed {sorted(platforms)}.")
    return {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "precision": spec.precision,
    }


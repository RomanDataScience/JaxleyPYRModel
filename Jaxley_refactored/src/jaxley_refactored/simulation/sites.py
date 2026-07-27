"""Resolve configured anatomical sites without leaking selection logic."""

from __future__ import annotations

from jaxley_refactored.config.schema import SiteSpec


def resolve_site(cell, spec: SiteSpec):
    if not hasattr(cell, spec.group):
        raise ValueError(f"Cell has no configured group {spec.group!r}.")
    group = getattr(cell, spec.group)
    try:
        return group.branch(spec.branch).loc(spec.location)
    except Exception as error:
        raise ValueError(
            f"Invalid site {spec.group}.branch({spec.branch}).loc({spec.location})."
        ) from error


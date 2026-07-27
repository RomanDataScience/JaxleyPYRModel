"""Declarative selection of spatial distribution families.

The numerical Combe rules currently remain behind ``LegacyCombeBackend`` for
validated HOC parity. This registry owns which catalog coefficients a profile
accepts, so adding a new family does not require changes to model orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from jaxley_refactored.parameters import ParameterCatalog


@dataclass(frozen=True)
class DistributionProfile:
    """One named rule family and the coefficients it permits."""

    name: str
    coefficient_names: frozenset[str]
    description: str


class DistributionRegistry:
    """Resolve and validate profile overrides against a parameter catalog."""

    def __init__(
        self,
        profiles: Iterable[DistributionProfile],
        catalog: ParameterCatalog,
    ):
        self._profiles = {profile.name: profile for profile in profiles}
        self._catalog = catalog
        if not self._profiles:
            raise ValueError("At least one distribution profile is required.")

    def resolve(
        self, profile_name: str, overrides: Mapping[str, float]
    ) -> dict[str, float]:
        try:
            profile = self._profiles[profile_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown distribution profile {profile_name!r}; "
                f"available: {sorted(self._profiles)}"
            ) from error
        resolved: dict[str, float] = {}
        for name, value in overrides.items():
            canonical = self._catalog.resolve(name)
            if canonical not in profile.coefficient_names:
                raise ValueError(
                    f"{canonical} is not a coefficient of {profile_name}."
                )
            if canonical in resolved:
                raise ValueError(
                    f"Distribution override aliases collide at {canonical}."
                )
            resolved[canonical] = self._catalog.get(canonical).validate(value)
        return resolved


def combe2023_distributions(
    catalog: ParameterCatalog,
) -> DistributionRegistry:
    """Return the validated CCh-driven active/passive rule family."""
    return DistributionRegistry(
        (
            DistributionProfile(
                name="combe2023_cch_driven",
                coefficient_names=frozenset(spec.name for spec in catalog),
                description=(
                    "Combe CCh-driven active conductance and passive-property "
                    "profiles evaluated on HOC assignment distances or final "
                    "SWC compartment centers."
                ),
            ),
        ),
        catalog,
    )

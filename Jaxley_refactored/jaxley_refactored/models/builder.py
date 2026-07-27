"""SOLID orchestration of providers, mechanisms, and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from jaxley_refactored.compatibility import LegacyCombeBackend
from jaxley_refactored.config.hashing import stable_hash
from jaxley_refactored.config.schema import ModelSpec
from jaxley_refactored.distributions import (
    DistributionRegistry,
    combe2023_distributions,
)
from jaxley_refactored.mechanisms import MechanismRegistry, combe2023_mechanisms
from jaxley_refactored.morphology import (
    HocArtifactProvider,
    HocLiveProvider,
    MorphologyProviderRegistry,
    StaticFeatures,
    SwcProvider,
)
from jaxley_refactored.parameters import (
    ParameterCatalog,
    Parameterizer,
    combe2023_catalog,
)


@dataclass(frozen=True)
class BuiltModel:
    """The static cell and its dynamic parameter interface."""

    cell: Any
    parameterizer: Parameterizer
    features: StaticFeatures
    signature: str
    provenance: Mapping[str, Any]
    enabled_mechanisms: frozenset[str]
    reference_values: tuple[float, ...]


class ModelBuilder:
    """Assemble a model from injected provider/registry abstractions."""

    def __init__(
        self,
        providers: MorphologyProviderRegistry,
        mechanisms: MechanismRegistry,
        distributions: DistributionRegistry,
        catalog: ParameterCatalog,
        parameter_backend: LegacyCombeBackend,
    ):
        self._providers = providers
        self._mechanisms = mechanisms
        self._distributions = distributions
        self._catalog = catalog
        self._parameter_backend = parameter_backend

    def build(self, spec: ModelSpec) -> BuiltModel:
        if spec.model_id != "combe2023":
            raise ValueError(f"Unsupported model recipe: {spec.model_id}")
        exact_provider = spec.morphology.provider in {"hoc_live", "hoc_artifact"}
        exact_profile = spec.profile_mode == "exact_hoc_frozen_grid"
        if exact_provider != exact_profile:
            raise ValueError(
                "hoc_live/hoc_artifact require exact_hoc_frozen_grid, while "
                "swc requires rule_based_final_centers."
            )
        morphology = self._providers.get(spec.morphology.provider).build(spec)
        enabled = self._mechanisms.apply(
            morphology.cell,
            spec.mechanisms.include,
            spec.mechanisms.exclude,
        )
        selected_specs = self._catalog.select(
            include_tags=spec.parameters.include_tags,
            include=spec.parameters.include,
            exclude=spec.parameters.exclude,
        )
        distribution_overrides = self._distributions.resolve(
            spec.distributions.preset, spec.distributions.overrides
        )
        overlap = (
            distribution_overrides.keys()
            & spec.parameters.value_overrides.keys()
        )
        if overlap:
            raise ValueError(
                "Distribution and parameter overrides both define "
                f"{sorted(overlap)}; keep one source of truth."
            )
        overrides = {
            **distribution_overrides,
            **spec.parameters.value_overrides,
        }
        reference = self._catalog.defaults(selected_specs, overrides)
        selected_names = {item.name for item in selected_specs}
        fixed_specs = tuple(
            self._catalog.get(name)
            for name in distribution_overrides
            if self._catalog.resolve(name) not in selected_names
        )
        fixed_values = self._catalog.defaults(
            fixed_specs, distribution_overrides
        )
        self._validate_parameter_targets(
            morphology.cell, (*selected_specs, *fixed_specs)
        )
        parameterizer = Parameterizer(
            cell=morphology.cell,
            specs=selected_specs,
            backend=self._parameter_backend,
            fixed_specs=fixed_specs,
            fixed_values=fixed_values,
        )
        signature = stable_hash(
            {
                "model": spec.model_id,
                "morphology": morphology.fingerprint,
                "mechanisms": sorted(enabled),
                "distribution_preset": spec.distributions.preset,
                "distribution_overrides": distribution_overrides,
                "parameters": parameterizer.keys,
                "profile_mode": spec.profile_mode,
            }
        )
        return BuiltModel(
            cell=morphology.cell,
            parameterizer=parameterizer,
            features=morphology.features,
            signature=signature,
            provenance=morphology.provenance,
            enabled_mechanisms=frozenset(enabled),
            reference_values=reference,
        )

    @staticmethod
    def _validate_parameter_targets(cell, specifications) -> None:
        for spec in specifications:
            for target in spec.targets:
                group, column = target.split(".", maxsplit=1)
                view = cell if group == "all" else getattr(cell, group)
                if column not in view.nodes:
                    raise ValueError(
                        f"Selected parameter {spec.name} targets disabled or "
                        f"missing field {target}."
                    )
                values = view.nodes[column].to_numpy()
                if not np.any(~np.isnan(values.astype(float))):
                    raise ValueError(
                        f"Selected parameter {spec.name} has no active target {target}."
                    )


def default_builder() -> ModelBuilder:
    backend = LegacyCombeBackend()
    catalog = combe2023_catalog()
    providers = MorphologyProviderRegistry()
    providers.register(HocLiveProvider(backend))
    providers.register(HocArtifactProvider(backend))
    providers.register(SwcProvider(backend))
    return ModelBuilder(
        providers=providers,
        mechanisms=combe2023_mechanisms(),
        distributions=combe2023_distributions(catalog),
        catalog=catalog,
        parameter_backend=backend,
    )

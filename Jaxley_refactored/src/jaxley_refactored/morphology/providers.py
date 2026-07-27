"""Live-HOC and SWC morphology providers plus an extensible registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jaxley_refactored.compatibility import LegacyCombeBackend
from jaxley_refactored.config.hashing import stable_hash
from jaxley_refactored.config.schema import ModelSpec

from .features import extract_features
from .records import MorphologyResult


class MorphologyProvider(Protocol):
    """Construct one structural/static model variant."""

    key: str

    def build(self, spec: ModelSpec) -> MorphologyResult: ...


@dataclass
class HocLiveProvider:
    """Build the exact HOC model through NEURON on a reference CPU node."""

    backend: LegacyCombeBackend
    key: str = "hoc_live"

    def build(self, spec: ModelSpec) -> MorphologyResult:
        cell = self.backend.build_cell(
            morphology_source="hoc",
            d_lambda=spec.morphology.d_lambda,
            calcium_diffusion=spec.mechanisms.calcium_diffusion,
        )
        features = extract_features(cell, spec.morphology.required_groups)
        fingerprint = stable_hash(
            {
                "provider": self.key,
                "d_lambda": spec.morphology.d_lambda,
                "n_branch": len(cell.xyzr),
                "n_comp": len(cell.nodes),
                "ncomp": cell.nodes.groupby("global_branch_index").size().tolist(),
            }
        )
        return MorphologyResult(
            cell=cell,
            features=features,
            fingerprint=fingerprint,
            provenance={
                "provider": self.key,
                "d_lambda": spec.morphology.d_lambda,
                "neuron_required": True,
            },
        )


@dataclass
class SwcProvider:
    """Build the portable rule-based Combe model from the registered SWC."""

    backend: LegacyCombeBackend
    key: str = "swc"

    def build(self, spec: ModelSpec) -> MorphologyResult:
        if spec.profile_mode != "rule_based_final_centers":
            raise ValueError(
                "The SWC provider requires profile_mode=rule_based_final_centers."
            )
        path = spec.morphology.path
        if path is None:
            path = self.backend.default_swc_path
        if not path.is_file():
            raise FileNotFoundError(f"Configured SWC morphology does not exist: {path}")
        cell = self.backend.build_swc_cell(
            path,
            d_lambda=spec.morphology.d_lambda,
            frequency_hz=spec.morphology.frequency_hz,
            calcium_diffusion=spec.mechanisms.calcium_diffusion,
            calcium_axial_diffusion=spec.mechanisms.calcium_axial_diffusion,
        )
        features = extract_features(cell, spec.morphology.required_groups)
        fingerprint = stable_hash(
            {
                "provider": self.key,
                "path": str(path),
                "d_lambda": spec.morphology.d_lambda,
                "n_branch": len(cell.xyzr),
                "n_comp": len(cell.nodes),
            }
        )
        return MorphologyResult(
            cell=cell,
            features=features,
            fingerprint=fingerprint,
            provenance={
                "provider": self.key,
                "path": str(path),
                "neuron_required": False,
            },
        )


class MorphologyProviderRegistry:
    """Registry that lets new morphology formats be added without editing builders."""

    def __init__(self):
        self._providers: dict[str, MorphologyProvider] = {}

    def register(self, provider: MorphologyProvider) -> None:
        if provider.key in self._providers:
            raise ValueError(f"Morphology provider already registered: {provider.key}")
        self._providers[provider.key] = provider

    def get(self, key: str) -> MorphologyProvider:
        try:
            return self._providers[key]
        except KeyError as error:
            raise KeyError(
                f"Unknown morphology provider {key!r}; "
                f"available: {sorted(self._providers)}"
            ) from error

"""Single source of truth for fitted Combe2023 parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata needed to configure, validate, and report one parameter."""

    name: str
    default: float
    bounds: tuple[float, float]
    units: str
    tags: tuple[str, ...]
    targets: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def with_expanded_bounds(self, factor: float) -> "ParameterSpec":
        """Return wider physically signed bounds while retaining the default."""
        factor = float(factor)
        if factor < 1.0:
            raise ValueError("Parameter-bound expansion factor must be at least 1.")
        if factor == 1.0:
            return self
        lower, upper = self.bounds
        if lower >= 0.0:
            expanded_lower = lower / factor if lower > 0.0 else 0.0
            expanded_upper = upper * factor
        elif upper <= 0.0:
            expanded_lower = self.default + factor * (lower - self.default)
            expanded_upper = self.default + factor * (upper - self.default)
        else:
            expanded_lower = self.default + factor * (lower - self.default)
            expanded_upper = self.default + factor * (upper - self.default)
        return ParameterSpec(
            name=self.name,
            default=self.default,
            bounds=(expanded_lower, expanded_upper),
            units=self.units,
            tags=self.tags,
            targets=self.targets,
            aliases=self.aliases,
        )

    def validate(self, value: float) -> float:
        lower, upper = self.bounds
        value = float(value)
        if not lower <= value <= upper:
            raise ValueError(
                f"{self.name}={value} is outside [{lower}, {upper}]."
            )
        return value


class ParameterCatalog:
    """Read-only registry with deterministic selection and alias resolution."""

    def __init__(self, specifications: Iterable[ParameterSpec]):
        self._specs = {spec.name: spec for spec in specifications}
        if len(self._specs) == 0:
            raise ValueError("A parameter catalog cannot be empty.")
        aliases: dict[str, str] = {}
        for spec in self._specs.values():
            for alias in spec.aliases:
                if alias in aliases or alias in self._specs:
                    raise ValueError(f"Duplicate parameter alias: {alias}")
                aliases[alias] = spec.name
        self._aliases = aliases

    def __iter__(self):
        return iter(self._specs.values())

    def resolve(self, name: str) -> str:
        canonical = self._aliases.get(name, name)
        if canonical not in self._specs:
            raise KeyError(f"Unknown parameter: {name}")
        return canonical

    def get(self, name: str) -> ParameterSpec:
        return self._specs[self.resolve(name)]

    def select(
        self,
        *,
        include_tags: Iterable[str] = (),
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
    ) -> tuple[ParameterSpec, ...]:
        explicit = tuple(self.resolve(name) for name in include)
        tags = set(include_tags)
        if explicit:
            selected = set(explicit)
        elif tags:
            selected = {
                spec.name
                for spec in self._specs.values()
                if tags.intersection(spec.tags)
            }
        else:
            selected = set(self._specs)
        selected.difference_update(self.resolve(name) for name in exclude)
        if not selected:
            raise ValueError("Parameter selection is empty.")
        return tuple(spec for spec in self._specs.values() if spec.name in selected)

    def defaults(
        self,
        specs: Iterable[ParameterSpec],
        overrides: Mapping[str, float] | None = None,
    ) -> tuple[float, ...]:
        resolved = {
            self.resolve(key): float(value) for key, value in (overrides or {}).items()
        }
        unknown = set(resolved) - self._specs.keys()
        if unknown:
            raise KeyError(f"Unknown parameter overrides: {sorted(unknown)}")
        return tuple(
            spec.validate(resolved.get(spec.name, spec.default)) for spec in specs
        )


_DEFAULT_CATALOG_PATH = Path(__file__).with_name("combe2023.yaml")

_PASSIVE = {
    "RmSoma",
    "RaSoma",
    "RmTuft",
    "RaTuft",
    "DistHalfRm",
    "DistHalfRa",
    "SlopeRm",
    "SlopeRa",
    "Epas",
    "CmSoma",
    "SpineFactorBasal",
    "SpineFactorTuft",
}
_KINETIC = {
    "kd_deactivation_tau_scale",
    "nat_fast_inactivation_tau_scale",
    "nat_slow_recovery_tau_scale",
    "h_tau_scale",
}

_TARGETS = {
    "soma_hbar": ("soma.h_gbar", "apical.h_gbar", "basal.h_gbar"),
    "KirGbar": ("apical.kir_gbar", "basal.kir_gbar"),
    "soma_caL": ("soma.cal_gcalbar",),
    "soma_car": ("apical.car_gcabar",),
    "gsomacar": ("soma.car_gcabar",),
    "soma_caLH": ("apical.calH_gcalbar",),
    "soma_caT": ("soma.cat_gcatbar", "apical.cat_gcatbar"),
    "soma_km": ("soma.km_gbar", "apical.km_gbar", "axon.km_gbar"),
    "mykca_init": ("soma.mykca_gkbar", "apical.mykca_gkbar"),
    "soma_kca": ("soma.kca_gbar", "apical.kca_gbar"),
    "AXNa": ("axon.nax_gbar",),
    "gkdrsoma": ("soma.kd_gbar",),
    "gkdrdend": ("basal.kd_gbar",),
    "soma_kap": ("soma.kap_gkabar", "apical.kap_gkabar"),
    "axon_kap": ("axon.kap_gkabar",),
    "basal_kap": ("basal.kap_gkabar",),
    "soma_kad": ("apical.kad_gkabar",),
    "gna": ("soma.na16a_gbar", "axon.nax_gbar"),
    "axongkdr": ("axon.kd_gbar",),
    "gnadend": ("apical.na16a_gbar", "basal.na3dend_gbar"),
    "gkdrapical": ("apical.kd_gbar",),
    "gkv2soma": ("soma.Kv2like_gbar",),
    "gkv2": ("apical.Kv2like_gbar", "basal.Kv2like_gbar"),
    "gkv2axon": ("axon.Kv2like_gbar",),
    "gkv2scale": ("apical.Kv2like_gbar", "basal.Kv2like_gbar"),
    "scale_Na_conduct": ("soma.na16a_gbar", "apical.na16a_gbar"),
    "icangbar": ("soma.icand_gbar", "apical.icand_gbar"),
    "nap_gnabar": ("soma.nap_gnabar", "basal.nap_gnabar"),
    "RmSoma": ("all.Leak_gLeak",),
    "RmTuft": ("all.Leak_gLeak",),
    "DistHalfRm": ("all.Leak_gLeak",),
    "SlopeRm": ("all.Leak_gLeak",),
    "RaSoma": ("all.axial_resistivity",),
    "RaTuft": ("all.axial_resistivity",),
    "DistHalfRa": ("all.axial_resistivity",),
    "SlopeRa": ("all.axial_resistivity",),
    "Epas": ("all.Leak_eLeak",),
    "CmSoma": ("all.capacitance",),
    "SpineFactorBasal": ("basal.capacitance", "basal.Leak_gLeak"),
    "SpineFactorTuft": ("apical.capacitance", "apical.Leak_gLeak"),
    "kd_deactivation_tau_scale": (
        "soma.kd_deactivation_tau_scale",
        "apical.kd_deactivation_tau_scale",
        "axon.kd_deactivation_tau_scale",
        "basal.kd_deactivation_tau_scale",
    ),
    "nat_fast_inactivation_tau_scale": (
        "soma.na16a_fast_inactivation_tau_scale",
        "apical.na16a_fast_inactivation_tau_scale",
        "axon.nax_fast_inactivation_tau_scale",
        "basal.na3dend_fast_inactivation_tau_scale",
    ),
    "nat_slow_recovery_tau_scale": (
        "soma.na16a_slow_recovery_tau_scale",
        "apical.na16a_slow_recovery_tau_scale",
    ),
    "h_tau_scale": (
        "soma.h_tau_scale",
        "apical.h_tau_scale",
        "basal.h_tau_scale",
    ),
}


def combe2023_catalog(path: str | Path | None = None) -> ParameterCatalog:
    """Return 40 legacy Combe parameters plus four kinetic time scales."""
    values = _load_values(Path(path) if path is not None else _DEFAULT_CATALOG_PATH)
    specs = []
    for name, default, limits in values:
        tags = (
            ("passive",)
            if name in _PASSIVE
            else ("kinetics",)
            if name in _KINETIC
            else ("conductance",)
        )
        specs.append(
            ParameterSpec(
                name=name,
                default=default,
                bounds=limits,
                units=_units(name),
                tags=tags,
                targets=_TARGETS[name],
            )
        )
    return ParameterCatalog(specs)


def _load_values(path: Path) -> tuple[tuple[str, float, tuple[float, float]], ...]:
    """Load the ordered, user-editable initial values and bounds."""
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or set(document) != {"parameters"}:
        raise ValueError(f"{path} must contain only a 'parameters' list.")
    entries = document["parameters"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'parameters' must be a non-empty list.")

    values = []
    seen = set()
    for index, entry in enumerate(entries):
        location = f"{path}: parameters[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"name", "initial", "bounds"}:
            raise ValueError(
                f"{location} must contain exactly name, initial, and bounds."
            )
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{location}.name must be a non-empty string.")
        if name in seen:
            raise ValueError(f"{path}: duplicate parameter name: {name}")
        if name not in _TARGETS:
            raise ValueError(f"{path}: unknown parameter name: {name}")
        bounds = entry["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{location}.bounds must be [lower, upper].")
        try:
            initial = float(entry["initial"])
            lower, upper = (float(value) for value in bounds)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{location} values and bounds must be numeric.") from error
        if lower > upper:
            raise ValueError(f"{location} lower bound exceeds upper bound.")
        if not lower <= initial <= upper:
            raise ValueError(f"{location} initial value is outside its bounds.")
        seen.add(name)
        values.append((name, initial, (lower, upper)))

    missing = set(_TARGETS) - seen
    if missing:
        raise ValueError(f"{path}: missing parameter(s): {sorted(missing)}")
    return tuple(values)


def _units(name: str) -> str:
    if name.startswith("Rm"):
        return "ohm*cm2"
    if name.startswith("Ra"):
        return "ohm*cm"
    if name.startswith(("Dist", "Slope")):
        return "um"
    if name == "Epas":
        return "mV"
    if name == "CmSoma":
        return "uF/cm2"
    if (
        name.startswith("Spine")
        or name.endswith("_tau_scale")
        or name in {"AXNa", "gkv2scale", "scale_Na_conduct"}
    ):
        return "dimensionless"
    return "S/cm2"

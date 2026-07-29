"""Single source of truth for fitted Combe2023 parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


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


_VALUES = {
    "soma_hbar": (3e-05, (0.0, 0.0003)),
    "KirGbar": (0.00101535, (0.0, 0.005)),
    "soma_caL": (6e-05, (0.0, 0.0006)),
    "soma_car": (3e-05, (0.0, 0.0003)),
    "gsomacar": (8e-05, (0.0, 0.0008)),
    "soma_caLH": (0.0, (0.0, 0.001)),
    "soma_caT": (0.0003, (0.0, 0.003)),
    "soma_km": (0.0, (0.0, 0.01)),
    "mykca_init": (0.0, (0.0, 0.01)),
    "soma_kca": (0.0, (0.0, 0.01)),
    "AXNa": (3.5, (0.1, 10.0)),
    "gkdrsoma": (0.0, (0.0, 0.02)),
    "gkdrdend": (0.0, (0.0, 0.02)),
    "soma_kap": (0.0385, (0.0, 0.2)),
    "axon_kap": (0.056, (0.0, 0.2)),
    "basal_kap": (0.0025036, (0.0, 0.05)),
    "soma_kad": (0.0385, (0.0, 0.2)),
    "gna": (0.035, (0.0, 0.1)),
    "axongkdr": (0.011, (0.0, 0.05)),
    "gnadend": (0.0225, (0.0, 0.1)),
    "gkdrapical": (0.0005, (0.0, 0.01)),
    "gkv2soma": (0.0132, (0.0, 0.1)),
    "gkv2": (0.0198, (0.0, 0.1)),
    "gkv2axon": (0.0198, (0.0, 0.1)),
    "gkv2scale": (0.3, (0.0, 2.0)),
    "scale_Na_conduct": (14.0, (1.0, 30.0)),
    "icangbar": (0.045, (0.0, 0.2)),
    "nap_gnabar": (0.0, (0.0, 0.001)),
    "RmSoma": (149999.0, (50000.0, 300000.0)),
    "RaSoma": (42.562, (20.0, 150.0)),
    "RmTuft": (45373.4, (10000.0, 150000.0)),
    "RaTuft": (35.0, (20.0, 150.0)),
    "DistHalfRm": (151.741, (20.0, 500.0)),
    "DistHalfRa": (90.8296, (20.0, 300.0)),
    "SlopeRm": (13.8656, (1.0, 80.0)),
    "SlopeRa": (7.76766, (1.0, 80.0)),
    "Epas": (-71.9879, (-90.0, -50.0)),
    "CmSoma": (1.0, (0.3, 5.0)),
    "SpineFactorBasal": (3.5, (1.0, 6.0)),
    "SpineFactorTuft": (3.5, (1.0, 6.0)),
    "kd_deactivation_tau_scale": (1.0, (0.25, 4.0)),
    "nat_fast_inactivation_tau_scale": (1.0, (0.5, 2.0)),
    "nat_slow_recovery_tau_scale": (1.0, (0.5, 2.0)),
    "h_tau_scale": (1.0, (0.5, 2.0)),
}

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


def combe2023_catalog() -> ParameterCatalog:
    """Return 40 legacy Combe parameters plus four kinetic time scales."""
    specs = []
    for name, (default, limits) in _VALUES.items():
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

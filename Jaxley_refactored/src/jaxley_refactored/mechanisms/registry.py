"""Declarative mechanism inventory and static channel selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MechanismInfo:
    name: str
    description: str
    requires: tuple[str, ...] = ()


class MechanismRegistry:
    """Apply a configured subset to a fully assembled reference recipe."""

    def __init__(self, mechanisms: Iterable[MechanismInfo]):
        self._items = {item.name: item for item in mechanisms}

    def names(self) -> tuple[str, ...]:
        return tuple(self._items)

    def selected(self, include: Iterable[str], exclude: Iterable[str]) -> set[str]:
        include = tuple(include)
        selected = set(self._items) if "all_from_preset" in include else set(include)
        unknown = (selected | set(exclude)) - self._items.keys()
        if unknown:
            raise KeyError(f"Unknown mechanism(s): {sorted(unknown)}")
        selected.difference_update(exclude)
        for name in tuple(selected):
            missing = set(self._items[name].requires) - selected
            if missing:
                raise ValueError(
                    f"Mechanism {name} requires enabled mechanism(s) {sorted(missing)}."
                )
        return selected

    def apply(self, cell, include: Iterable[str], exclude: Iterable[str]) -> set[str]:
        selected = self.selected(include, exclude)
        for channel in tuple(cell.channels) + tuple(cell.pumps):
            if channel._name not in selected:
                cell.delete(channel)
        return selected


def combe2023_mechanisms() -> MechanismRegistry:
    """Return the channel inventory used by the Combe CCh-driven recipe."""
    descriptions = {
        "d3": "Shared calcium state dynamics.",
        "Leak": "Passive leak current.",
        "cal4": "Calcium/IP3 dynamics and diffusion state.",
        "icand": "Calcium-activated nonspecific cation current.",
        "na16a": "Somatic and apical Nav1.6 current.",
        "kd": "Delayed rectifier potassium current.",
        "Kv2like": "Kv2-like potassium current.",
        "h": "Hyperpolarization-activated current.",
        "kap": "Proximal A-type potassium current.",
        "km": "M-type potassium current.",
        "kca": "Calcium-activated potassium current.",
        "mykca": "Fast calcium-activated potassium current.",
        "nap": "Persistent sodium current.",
        "cal": "L-type calcium current.",
        "cat": "T-type calcium current.",
        "car": "R-type calcium current.",
        "calH": "High-voltage calcium current.",
        "kad": "Distal A-type potassium current.",
        "kir": "Inward rectifier potassium current.",
        "nax": "Axonal sodium current.",
        "na3dend": "Basal dendritic sodium current.",
    }
    calcium_dependent = {"icand", "kca", "mykca"}
    return MechanismRegistry(
        MechanismInfo(
            name,
            description,
            requires=("d3", "cal4") if name in calcium_dependent else (),
        )
        for name, description in descriptions.items()
    )

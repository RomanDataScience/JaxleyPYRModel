from dataclasses import dataclass
from typing import Optional, Type

from channels_converted.channels_jaxley import (
    Cal,
    Cal4,
    CalH,
    Car,
    Cat,
    D3,
    H,
    Icand,
    Kad,
    Kap,
    Kca,
    Kd,
    Kir,
    Km,
    Kv2like,
    MyKca,
    Na3Dend,
    Nap,
    Nav16A,
    Nax,
)


@dataclass(frozen=True)
class StateSpec:
    label: str
    neuron_name: Optional[str] = None
    jaxley_name: Optional[str] = None
    neuron_source: str = "mechanism"

    def neuron_var(self) -> str:
        return self.neuron_name or self.label

    def jaxley_var(self, prefix: str) -> str:
        return self.jaxley_name or f"{prefix}_{self.label}"


@dataclass(frozen=True)
class CurrentSpec:
    label: str = "current"
    neuron_name: str = ""
    neuron_source: str = "segment"

    def neuron_var(self) -> str:
        return self.neuron_name


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    mechanism: str
    jaxley_class: Type
    states: tuple[StateSpec, ...] = ()
    current: Optional[CurrentSpec] = None
    skip_by_default: bool = False
    note: str = ""


CHANNELS: dict[str, ChannelSpec] = {
    "mykca": ChannelSpec(
        key="mykca",
        mechanism="mykca",
        jaxley_class=MyKca,
        states=(StateSpec("o"),),
        current=CurrentSpec(neuron_name="ik", neuron_source="formula"),
    ),
    "cal": ChannelSpec(
        key="cal",
        mechanism="cal",
        jaxley_class=Cal,
        states=(StateSpec("m"),),
        current=CurrentSpec(neuron_name="ica", neuron_source="mechanism"),
    ),
    "cal4": ChannelSpec(
        key="cal4",
        mechanism="cal4",
        jaxley_class=Cal4,
        states=(StateSpec("CaCon_i", neuron_name="cai", jaxley_name="CaCon_i", neuron_source="segment"),),
        skip_by_default=True,
        note=(
            "Cal4 is a reduced Jaxley Pump; the MOD file uses radial annuli, buffering, "
            "KINETIC reactions, and longitudinal diffusion."
        ),
    ),
    "calh": ChannelSpec(
        key="calh",
        mechanism="calH",
        jaxley_class=CalH,
        states=(StateSpec("m"), StateSpec("h")),
        current=CurrentSpec(neuron_name="ica", neuron_source="mechanism"),
    ),
    "car": ChannelSpec(
        key="car",
        mechanism="car",
        jaxley_class=Car,
        states=(StateSpec("m"), StateSpec("h")),
        current=CurrentSpec(neuron_name="ica", neuron_source="mechanism"),
    ),
    "cat": ChannelSpec(
        key="cat",
        mechanism="cat",
        jaxley_class=Cat,
        states=(StateSpec("m"), StateSpec("h")),
        current=CurrentSpec(neuron_name="ica", neuron_source="mechanism"),
    ),
    "d3": ChannelSpec(
        key="d3",
        mechanism="d3",
        jaxley_class=D3,
        skip_by_default=True,
        note="Geometry metadata only; no kinetic state or membrane current.",
    ),
    "h": ChannelSpec(
        key="h",
        mechanism="h",
        jaxley_class=H,
        states=(StateSpec("n"),),
        current=CurrentSpec(neuron_name="i", neuron_source="formula"),
    ),
    "icand": ChannelSpec(
        key="icand",
        mechanism="icand",
        jaxley_class=Icand,
        states=(StateSpec("Po"),),
        current=CurrentSpec(neuron_name="itrpm4", neuron_source="formula"),
    ),
    "kad": ChannelSpec(
        key="kad",
        mechanism="kad",
        jaxley_class=Kad,
        states=(StateSpec("n"), StateSpec("l")),
        current=CurrentSpec(neuron_name="i", neuron_source="formula"),
    ),
    "kap": ChannelSpec(
        key="kap",
        mechanism="kap",
        jaxley_class=Kap,
        states=(StateSpec("n"), StateSpec("l")),
        current=CurrentSpec(neuron_name="i", neuron_source="formula"),
    ),
    "kca": ChannelSpec(
        key="kca",
        mechanism="kca",
        jaxley_class=Kca,
        states=(StateSpec("m"),),
        current=CurrentSpec(neuron_name="ik", neuron_source="formula"),
    ),
    "kd": ChannelSpec(
        key="kd",
        mechanism="kd",
        jaxley_class=Kd,
        states=(StateSpec("m"), StateSpec("h")),
        current=CurrentSpec(neuron_name="i", neuron_source="formula"),
    ),
    "kir": ChannelSpec(
        key="kir",
        mechanism="kir",
        jaxley_class=Kir,
        current=CurrentSpec(neuron_name="ik", neuron_source="formula"),
    ),
    "km": ChannelSpec(
        key="km",
        mechanism="km",
        jaxley_class=Km,
        states=(StateSpec("m"),),
        current=CurrentSpec(neuron_name="ik", neuron_source="formula"),
    ),
    "kv2like": ChannelSpec(
        key="kv2like",
        mechanism="Kv2like",
        jaxley_class=Kv2like,
        states=(StateSpec("m"), StateSpec("h1"), StateSpec("h2")),
        current=CurrentSpec(neuron_name="ik", neuron_source="formula"),
    ),
    "na3dend": ChannelSpec(
        key="na3dend",
        mechanism="na3dend",
        jaxley_class=Na3Dend,
        states=(StateSpec("m"), StateSpec("h"), StateSpec("s")),
        current=CurrentSpec(neuron_name="ina", neuron_source="formula"),
    ),
    "nap": ChannelSpec(
        key="nap",
        mechanism="nap",
        jaxley_class=Nap,
        states=(StateSpec("n"),),
        current=CurrentSpec(neuron_name="ina", neuron_source="formula"),
    ),
    "na16a": ChannelSpec(
        key="na16a",
        mechanism="na16a",
        jaxley_class=Nav16A,
        states=(StateSpec("C1"), StateSpec("O1"), StateSpec("I1"), StateSpec("I2")),
        current=CurrentSpec(neuron_name="ina", neuron_source="formula"),
        note="The Jaxley version uses an implicit sparse-style Markov update to match NEURON.",
    ),
    "nax": ChannelSpec(
        key="nax",
        mechanism="nax",
        jaxley_class=Nax,
        states=(StateSpec("m"), StateSpec("h")),
        current=CurrentSpec(neuron_name="ina", neuron_source="formula"),
    ),
}


def default_channel_keys() -> list[str]:
    return [key for key, spec in CHANNELS.items() if not spec.skip_by_default]


def channel_keys(include_skipped: bool = False) -> list[str]:
    if include_skipped:
        return list(CHANNELS)
    return default_channel_keys()

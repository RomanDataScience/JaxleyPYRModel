import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

import neuron  # noqa: E402
from neuron import h  # noqa: E402

import numpy as np

from channels_converted.validation.channel_registry import ChannelSpec
from channels_converted.validation.jaxley_runner import build_jaxley_params, scalar
from channels_converted.validation.protocols import VoltageProtocol


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOD_DIR = REPO_ROOT / "channels_converted" / "mod"
_LOADED_MOD_DIRS: set[Path] = set()


def nrnivmodl_path() -> str:
    env_path = Path(sys.executable).with_name("nrnivmodl")
    if env_path.exists():
        return str(env_path)
    found = shutil.which("nrnivmodl")
    if found:
        return found
    raise FileNotFoundError("Could not find nrnivmodl. Activate the NEURON environment first.")


def compile_mods(mod_dir: Path = DEFAULT_MOD_DIR) -> None:
    env = dict(os.environ)
    env.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    subprocess.run([nrnivmodl_path()], cwd=mod_dir, env=env, check=True)


def load_mechanisms(mod_dir: Path = DEFAULT_MOD_DIR) -> None:
    mod_dir = mod_dir.resolve()
    if mod_dir in _LOADED_MOD_DIRS:
        return
    if not neuron.load_mechanisms(str(mod_dir)):
        compiled = sorted(mod_dir.glob("*/special"))
        if not compiled:
            raise RuntimeError(
                f"No compiled NEURON mechanisms found under {mod_dir}. "
                "Run with --compile or run nrnivmodl in that directory."
            )
    _LOADED_MOD_DIRS.add(mod_dir)


def _delete_sections() -> None:
    for section in list(h.allsec()):
        h.delete_section(sec=section)


def _set_attr_if_present(target, attr: str, value: float) -> bool:
    try:
        present = hasattr(target, attr)
    except TypeError:
        return False
    if present:
        setattr(target, attr, value)
        return True
    return False


def _set_common_neuron_values(seg, mechanism: str, value_by_name: Mapping[str, float]) -> None:
    common_targets = {
        "eNa": "ena",
        "eK": "ek",
        "eCa": "eca",
        "CaCon_i": "cai",
        "CaCon_e": "cao",
    }
    for source, target in common_targets.items():
        if source in value_by_name:
            _set_attr_if_present(seg, target, value_by_name[source])
            _set_attr_if_present(h, f"{target}_{mechanism}", value_by_name[source])

    if "celsius" in value_by_name:
        h.celsius = value_by_name["celsius"]


def _apply_jaxley_defaults_to_neuron(spec: ChannelSpec, seg, mech, param_overrides=None) -> None:
    channel = spec.jaxley_class()
    prefix = channel.name
    params = build_jaxley_params(spec, param_overrides)
    values = {name: scalar(value) for name, value in params.items()}

    _set_common_neuron_values(seg, spec.mechanism, values)
    if spec.key == "kir":
        kir_ek = values.get(f"{prefix}_ek", values.get("eK"))
        if kir_ek is not None:
            _set_attr_if_present(mech, "ek", kir_ek)
    if spec.key == "icand":
        # icand.mod has USEION ca commented out; INITIAL copies the mechanism
        # global parameter cai_icand into can. Keep this independent from CaCon_i.
        _set_attr_if_present(h, "cai_icand", values.get(f"{prefix}_can", 0.0))

    for name, value in values.items():
        if name in {"eNa", "eK", "eCa", "CaCon_i", "CaCon_e", "i_Ca", "celsius"}:
            continue
        unprefixed = name[len(prefix) + 1 :] if name.startswith(f"{prefix}_") else name
        if not _set_attr_if_present(mech, unprefixed, value):
            _set_attr_if_present(h, f"{unprefixed}_{spec.mechanism}", value)


def _read_value(seg, mech, source: str, name: str) -> float:
    target = seg if source == "segment" else mech
    if not hasattr(target, name):
        raise AttributeError(f"NEURON object has no attribute '{name}'.")
    return float(getattr(target, name))


def _formula_current(spec: ChannelSpec, seg, mech) -> float:
    if spec.key == "h":
        return float(mech.gbar * mech.n * (seg.v - getattr(mech, "eh", -10.0)))
    if spec.key == "icand":
        return float(mech.gbar * mech.Po * (seg.v - mech.erev))
    if spec.key == "mykca":
        return float(mech.gkbar * mech.o ** getattr(mech, "st", 1.0) * (seg.v - seg.ek))
    if spec.key in {"kad", "kap"}:
        return float(mech.gkabar * mech.n * mech.l * (seg.v - seg.ek))
    if spec.key == "kca":
        return float(mech.gbar * mech.m**3 * (seg.v - seg.ek))
    if spec.key == "kd":
        return float(mech.gbar * mech.m * mech.h * (seg.v - seg.ek))
    if spec.key == "kir":
        arg = -(seg.v - mech.ek + mech.Offset) / mech.Slope
        if arg < -50.0:
            conductance = mech.gbar
        elif arg > 50.0:
            conductance = 0.0
        else:
            conductance = mech.gbar / (1.0 + np.exp(arg))
        return float(conductance * (seg.v - mech.ek))
    if spec.key == "km":
        return float(mech.gbar * mech.m ** getattr(mech, "st", 1.0) * (seg.v - seg.ek))
    if spec.key == "kv2like":
        return float(mech.gbar * mech.m**2 * (0.5 * mech.h1 + 0.5 * mech.h2) * (seg.v - seg.ek))
    if spec.key == "nap":
        return float(mech.gnabar * mech.n**3 * (seg.v - seg.ena))
    if spec.key == "nax":
        return float(mech.gbar * mech.m**3 * mech.h * (seg.v - seg.ena))
    if spec.key == "na3dend":
        return float(mech.gbar * mech.m**3 * mech.h * mech.s * (seg.v - seg.ena))
    if spec.key == "na16a":
        return float(mech.gbar * mech.O1 * (seg.v - seg.ena))
    raise NotImplementedError(f"No formula current implemented for '{spec.key}'.")


def _record(spec: ChannelSpec, seg, mech, output, index: int) -> None:
    for state in spec.states:
        output[state.label][index] = _read_value(
            seg,
            mech,
            state.neuron_source,
            state.neuron_var(),
        )

    if spec.current is not None:
        if spec.current.neuron_source == "formula":
            output[spec.current.label][index] = _formula_current(spec, seg, mech)
        else:
            output[spec.current.label][index] = _read_value(
                seg,
                mech,
                spec.current.neuron_source,
                spec.current.neuron_var(),
            )


def run_neuron_channel(
    spec: ChannelSpec,
    protocol: VoltageProtocol,
    *,
    mod_dir: Path = DEFAULT_MOD_DIR,
    param_overrides: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    load_mechanisms(mod_dir)
    _delete_sections()

    cvode = h.CVode()
    cvode.active(0)
    h.dt = protocol.dt

    section = h.Section(name=f"validation_{spec.key}")
    section.L = 20.0
    section.diam = 20.0
    section.nseg = 1
    section.cm = 1.0
    section.Ra = 100.0
    section.insert(spec.mechanism)

    seg = section(0.5)
    mech = getattr(seg, spec.mechanism)
    _apply_jaxley_defaults_to_neuron(spec, seg, mech, param_overrides)

    clamp = h.SEClamp(seg)
    clamp.rs = 1e-6
    clamp.dur1 = float(protocol.time[-1] + 2.0 * protocol.dt)
    clamp.amp1 = float(protocol.voltage[0])

    output = {state.label: np.zeros_like(protocol.time) for state in spec.states}
    if spec.current is not None:
        output[spec.current.label] = np.zeros_like(protocol.time)

    h.finitialize(float(protocol.voltage[0]))
    h.fcurrent()
    _record(spec, seg, mech, output, 0)

    for index in range(1, protocol.time.size):
        clamp.amp1 = float(protocol.voltage[index])
        h.fadvance()
        _record(spec, seg, mech, output, index)

    return output

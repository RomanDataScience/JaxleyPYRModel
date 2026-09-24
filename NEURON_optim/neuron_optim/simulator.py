"""NEURON-backed Combe simulation and parameter application."""

from __future__ import annotations

import os
from pathlib import Path
import zipfile

import numpy as np

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

from .data import Trace
from .mechanisms import ensure_patched_mod_dir
from .objective import SimulationOutput


class NeuronUnavailable(RuntimeError):
    pass


def make_simulator(backend: str, *, d_lambda: float = 0.3, quiet: bool = True,
                   morphology_source: str = "hoc"):
    """Construct the configured simulator without importing both backends."""
    normalized = str(backend).lower()
    if normalized == "neuron":
        return NeuronSimulator(d_lambda=d_lambda, quiet=quiet)
    if normalized == "jaxley":
        from .jaxley_simulator import JaxleySimulator
        return JaxleySimulator(d_lambda=d_lambda, quiet=quiet,
                               morphology_source=morphology_source)
    raise ValueError(f"Unsupported simulation backend: {backend}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_combe_source(cache_dir: Path | None = None) -> Path:
    root = repository_root()
    source = root / "Combe2023"
    if (source / "pc2b" / "cell.hoc").exists():
        return source
    archive = root / "Combe2023.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Missing Combe2023 source or archive: {archive}")
    target = cache_dir or (root / ".cache" / "Combe2023")
    target.mkdir(parents=True, exist_ok=True)
    extracted = target / "267599-main"
    if not (extracted / "pc2b" / "cell.hoc").exists():
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(target)
    return extracted


def _mechanism(seg, name: str):
    try:
        return getattr(seg, name)
    except (AttributeError, RuntimeError):
        return None


def _set_mechanism(seg, name: str, attr: str, value: float) -> None:
    mech = _mechanism(seg, name)
    if mech is not None and hasattr(mech, attr):
        setattr(mech, attr, float(value))


def _group(section_name: str) -> str:
    base = section_name.split("[", 1)[0].lower()
    if base == "soma":
        return "soma"
    if base == "axon":
        return "axon"
    if base in {"dend", "basal"}:
        return "basal"
    return "apical"


def _sigmoid_distance(distance: float, soma: float, tuft: float,
                      half: float, slope: float) -> float:
    argument = np.clip((distance - half) / slope, -50.0, 50.0)
    return float(tuft + (soma - tuft) / (1.0 + np.exp(argument)))


def apply_parameters(soma, values: dict[str, float], h) -> None:
    """Apply Combe values to the HOC cell's RANGE variables.

    The original setup file assigns the distributions once during construction;
    CMA candidates therefore need those distributions replayed explicitly.
    """
    for key, value in values.items():
        try:
            setattr(h, key, float(value))
        except (AttributeError, LookupError, TypeError):
            pass

    h.distance(0.0, sec=soma)
    for sec in list(h.allsec()):
        group = _group(sec.name())
        for seg in sec:
            distance = float(h.distance(seg.x, sec=sec))
            rm = _sigmoid_distance(distance, values["RmSoma"], values["RmTuft"],
                                   values["DistHalfRm"], values["SlopeRm"])
            ra = _sigmoid_distance(distance, values["RaSoma"], values["RaTuft"],
                                   values["DistHalfRa"], values["SlopeRa"])
            spine = (values["SpineFactorBasal"] if group == "basal" else
                     values["SpineFactorTuft"] if group == "apical" else 1.0)
            if group == "apical" and distance > 394.0:
                rm = _sigmoid_distance(394.0, values["RmSoma"], values["RmTuft"],
                                       values["DistHalfRm"], values["SlopeRm"])
                ra = _sigmoid_distance(394.0, values["RaSoma"], values["RaTuft"],
                                       values["DistHalfRa"], values["SlopeRa"])
            try:
                sec.Ra = ra
                seg.g_pas = spine / rm
                seg.e_pas = values["Epas"]
                seg.cm = values["CmSoma"] * spine
            except (AttributeError, RuntimeError):
                pass

            if group == "soma":
                assignments = {
                    "icand": {"gbar": values["icangbar"]},
                    "na16a": {
                        "gbar": values["gna"],
                        "persist": values["persist"],
                        "persist_vhalf": values["nav16_persist_vhalf"],
                        "persist_k": values["nav16_persist_k"],
                        "C1O1v2": values["nav16_C1O1v2"],
                        "C1O1k2": values["nav16_C1O1k2"],
                        "C1I1b2": values["nav16_C1I1b2"],
                        "C1I1v2": values["nav16_C1I1v2"],
                        "C1I1k2": values["nav16_C1I1k2"],
                        "O1I1b2": values["nav16_O1I1b2"],
                        "O1I1v2": values["nav16_O1I1v2"],
                        "O1I1k2": values["nav16_O1I1k2"],
                    },
                    "kd": {"gbar": values["gkdrsoma"]},
                    "Kv2like": {"gbar": values["gkv2soma"]},
                    "h": {"gbar": values["soma_hbar"]},
                    "kap": {"gkabar": values["soma_kap"]},
                    "km": {"gbar": values["soma_km"]},
                    "cal": {"gcalbar": 0.1 * values["soma_caL"]},
                    "cat": {"gcatbar": values["soma_caT"]},
                    "car": {"gcabar": values["gsomacar"]},
                    "kca": {"gbar": 0.5 * values["soma_kca"]},
                    "mykca": {"gkbar": 5.5 * values["mykca_init"]},
                }
            elif group == "axon":
                assignments = {
                    "nax": {"gbar": values["gna"] * values["AXNa"]},
                    "kd": {"gbar": values["axongkdr"]},
                    "km": {"gbar": 3.0 * values["soma_km"]},
                    "kap": {"gkabar": values["axon_kap"]},
                    "Kv2like": {"gbar": values["gkv2axon"]},
                }
            elif group == "basal":
                assignments = {
                    "na3dend": {"gbar": values["gnadend"]},
                    "h": {"gbar": values["soma_hbar"]},
                    "kd": {"gbar": values["gkdrdend"]},
                    "kap": {"gkabar": values["basal_kap"]},
                    "Kv2like": {"gbar": values["gkv2"] * values["gkv2scale"]},
                    "kir": {"gbar": values["KirGbar"] * min(distance / 40.0, 1.0)},
                }
            else:
                hbar = values["soma_hbar"] * (1.0 + 1.2 * min(distance, 100.0) / 100.0)
                assignments = {
                    "icand": {"gbar": values["icangbar"]},
                    "car": {"gcabar": 0.1 * values["soma_car"]},
                    "calH": {"gcalbar": (2.0 if distance > 50.0 else 0.1) * values["soma_caLH"]},
                    "cat": {"gcatbar": values["soma_caT"]},
                    "kca": {"gbar": (5.0 if 50.0 < distance < 200.0 else 0.5) * values["soma_kca"]},
                    "mykca": {"gkbar": (2.0 if 50.0 < distance < 200.0 else 0.5) * values["mykca_init"]},
                    "h": {"gbar": hbar},
                    "kap": {"gkabar": values["soma_kap"] * (1.0 + distance / 100.0)},
                    "kad": {"gkabar": values["soma_kad"] * (1.0 + distance / 100.0)},
                    "Kv2like": {"gbar": values["gkv2"] * (values["gkv2scale"] if distance > 100.0 else 1.0)},
                    "na16a": {
                        "gbar": values["gnadend"],
                        "persist": values["persist"],
                        "persist_vhalf": values["nav16_persist_vhalf"],
                        "persist_k": values["nav16_persist_k"],
                        "C1O1v2": values["nav16_C1O1v2"],
                        "C1O1k2": values["nav16_C1O1k2"],
                        "C1I1b2": values["nav16_C1I1b2"],
                        "C1I1v2": values["nav16_C1I1v2"],
                        "C1I1k2": values["nav16_C1I1k2"],
                        "O1I1b2": values["nav16_O1I1b2"],
                        "O1I1v2": values["nav16_O1I1v2"],
                        "O1I1k2": values["nav16_O1I1k2"],
                    },
                    "kd": {"gbar": values["gkdrapical"]},
                    "km": {"gbar": values["soma_km"]},
                    "kir": {"gbar": values["KirGbar"] * min(distance / 100.0, 1.0)},
                }
            for mechanism, attrs in assignments.items():
                for attr, value in attrs.items():
                    _set_mechanism(seg, mechanism, attr, value)

            # These fields are exposed by the patched kinetic mechanisms. With
            # the historical compiled mechanisms they simply remain at one.
            for mechanism, attr, key in (
                ("kd", "deactivation_tau_scale", "kd_deactivation_tau_scale"),
                ("na16a", "fast_inactivation_tau_scale", "nat_fast_inactivation_tau_scale"),
                ("na16a", "slow_recovery_tau_scale", "nat_slow_recovery_tau_scale"),
                ("nax", "fast_inactivation_tau_scale", "nat_fast_inactivation_tau_scale"),
                ("na3dend", "fast_inactivation_tau_scale", "nat_fast_inactivation_tau_scale"),
                ("h", "tau_scale", "h_tau_scale"),
            ):
                _set_mechanism(seg, mechanism, attr, values[key])


class NeuronSimulator:
    def __init__(self, *, d_lambda: float = 0.3, quiet: bool = True,
                 combe_dir: Path | None = None, mod_dir: Path | None = None,
                 reuse_model: bool = True):
        self.d_lambda = d_lambda
        self.quiet = quiet
        self.combe_dir = combe_dir or ensure_combe_source()
        self.mod_dir = mod_dir or ensure_patched_mod_dir()
        self.reuse_model = bool(reuse_model)
        self._h = None
        self._soma = None
        self._build_combe_neuron_model = None

    def _model(self):
        try:
            import sys
            root = str(repository_root())
            if root not in sys.path:
                sys.path.insert(0, root)
            from neuron import h
            from channels_converted.modelComparison.neuron_model import build_combe_neuron_model
        except Exception as exc:  # pragma: no cover - depends on external NEURON install
            raise NeuronUnavailable(
                "NEURON is required for production runs. Install NEURON and compile "
                "channels_converted/mod before running this pipeline."
            ) from exc
        if self.reuse_model and self._soma is not None:
            return h, self._soma
        soma = build_combe_neuron_model(quiet=self.quiet, d_lambda=self.d_lambda,
                                        combe_dir=self.combe_dir, mod_dir=self.mod_dir)
        if self.reuse_model:
            self._h = h
            self._soma = soma
            self._build_combe_neuron_model = build_combe_neuron_model
        return h, soma

    def simulate_many(self, traces: list[Trace], values: dict[str, float],
                      *, v_init_mode: str = "observed_first_sample") -> list[SimulationOutput]:
        h, soma = self._model()
        apply_parameters(soma, values, h)
        h.CVode().active(0)
        outputs = []
        for trace in traces:
            if trace.time_ms.size < 2:
                raise ValueError("Trace has fewer than two samples")
            dt = float(np.median(np.diff(trace.time_ms)))
            time = np.asarray(trace.time_ms - trace.time_ms[0], dtype=float)
            current = np.asarray(trace.current_nA, dtype=float)
            h.dt = dt
            h.tstop = float(time[-1])
            clamp = h.IClamp(soma(0.5))
            # IClamp defaults to a zero-duration pulse.  Vector.play updates
            # ``amp`` but does not change the clamp's delay/duration window,
            # so without these assignments the recorded current is never
            # delivered to the cell.
            clamp.delay = 0.0
            clamp.dur = max(float(time[-1]) + dt, dt)
            tvec = h.Vector(time)
            ivec = h.Vector(current)
            ivec.play(clamp._ref_amp, tvec, 1)
            recorded_t = h.Vector().record(h._ref_t)
            recorded_v = h.Vector().record(soma(0.5)._ref_v)
            initial = float(trace.voltage_mV[0]) if v_init_mode == "observed_first_sample" else float(values["Epas"])
            h.finitialize(initial)
            h.fcurrent()
            h.continuerun(float(time[-1]))
            # NEURON vectors reuse their underlying storage between runs;
            # copy before appending so a later trace cannot overwrite an
            # earlier SimulationOutput.
            out_t = np.asarray(recorded_t, dtype=float).copy()
            out_v = np.asarray(recorded_v, dtype=float).copy()
            if out_t.size < 2 or not np.isfinite(out_v).all():
                raise RuntimeError("NEURON returned an invalid voltage trace")
            outputs.append(SimulationOutput(out_t, out_v))
            try:
                ivec.play_remove()
            except AttributeError:
                pass
            clamp = None
        return outputs

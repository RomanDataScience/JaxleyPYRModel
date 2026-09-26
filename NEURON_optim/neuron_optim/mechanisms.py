"""Build a private NEURON MOD directory with the Combe kinetic scales."""

from __future__ import annotations

import fcntl
from pathlib import Path
import shutil
import subprocess


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _replace(text: str, old: str, new: str, name: str) -> str:
    # A missing target would silently leave the kinetic scale without effect.
    if old not in text:
        raise RuntimeError(f"Cannot patch {name}: expected text not found: {old!r}")
    return text.replace(old, new)


def _patch_sources(source: Path, target: Path) -> None:
    for path in source.glob("*.mod"):
        text = path.read_text(encoding="utf-8")
        name = path.name
        if name == "kd.mod":
            text = _replace(text, "RANGE gk, gbar, i", "RANGE gk, gbar, i, deactivation_tau_scale", name)
            text = _replace(text, "gbar = 0.1", "deactivation_tau_scale = 1\n\tgbar = 0.1", name)
            text = _replace(text, "mtau = 0.6", "mtau = 0.6 * deactivation_tau_scale", name)
        elif name == "h.mod":
            text = _replace(text, "RANGE gbar,vhalf,K,taun,ninf,g,i", "RANGE gbar,vhalf,K,taun,ninf,g,i,tau_scale", name)
            text = _replace(text, "vhalf=-81 (mV)", "tau_scale=1\n    vhalf=-81 (mV)", name)
            marker = "if (taun<5) {"
            text = _replace(text, marker, "taun = taun * tau_scale\n\n    " + marker, name)
        elif name in {"nax.mod", "na3dend.mod"}:
            text = _replace(text, "RANGE  gbar", "RANGE  gbar, fast_inactivation_tau_scale", name)
            text = _replace(text, "gbar = 0.010", "fast_inactivation_tau_scale = 1\n\tgbar = 0.010", name)
            text = _replace(text, "if (htau<hmin)", "htau = htau * fast_inactivation_tau_scale\n        if (htau<hmin)", name)
        elif name == "Nav16_a.mod":
            text = _replace(
                text,
                "RANGE gbar, ina, g, dist, slowdown",
                "RANGE gbar, ina, g, dist, slowdown, fast_inactivation_tau_scale, slow_recovery_tau_scale",
                name,
            )
            text = _replace(text, "gbar  = 0.1", "fast_inactivation_tau_scale = 1\n\tslow_recovery_tau_scale = 1\n\tgbar  = 0.1", name)
            # The Jaxley Nav16A.rates() applies the same scales to the same rates.
            for old, new in (
                ("rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2))",
                 "rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2)) / fast_inactivation_tau_scale"),
                ("I1C1_a = Q10 * (rates2(v, I1C1b1, I1C1v1, I1C1k1))",
                 "I1C1_a = Q10 * (rates2(v, I1C1b1, I1C1v1, I1C1k1)) / fast_inactivation_tau_scale"),
                ("C1I1_a = Q10 * (rates2(v, C1I1b2, C1I1v2, C1I1k2))",
                 "C1I1_a = Q10 * (rates2(v, C1I1b2, C1I1v2, C1I1k2)) / fast_inactivation_tau_scale"),
                ("I1I2_a = slowdown * dist * Q10 * (rates2(v, I1I2b2, I1I2v2, I1I2k2))",
                 "I1I2_a = slowdown * dist * Q10 * (rates2(v, I1I2b2, I1I2v2, I1I2k2)) / slow_recovery_tau_scale"),
                ("I2I1_a = slowdown * Q10 * (rates2(v, I2I1b1, I2I1v1, I2I1k1))",
                 "I2I1_a = slowdown * Q10 * (rates2(v, I2I1b1, I2I1v1, I2I1k1)) / slow_recovery_tau_scale"),
            ):
                text = _replace(text, old, new, name)
        (target / name).write_text(text, encoding="utf-8")


def ensure_patched_mod_dir(*, force_recompile: bool = False) -> Path:
    root = _repo_root()
    source = root / "channels_converted" / "mod"
    target = root / ".cache" / "neuron_optim_mod"
    target.mkdir(parents=True, exist_ok=True)
    lock_path = target / ".build.lock"
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        # Rebuild when any copied MOD source is newer than the compiled
        # library. Otherwise NEURON could silently keep loading an older
        # version of ``na16a``.
        compiled = list(target.glob("*/special"))
        source_mods = list(source.glob("*.mod"))
        if (
            not force_recompile
            and compiled
            and source_mods
            and max(path.stat().st_mtime for path in source_mods)
            <= max(path.stat().st_mtime for path in compiled)
        ):
            return target
        for path in target.glob("*.mod"):
            path.unlink()
        if force_recompile:
            # Only remove generated compiler directories.  The copied MOD
            # sources and the build lock are recreated/retained below.
            for generated in target.iterdir():
                if generated.is_dir():
                    shutil.rmtree(generated)
        _patch_sources(source, target)
        try:
            subprocess.run(["nrnivmodl"], cwd=target, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("nrnivmodl is required in the Jaxley environment") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Could not compile private Combe MOD mechanisms:\n{exc.stdout}") from exc
        return target

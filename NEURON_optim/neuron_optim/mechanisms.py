"""Build a private NEURON MOD directory with the Combe kinetic scales."""

from __future__ import annotations

import fcntl
from pathlib import Path
import shutil
import subprocess


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _patch_sources(source: Path, target: Path) -> None:
    for path in source.glob("*.mod"):
        text = path.read_text(encoding="utf-8")
        name = path.name
        if name == "kd.mod":
            text = text.replace("RANGE gk, gbar, i", "RANGE gk, gbar, i, deactivation_tau_scale")
            text = text.replace("gbar = 0.1", "gbar = 0.1\n\tdeactivation_tau_scale = 1")
            text = text.replace("mtau = 0.6", "mtau = 0.6 * deactivation_tau_scale")
        elif name == "h.mod":
            text = text.replace("RANGE gbar,vhalf,K,taun,ninf,g,i", "RANGE gbar,vhalf,K,taun,ninf,g,i,tau_scale")
            text = text.replace("vhalf=-81 (mV)", "vhalf=-81 (mV)\n    tau_scale=1")
            text = text.replace("taun=2*(1/(MyExp", "taun=2*(1/(MyExp")
            marker = "if (taun<5) {"
            text = text.replace(marker, "taun = taun * tau_scale\n\n    " + marker)
        elif name in {"nax.mod", "na3dend.mod"}:
            text = text.replace("RANGE  gbar", "RANGE  gbar, fast_inactivation_tau_scale")
            text = text.replace("RANGE  gbar, ar2", "RANGE  gbar, ar2, fast_inactivation_tau_scale")
            text = text.replace("gbar = 0.010", "gbar = 0.010\n\tfast_inactivation_tau_scale = 1")
            text = text.replace("if (htau<hmin)", "htau = htau * fast_inactivation_tau_scale\n        if (htau<hmin)")
        elif name == "Nav16_a.mod":
            text = text.replace(
                "RANGE gbar, ina, g, dist, persist, slowdown, C1O1v2, C1O1k2, C1I1b2, C1I1v2, C1I1k2, O1I1b2, O1I1v2, O1I1k2, I2init",
                "RANGE gbar, ina, g, dist, persist, slowdown, C1O1v2, C1O1k2, C1I1b2, C1I1v2, C1I1k2, O1I1b2, O1I1v2, O1I1k2, I2init, fast_inactivation_tau_scale, slow_recovery_tau_scale",
            )
            text = text.replace("gbar  = 0.1", "gbar  = 0.1\n\tfast_inactivation_tau_scale = 1\n\tslow_recovery_tau_scale = 1")
            text = text.replace("O1I1_a = 0.5*Q10*", "O1I1_a = 0.5*Q10*")
            text = text.replace("rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2))",
                                "rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2)) / fast_inactivation_tau_scale")
            text = text.replace("I1C1_a = Q10*(rates2(v, I1C1b1, I1C1v1, I1C1k1))",
                                "I1C1_a = Q10*(rates2(v, I1C1b1, I1C1v1, I1C1k1)) / fast_inactivation_tau_scale")
            text = text.replace("C1I1_a = Q10*(rates2(v, C1I1b2, C1I1v2, C1I1k2))",
                                "C1I1_a = Q10*(rates2(v, C1I1b2, C1I1v2, C1I1k2)) / fast_inactivation_tau_scale")
            text = text.replace("I1I2_a = slowdown*dist*Q10*(rates2(v, I1I2b2, I1I2v2, I1I2k2))",
                                "I1I2_a = slowdown*dist*Q10*(rates2(v, I1I2b2, I1I2v2, I1I2k2)) / slow_recovery_tau_scale")
            text = text.replace("I2I1_a = slowdown*Q10*(rates2(v, I2I1b1, I2I1v1, I2I1k1))",
                                "I2I1_a = slowdown*Q10*(rates2(v, I2I1b1, I2I1v1, I2I1k1)) / slow_recovery_tau_scale")
        (target / name).write_text(text, encoding="utf-8")


def ensure_patched_mod_dir() -> Path:
    root = _repo_root()
    source = root / "channels_converted" / "mod"
    target = root / ".cache" / "neuron_optim_mod"
    target.mkdir(parents=True, exist_ok=True)
    lock_path = target / ".build.lock"
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        # Rebuild when the gated Nav1.6 source is missing or any copied MOD
        # source is newer than the compiled library. Otherwise NEURON could
        # silently keep loading an older version of ``na16a``.
        compiled = list(target.glob("*/special"))
        source_mods = list(source.glob("*.mod"))
        if (
            compiled
            and (target / "Nav16_a_vgated_persist.mod").exists()
            and source_mods
            and max(path.stat().st_mtime for path in source_mods)
            <= max(path.stat().st_mtime for path in compiled)
        ):
            return target
        for path in target.glob("*.mod"):
            path.unlink()
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

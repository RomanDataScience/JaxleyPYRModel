"""Collect enough environment state to reconstruct a run."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_provenance(repository_root: Path, device: dict) -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": sys.version,
        "packages": {
            name: _version(name)
            for name in ("jax", "jaxlib", "jaxley", "numpy", "PyYAML")
        },
        "git_revision": revision,
        "git_dirty": dirty,
        "device": device,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_RESTART_COUNT",
                "SLURMD_NODENAME",
            )
            if os.environ.get(key) is not None
        },
    }


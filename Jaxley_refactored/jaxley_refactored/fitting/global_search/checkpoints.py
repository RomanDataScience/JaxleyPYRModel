"""Atomic non-pickle CMA-ES checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from .cma_es import CMAES


class CMACheckpoint:
    def __init__(self, directory: Path, compatibility_hash: str):
        self.directory = directory
        self.compatibility_hash = compatibility_hash
        directory.mkdir(parents=True, exist_ok=True)

    def save(self, optimizer: CMAES) -> None:
        with tempfile.NamedTemporaryFile(
            dir=self.directory, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **optimizer.arrays())
        temporary.replace(self.directory / "latest.npz")
        metadata = {
            "version": 1,
            "compatibility_hash": self.compatibility_hash,
            "population_size": optimizer.population_size,
            "rng_state": optimizer.rng.bit_generator.state,
        }
        with tempfile.NamedTemporaryFile(
            "w", dir=self.directory, suffix=".json", delete=False
        ) as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.directory / "latest.json")

    def load(self, *, seed: int) -> CMAES | None:
        array_path = self.directory / "latest.npz"
        metadata_path = self.directory / "latest.json"
        if not array_path.is_file() or not metadata_path.is_file():
            return None
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("compatibility_hash") != self.compatibility_hash:
            raise ValueError("CMA checkpoint is incompatible with this hybrid run.")
        with np.load(array_path, allow_pickle=False) as arrays:
            copied = {name: arrays[name].copy() for name in arrays.files}
        return CMAES.from_arrays(
            copied,
            seed=seed,
            population_size=int(metadata["population_size"]),
            rng_state=metadata["rng_state"],
        )

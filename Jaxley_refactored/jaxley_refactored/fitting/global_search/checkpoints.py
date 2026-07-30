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
            "version": 2,
            "compatibility_hash": self.compatibility_hash,
            "population_size": optimizer.population_size,
            "parent_fraction": optimizer.parent_fraction,
            "mu": optimizer.mu,
            "rng_state": optimizer.rng.bit_generator.state,
        }
        with tempfile.NamedTemporaryFile(
            "w", dir=self.directory, suffix=".json", delete=False
        ) as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.directory / "latest.json")

    def load(
        self,
        *,
        seed: int,
        expected_parent_fraction: float | None = None,
    ) -> CMAES | None:
        array_path = self.directory / "latest.npz"
        metadata_path = self.directory / "latest.json"
        if not array_path.is_file() or not metadata_path.is_file():
            return None
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        version = int(metadata.get("version", 1))
        if version not in {1, 2}:
            raise ValueError(f"Unsupported CMA checkpoint version: {version}.")
        if metadata.get("compatibility_hash") != self.compatibility_hash:
            raise ValueError("CMA checkpoint is incompatible with this hybrid run.")
        parent_fraction = float(metadata.get("parent_fraction", 0.5))
        if (
            expected_parent_fraction is not None
            and not np.isclose(
                parent_fraction,
                expected_parent_fraction,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "CMA checkpoint parent_fraction is incompatible with this run."
            )
        with np.load(array_path, allow_pickle=False) as arrays:
            copied = {name: arrays[name].copy() for name in arrays.files}
        optimizer = CMAES.from_arrays(
            copied,
            seed=seed,
            population_size=int(metadata["population_size"]),
            parent_fraction=parent_fraction,
            rng_state=metadata["rng_state"],
        )
        if "mu" in metadata and int(metadata["mu"]) != optimizer.mu:
            raise ValueError("CMA checkpoint parent count is inconsistent.")
        return optimizer

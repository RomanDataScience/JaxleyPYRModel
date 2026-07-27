"""Atomic, versioned, non-pickle training checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .optimizer import AdamState


CHECKPOINT_VERSION = 1


class CheckpointManager:
    def __init__(self, directory: Path, compatibility_hash: str):
        self.directory = directory
        self.compatibility_hash = compatibility_hash
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.npz"

    def save(
        self,
        *,
        epoch: int,
        normalized,
        optimizer: AdamState,
        best_normalized,
        best_loss: float,
        is_best: bool = False,
    ) -> None:
        arrays = {
            "version": np.asarray(CHECKPOINT_VERSION),
            "epoch": np.asarray(epoch),
            "normalized": np.asarray(normalized),
            "first_moment": np.asarray(optimizer.first_moment),
            "second_moment": np.asarray(optimizer.second_moment),
            "optimizer_step": np.asarray(optimizer.step),
            "best_normalized": np.asarray(best_normalized),
            "best_loss": np.asarray(best_loss),
        }
        metadata = {
            "version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "compatibility_hash": self.compatibility_hash,
        }
        self._atomic_npz(self.latest_path, arrays)
        self._atomic_json(self.directory / "latest.json", metadata)
        if is_best:
            self._atomic_npz(self.directory / "best.npz", arrays)
            self._atomic_json(self.directory / "best.json", metadata)

    def load(self):
        metadata_path = self.directory / "latest.json"
        if not self.latest_path.is_file() or not metadata_path.is_file():
            return None
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("compatibility_hash") != self.compatibility_hash:
            raise ValueError(
                "Checkpoint is incompatible with the current model/data/fit config."
            )
        with np.load(self.latest_path, allow_pickle=False) as arrays:
            if int(arrays["version"]) != CHECKPOINT_VERSION:
                raise ValueError("Unsupported checkpoint version.")
            return {
                "epoch": int(arrays["epoch"]),
                "normalized": arrays["normalized"],
                "optimizer": AdamState(
                    step=int(arrays["optimizer_step"]),
                    first_moment=arrays["first_moment"],
                    second_moment=arrays["second_moment"],
                ),
                "best_normalized": arrays["best_normalized"],
                "best_loss": float(arrays["best_loss"]),
            }

    @staticmethod
    def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)

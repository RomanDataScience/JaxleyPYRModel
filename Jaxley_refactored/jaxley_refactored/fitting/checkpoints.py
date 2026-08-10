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
    def __init__(
        self,
        directory: Path,
        compatibility_hash: str,
        parameter_names: tuple[str, ...] = (),
    ):
        self.directory = directory
        self.compatibility_hash = compatibility_hash
        self.parameter_names = tuple(parameter_names)
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("Checkpoint parameter names must be unique.")
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
            "optimizer_learning_rate": np.asarray(
                np.nan
                if optimizer.current_learning_rate is None
                else optimizer.current_learning_rate
            ),
            "best_normalized": np.asarray(best_normalized),
            "best_loss": np.asarray(best_loss),
        }
        if self.parameter_names:
            if len(self.parameter_names) != len(arrays["normalized"]):
                raise ValueError(
                    "Checkpoint parameter-name count does not match parameter vector."
                )
            arrays["parameter_names"] = np.asarray(self.parameter_names)
        metadata = {
            "version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "compatibility_hash": self.compatibility_hash,
        }
        if self.parameter_names:
            metadata["parameters_normalized"] = dict(
                zip(self.parameter_names, arrays["normalized"].tolist(), strict=True)
            )
            metadata["best_parameters_normalized"] = dict(
                zip(
                    self.parameter_names,
                    arrays["best_normalized"].tolist(),
                    strict=True,
                )
            )
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
            order = self._load_order(arrays)
            return {
                "epoch": int(arrays["epoch"]),
                "normalized": arrays["normalized"][order],
                "optimizer": AdamState(
                    step=int(arrays["optimizer_step"]),
                    first_moment=arrays["first_moment"][order],
                    second_moment=arrays["second_moment"][order],
                    current_learning_rate=(
                        None
                        if "optimizer_learning_rate" not in arrays.files
                        or np.isnan(float(arrays["optimizer_learning_rate"]))
                        else float(arrays["optimizer_learning_rate"])
                    ),
                ),
                "best_normalized": arrays["best_normalized"][order],
                "best_loss": float(arrays["best_loss"]),
            }

    def _load_order(self, arrays) -> np.ndarray:
        """Map saved vectors into the current catalog order by parameter name."""
        size = len(arrays["normalized"])
        if "parameter_names" not in arrays.files:
            return np.arange(size)
        saved = tuple(str(name) for name in arrays["parameter_names"].tolist())
        if not self.parameter_names:
            return np.arange(size)
        if len(set(saved)) != len(saved) or set(saved) != set(self.parameter_names):
            raise ValueError(
                "Checkpoint parameter names are incompatible with the current fit."
            )
        positions = {name: index for index, name in enumerate(saved)}
        return np.asarray([positions[name] for name in self.parameter_names])

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

"""Optuna-backed CMA-ES adapter for the population-based optimizer loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import tempfile
import warnings

import numpy as np
import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class CMAState:
    """Small compatibility state exposed to the existing stage runner."""

    mean: np.ndarray
    sigma: float
    generation: int = 0
    evaluations: int = 0


class CMAES:
    """Use Optuna's CmaEsSampler with the existing ask/tell interface.

    Parameters are sampled in the normalized [0, 1] space. Optuna owns the
    covariance adaptation and random state; the complete Optuna study is
    checkpointed so a resumed run continues with the same sampler state.
    """

    _PARAM_PREFIX = "x"
    _CHECKPOINT_VERSION = 2

    def __init__(
        self,
        mean,
        *,
        sigma: float,
        seed: int,
        population_size: int = 30,
        parent_fraction: float = 0.5,
        study: optuna.Study | None = None,
    ):
        mean = np.asarray(mean, dtype=float)
        if mean.ndim != 1 or not mean.size:
            raise ValueError("CMA mean must be a non-empty vector")
        if not np.isfinite(mean).all() or not np.all((mean >= 0.0) & (mean <= 1.0)):
            raise ValueError("CMA mean must be finite and in [0, 1]")
        if sigma <= 0.0:
            raise ValueError("CMA sigma must be positive")
        if population_size < 2:
            raise ValueError("CMA population_size must be at least 2")
        if not np.isclose(parent_fraction, 0.5):
            raise ValueError(
                "Optuna's CmaEsSampler uses its own parent weighting; "
                "parent_fraction must be 0.5"
            )

        self.dimension = mean.size
        self.population_size = int(population_size)
        self.parent_fraction = float(parent_fraction)
        self._initial_mean = mean.copy()
        self._sigma0 = float(sigma)
        self._seed = int(seed)
        self._pending_trials: list[optuna.trial.Trial] = []

        if study is None:
            sampler = self._make_sampler(
                mean,
                sigma=self._sigma0,
                seed=self._seed,
                population_size=self.population_size,
            )
            self.study = optuna.create_study(direction="minimize", sampler=sampler)
        else:
            self.study = study

        completed = self._completed_trials()
        if len(completed) % self.population_size:
            raise ValueError("Optuna study contains an incomplete CMA generation")
        self.state = CMAState(
            mean=self._initial_mean.copy(),
            sigma=self._sigma0,
            generation=len(completed) // self.population_size,
            evaluations=len(completed),
        )

    @classmethod
    def _make_sampler(cls, mean, *, sigma: float, seed: int, population_size: int):
        x0 = {f"{cls._PARAM_PREFIX}{index}": float(value) for index, value in enumerate(mean)}
        # x0 and sigma0 are the Optuna 4/5 API for initializing CMA-ES. Keep
        # the warning local so normal optimizer output remains readable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return optuna.samplers.CmaEsSampler(
                x0=x0,
                sigma0=float(sigma),
                n_startup_trials=0,
                popsize=int(population_size),
                seed=int(seed),
                warn_independent_sampling=False,
            )

    def _completed_trials(self):
        return [
            trial
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

    def ask(self) -> np.ndarray:
        if self._pending_trials:
            raise RuntimeError("CMA population is already pending; call tell first")

        population = []
        pending = []
        for _ in range(self.population_size):
            trial = self.study.ask()
            values = [
                trial.suggest_float(f"{self._PARAM_PREFIX}{index}", 0.0, 1.0)
                for index in range(self.dimension)
            ]
            pending.append(trial)
            population.append(values)
        self._pending_trials = pending
        return np.asarray(population, dtype=float)

    def tell(self, population, losses) -> None:
        population = np.asarray(population, dtype=float)
        losses = np.asarray(losses, dtype=float)
        expected_shape = (self.population_size, self.dimension)
        if population.shape != expected_shape or losses.shape != (self.population_size,):
            raise ValueError("CMA population/loss shape mismatch")
        if not np.isfinite(population).all() or not np.all((population >= 0.0) & (population <= 1.0)):
            raise ValueError("CMA population must be finite and in [0, 1]")
        if not np.isfinite(losses).all():
            raise ValueError("CMA losses must be finite")
        if len(self._pending_trials) != self.population_size:
            raise RuntimeError("No pending CMA population; call ask first")

        for trial, loss in zip(self._pending_trials, losses, strict=True):
            self.study.tell(trial, float(loss))
        self._pending_trials = []
        # The population is fully evaluated here, so no full-study scan is
        # needed to derive these counters on every generation.
        self.state.generation += 1
        self.state.evaluations += self.population_size

    def save(self, directory: Path, compatibility_hash: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        study_path = directory / "study.pkl"
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".pkl", delete=False) as handle:
            pickle.dump(self.study, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_study_path = Path(handle.name)
        temporary_study_path.replace(study_path)

        metadata = {
            "version": self._CHECKPOINT_VERSION,
            "compatibility_hash": compatibility_hash,
            "dimension": self.dimension,
            "population_size": self.population_size,
            "parent_fraction": self.parent_fraction,
            "initial_mean": self._initial_mean.tolist(),
            "sigma0": self._sigma0,
            "seed": self._seed,
            "generation": self.state.generation,
            "evaluations": self.state.evaluations,
        }
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2)
            metadata_path = Path(handle.name)
        metadata_path.replace(directory / "latest.json")

    @classmethod
    def load(cls, directory: Path, *, seed: int, compatibility_hash: str) -> "CMAES | None":
        metadata_path = directory / "latest.json"
        study_path = directory / "study.pkl"
        if not metadata_path.exists() or not study_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") != cls._CHECKPOINT_VERSION:
            raise ValueError("CMA checkpoint uses an unsupported Optuna checkpoint version")
        if metadata.get("compatibility_hash") != compatibility_hash:
            raise ValueError("CMA checkpoint is incompatible with this run")
        if int(metadata["seed"]) != int(seed):
            raise ValueError("CMA checkpoint seed does not match this run")

        with study_path.open("rb") as handle:
            study = pickle.load(handle)
        if not isinstance(study, optuna.Study):
            raise ValueError("CMA checkpoint does not contain an Optuna study")

        return cls(
            np.asarray(metadata["initial_mean"], dtype=float),
            sigma=float(metadata["sigma0"]),
            seed=int(seed),
            population_size=int(metadata["population_size"]),
            parent_fraction=float(metadata["parent_fraction"]),
            study=study,
        )

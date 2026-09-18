"""Small bounded full-covariance CMA-ES with resumable checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile

import numpy as np


@dataclass
class CMAState:
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    path_c: np.ndarray
    path_sigma: np.ndarray
    generation: int = 0
    evaluations: int = 0


class CMAES:
    def __init__(self, mean, *, sigma: float, seed: int, population_size: int = 30,
                 parent_fraction: float = 0.5, state: CMAState | None = None,
                 rng_state: dict | None = None):
        mean = np.asarray(mean, dtype=float)
        if mean.ndim != 1 or not mean.size:
            raise ValueError("CMA mean must be a non-empty vector")
        self.dimension = mean.size
        self.population_size = int(population_size)
        self.parent_fraction = float(parent_fraction)
        self.mu = max(1, int(math.floor(self.population_size * self.parent_fraction)))
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        n = self.dimension
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + self.mu_eff))
        self.damping = 1.0 + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        self.state = state or CMAState(mean.copy(), float(sigma), np.eye(n), np.zeros(n), np.zeros(n))
        self.rng = np.random.default_rng(seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _eigensystem(self):
        values, vectors = np.linalg.eigh(0.5 * (self.state.covariance + self.state.covariance.T))
        values = np.maximum(values, 1e-14)
        return vectors @ np.diag(np.sqrt(values)), vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T

    def ask(self) -> np.ndarray:
        transform, _ = self._eigensystem()
        population = []
        for _ in range(self.population_size):
            for _attempt in range(1000):
                candidate = self.state.mean + self.state.sigma * (transform @ self.rng.standard_normal(self.dimension))
                if np.all((candidate >= 0.0) & (candidate <= 1.0)):
                    break
            population.append(np.clip(candidate, 0.0, 1.0))
        return np.asarray(population)

    def tell(self, population, losses) -> None:
        population = np.asarray(population, dtype=float)
        losses = np.asarray(losses, dtype=float)
        if population.shape != (self.population_size, self.dimension) or losses.shape != (self.population_size,):
            raise ValueError("CMA population/loss shape mismatch")
        if not np.isfinite(losses).all():
            raise ValueError("CMA losses must be finite")
        order = np.argsort(losses, kind="stable")
        selected = population[order[:self.mu]]
        old_mean = self.state.mean.copy()
        new_mean = np.sum(self.weights[:, None] * selected, axis=0)
        y = (new_mean - old_mean) / max(self.state.sigma, 1e-15)
        _, inverse_sqrt = self._eigensystem()
        self.state.path_sigma = (1.0 - self.c_sigma) * self.state.path_sigma + math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * (inverse_sqrt @ y)
        norm_path = np.linalg.norm(self.state.path_sigma)
        generation = self.state.generation + 1
        decay = math.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * generation))
        h_sigma = float(norm_path / max(decay, 1e-12) < (1.4 + 2.0 / (self.dimension + 1.0)) * self.chi_n)
        self.state.path_c = (1.0 - self.c_c) * self.state.path_c + h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y
        steps = (selected - old_mean) / max(self.state.sigma, 1e-15)
        rank_mu = sum(weight * np.outer(step, step) for weight, step in zip(self.weights, steps, strict=True))
        covariance_scale = 1.0 - self.c1 - self.c_mu + self.c1 * (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        self.state.covariance = covariance_scale * self.state.covariance + self.c1 * np.outer(self.state.path_c, self.state.path_c) + self.c_mu * rank_mu
        self.state.mean = np.clip(new_mean, 0.0, 1.0)
        self.state.sigma *= math.exp((self.c_sigma / self.damping) * (norm_path / self.chi_n - 1.0))
        self.state.generation = generation
        self.state.evaluations += self.population_size

    def save(self, directory: Path, compatibility_hash: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as handle:
            array_path = Path(handle.name)
        np.savez_compressed(array_path, mean=self.state.mean, sigma=np.asarray(self.state.sigma), covariance=self.state.covariance,
                            path_c=self.state.path_c, path_sigma=self.state.path_sigma,
                            generation=np.asarray(self.state.generation), evaluations=np.asarray(self.state.evaluations))
        array_path.replace(directory / "latest.npz")
        metadata = {"version": 1, "compatibility_hash": compatibility_hash,
                    "population_size": self.population_size, "parent_fraction": self.parent_fraction,
                    "rng_state": self.rng.bit_generator.state}
        with tempfile.NamedTemporaryFile("w", dir=directory, suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            metadata_path = Path(handle.name)
        metadata_path.replace(directory / "latest.json")

    @classmethod
    def load(cls, directory: Path, *, seed: int, compatibility_hash: str) -> "CMAES | None":
        if not (directory / "latest.npz").exists() or not (directory / "latest.json").exists():
            return None
        metadata = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
        if metadata.get("compatibility_hash") != compatibility_hash:
            raise ValueError("CMA checkpoint is incompatible with this run")
        with np.load(directory / "latest.npz", allow_pickle=False) as values:
            state = CMAState(mean=values["mean"].copy(), sigma=float(values["sigma"]), covariance=values["covariance"].copy(),
                             path_c=values["path_c"].copy(), path_sigma=values["path_sigma"].copy(),
                             generation=int(values["generation"]), evaluations=int(values["evaluations"]))
        return cls(state.mean, sigma=state.sigma, seed=seed, population_size=int(metadata["population_size"]),
                   parent_fraction=float(metadata["parent_fraction"]), state=state, rng_state=metadata["rng_state"])

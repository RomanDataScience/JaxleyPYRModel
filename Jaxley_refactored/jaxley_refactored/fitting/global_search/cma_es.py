"""Small full-covariance CMA-ES with deterministic non-pickle state."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    """Canonical bounded ask/tell CMA-ES in normalized coordinates."""

    def __init__(
        self,
        mean,
        *,
        sigma: float,
        seed: int,
        population_size: int = 0,
        parent_fraction: float = 0.5,
        state: CMAState | None = None,
        rng_state: dict | None = None,
    ):
        mean = np.asarray(mean, dtype=float)
        if mean.ndim != 1 or not len(mean):
            raise ValueError("CMA mean must be a non-empty vector.")
        self.dimension = len(mean)
        self.population_size = population_size or (
            4 + int(math.floor(3.0 * math.log(self.dimension)))
        )
        if self.population_size < 2:
            raise ValueError("CMA population_size must be at least 2.")
        self.parent_fraction = float(parent_fraction)
        if (
            not math.isfinite(self.parent_fraction)
            or not 0.0 < self.parent_fraction <= 1.0
        ):
            raise ValueError("CMA parent_fraction must be in the interval (0, 1].")
        # Use the largest whole parent set no larger than the requested
        # fraction, while retaining at least one parent for small smoke runs.
        self.mu = max(
            1,
            int(math.floor(self.population_size * self.parent_fraction)),
        )
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        n = self.dimension
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((n + 2.0) ** 2 + self.mu_eff),
        )
        self.damping = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self.c_sigma
        )
        self.chi_n = math.sqrt(n) * (
            1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2)
        )
        self.state = state or CMAState(
            mean=mean.copy(),
            sigma=float(sigma),
            covariance=np.eye(n),
            path_c=np.zeros(n),
            path_sigma=np.zeros(n),
        )
        self.rng = np.random.default_rng(seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _eigensystem(self):
        values, vectors = np.linalg.eigh(
            0.5 * (self.state.covariance + self.state.covariance.T)
        )
        values = np.maximum(values, 1e-14)
        transform = vectors @ np.diag(np.sqrt(values))
        inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
        return transform, inverse_sqrt

    def ask(self) -> np.ndarray:
        transform, _ = self._eigensystem()
        population = []
        for _ in range(self.population_size):
            for _attempt in range(1000):
                candidate = self.state.mean + self.state.sigma * (
                    transform @ self.rng.standard_normal(self.dimension)
                )
                if np.all((candidate >= 0.0) & (candidate <= 1.0)):
                    population.append(candidate)
                    break
            else:
                population.append(np.clip(candidate, 0.0, 1.0))
        return np.asarray(population)

    def tell(self, population, losses) -> None:
        population = np.asarray(population, dtype=float)
        losses = np.asarray(losses, dtype=float)
        if population.shape != (self.population_size, self.dimension):
            raise ValueError("CMA population has an unexpected shape.")
        if losses.shape != (self.population_size,) or not np.isfinite(losses).all():
            raise ValueError("CMA losses must be one finite value per candidate.")
        order = np.argsort(losses, kind="stable")
        selected = population[order[: self.mu]]
        old_mean = self.state.mean.copy()
        new_mean = np.sum(self.weights[:, None] * selected, axis=0)
        y = (new_mean - old_mean) / self.state.sigma
        _, inverse_sqrt = self._eigensystem()
        self.state.path_sigma = (
            (1.0 - self.c_sigma) * self.state.path_sigma
            + math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff)
            * (inverse_sqrt @ y)
        )
        norm_path = np.linalg.norm(self.state.path_sigma)
        generation = self.state.generation + 1
        decay = math.sqrt(
            1.0 - (1.0 - self.c_sigma) ** (2.0 * generation)
        )
        h_sigma = float(
            norm_path / max(decay, 1e-12)
            < (1.4 + 2.0 / (self.dimension + 1.0)) * self.chi_n
        )
        self.state.path_c = (
            (1.0 - self.c_c) * self.state.path_c
            + h_sigma
            * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff)
            * y
        )
        steps = (selected - old_mean) / self.state.sigma
        rank_mu = sum(
            weight * np.outer(step, step)
            for weight, step in zip(self.weights, steps, strict=True)
        )
        covariance_scale = (
            1.0 - self.c1 - self.c_mu
            + self.c1 * (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        )
        self.state.covariance = (
            covariance_scale * self.state.covariance
            + self.c1 * np.outer(self.state.path_c, self.state.path_c)
            + self.c_mu * rank_mu
        )
        self.state.mean = np.clip(new_mean, 0.0, 1.0)
        self.state.sigma *= math.exp(
            (self.c_sigma / self.damping) * (norm_path / self.chi_n - 1.0)
        )
        self.state.generation = generation
        self.state.evaluations += self.population_size

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.state.mean,
            "sigma": np.asarray(self.state.sigma),
            "covariance": self.state.covariance,
            "path_c": self.state.path_c,
            "path_sigma": self.state.path_sigma,
            "generation": np.asarray(self.state.generation),
            "evaluations": np.asarray(self.state.evaluations),
        }

    @classmethod
    def from_arrays(
        cls,
        arrays,
        *,
        seed,
        population_size,
        parent_fraction=0.5,
        rng_state,
    ):
        state = CMAState(
            mean=np.asarray(arrays["mean"]),
            sigma=float(arrays["sigma"]),
            covariance=np.asarray(arrays["covariance"]),
            path_c=np.asarray(arrays["path_c"]),
            path_sigma=np.asarray(arrays["path_sigma"]),
            generation=int(arrays["generation"]),
            evaluations=int(arrays["evaluations"]),
        )
        return cls(
            state.mean,
            sigma=state.sigma,
            seed=seed,
            population_size=population_size,
            parent_fraction=parent_fraction,
            state=state,
            rng_state=rng_state,
        )

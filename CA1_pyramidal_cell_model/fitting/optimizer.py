"""Small dependency-light bounded CMA-ES implementation and run utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class CMAES:
    mean: np.ndarray
    sigma: float
    seed: int
    population_size: int
    generation: int = 0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).copy()
        self.n = self.mean.size
        self.rng = np.random.default_rng(self.seed)
        self.covariance = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self._population: np.ndarray | None = None
        self._old_mean: np.ndarray | None = None
        self._eigen()

    def _eigen(self) -> None:
        eigenvalues, eigenvectors = np.linalg.eigh((self.covariance + self.covariance.T) * 0.5)
        eigenvalues = np.maximum(eigenvalues, 1.0e-12)
        self._basis = eigenvectors
        self._scales = np.sqrt(eigenvalues)
        self._invsqrt = (eigenvectors / self._scales) @ eigenvectors.T

    def ask(self) -> np.ndarray:
        if self._population is not None:
            raise RuntimeError("tell() must be called before asking for another population")
        z = self.rng.normal(size=(self.population_size, self.n))
        y = z @ (self._basis * self._scales).T
        population = np.clip(self.mean[None, :] + self.sigma * y, 0.0, 1.0)
        self._population = population
        self._old_mean = self.mean.copy()
        return population

    def tell(self, losses: np.ndarray) -> None:
        if self._population is None or self._old_mean is None:
            raise RuntimeError("ask() must be called before tell()")
        losses = np.asarray(losses, dtype=float)
        if losses.shape != (self.population_size,) or not np.isfinite(losses).all():
            raise ValueError("CMA losses must be finite and match the population size")
        order = np.argsort(losses)
        mu = max(1, self.population_size // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        selected = self._population[order[:mu]]
        old_mean = self._old_mean
        self.mean = np.sum(weights[:, None] * selected, axis=0)
        y = (self.mean - old_mean) / max(self.sigma, 1.0e-12)
        c_sigma = (self.n + 2.0) / (self.n + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.n - 1.0) / (self.n + 1.0)) - 1.0) + c_sigma
        c_c = 4.0 / (self.n + 4.0)
        c1 = 2.0 / ((self.n + 1.4) ** 2 + mu)
        c_mu = min(1.0 - c1, 2.0 * (mu - 2.0 + 1.0 / mu) / ((self.n + 2.0) ** 2 + mu))
        mueff = 1.0 / np.sum(weights**2)
        self.ps = (1.0 - c_sigma) * self.ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * (self._invsqrt @ y)
        expected_norm = np.sqrt(self.n) * (1.0 - 1.0 / (4.0 * self.n) + 1.0 / (21.0 * self.n**2))
        h_sigma = float(np.linalg.norm(self.ps) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (self.generation + 1))) <
                        (1.4 + 2.0 / (self.n + 1.0)) * expected_norm)
        self.pc = (1.0 - c_c) * self.pc + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mueff) * y
        centered = (selected - old_mean[None, :]) / max(self.sigma, 1.0e-12)
        rank_mu = sum(weight * np.outer(vector, vector) for weight, vector in zip(weights, centered, strict=True))
        self.covariance = ((1.0 - c1 - c_mu) * self.covariance + c1 * np.outer(self.pc, self.pc) + c_mu * rank_mu)
        self.sigma *= float(np.exp((c_sigma / d_sigma) * (np.linalg.norm(self.ps) / expected_norm - 1.0)))
        self.sigma = float(np.clip(self.sigma, 0.01, 0.5))
        self.generation += 1
        self._population = None
        self._old_mean = None
        self._eigen()

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, covariance=self.covariance, pc=self.pc, ps=self.ps,
                 sigma=self.sigma, generation=self.generation, seed=self.seed,
                 population_size=self.population_size)

    @classmethod
    def load(cls, path: Path) -> "CMAES":
        with np.load(path) as state:
            result = cls(state["mean"], float(state["sigma"]), int(state["seed"]),
                         int(state["population_size"]), int(state["generation"]))
            result.covariance = state["covariance"]
            result.pc = state["pc"]
            result.ps = state["ps"]
            result._eigen()
            return result


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), encoding="utf-8")


def _append_evaluations(path: Path, generation: int, candidates: list[dict[str, float]],
                        losses: np.ndarray, evaluated: list | None) -> None:
    """Append one JSON line per candidate: parameters, loss and per-trace loss terms.

    Every evaluation is kept so the loss landscape can be analysed afterwards;
    spike lists are dropped to keep the file small.
    """
    with path.open("a", encoding="utf-8") as handle:
        for index, (values, loss) in enumerate(zip(candidates, losses, strict=True)):
            record = {"generation": generation, "index": index, "loss": float(loss),
                      "parameters": {key: float(value) for key, value in values.items()}}
            details = evaluated[index][1] if evaluated is not None and len(evaluated[index]) >= 2 else None
            if isinstance(details, dict) and "traces" in details:
                record["traces"] = [
                    {"trace": item.get("trace"), "kind": item.get("kind"),
                     "loss": item.get("loss"), "components": item.get("components", {})}
                    for item in details["traces"]
                ]
            elif isinstance(details, dict) and "error" in details:
                record["error"] = details["error"]
            handle.write(json.dumps(record) + "\n")


def run_optimization(config: dict, *, config_path: Path, output_dir: Path,
                     evaluator: Callable[[dict[str, float]], tuple[float, dict]],
                     simulator: Callable[[dict[str, float]], list[tuple[np.ndarray, np.ndarray]]],
                     plotter: Callable[[dict[str, float], Path], None] | None = None,
                     population_evaluator: Callable[[list[dict[str, float]]], list[tuple[float, dict]]] | None = None,
                     generation_plotter: Callable[[int, list[dict[str, float]], np.ndarray, list], None] | None = None,
                     resume: bool = True) -> dict:
    parameters = config["parameters"]
    keys = list(parameters)
    lower = np.asarray([float(parameters[key]["lower"]) for key in keys])
    upper = np.asarray([float(parameters[key]["upper"]) for key in keys])
    defaults = np.asarray([float(parameters[key]["default"]) for key in keys])
    if np.any(lower >= upper) or np.any((defaults < lower) | (defaults > upper)):
        raise ValueError("Every parameter requires lower < default < upper bounds")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "cma_state.npz"
    if resume and state_path.exists():
        optimizer = CMAES.load(state_path)
        if optimizer.n != len(keys):
            raise ValueError("Checkpoint parameter dimension does not match config")
        history = json.loads((output_dir / "history.json").read_text(encoding="utf-8")) if (output_dir / "history.json").exists() else []
    else:
        initial = np.clip((defaults - lower) / (upper - lower), 0.0, 1.0)
        section = config.get("optimizer", {})
        optimizer = CMAES(initial, float(section.get("sigma0", 0.2)), int(section.get("seed", 0)),
                          int(section.get("population_size", 16)))
        history = []
    generations = int(config.get("optimizer", {}).get("generations", 20))
    plot_every = int(config.get("optimizer", {}).get("plot_every", 1))
    best_loss = min((float(item["best_loss"]) for item in history), default=np.inf)
    best_values: dict[str, float] | None = None
    if (output_dir / "best_parameters.json").exists():
        best_values = json.loads((output_dir / "best_parameters.json").read_text(encoding="utf-8"))["parameters"]
    for _ in range(optimizer.generation, generations):
        population = optimizer.ask()
        candidates = []
        for normalized in population:
            physical = lower + normalized * (upper - lower)
            values = dict(zip(keys, physical, strict=True))
            candidates.append(values)
        failures = []
        if population_evaluator is not None:
            evaluated = population_evaluator(candidates)
            if len(evaluated) != len(candidates):
                raise RuntimeError("Parallel evaluator returned the wrong number of candidate results")
            losses = [float(result[0]) for result in evaluated]
        else:
            losses = []
            for values in candidates:
                try:
                    loss, _ = evaluator(values)
                except Exception as exc:
                    loss = 1.0e12
                    failures.append(f"{type(exc).__name__}: {exc}")
                losses.append(float(loss))
        losses_array = np.asarray(losses)
        if len(failures) == len(candidates):
            raise RuntimeError("Every candidate failed in one generation; last error: " + failures[-1])
        winner = int(np.argmin(losses_array))
        optimizer.tell(losses_array)
        generation_best = float(losses_array[winner])
        if generation_plotter is not None and (optimizer.generation % max(1, plot_every) == 0):
            if population_evaluator is None or not all(len(result) >= 3 for result in evaluated):
                raise RuntimeError("Generation plotting requires simulations from the population evaluator")
            generation_plotter(optimizer.generation, candidates, losses_array, evaluated)
        if generation_best < best_loss:
            best_loss = generation_best
            best_values = candidates[winner]
            _write_json(output_dir / "best_parameters.json", {"loss": best_loss, "parameters": best_values, "keys": keys})
            if population_evaluator is not None and len(evaluated[winner]) >= 2:
                _write_json(output_dir / "best_objective.json", evaluated[winner][1])
            simulations = simulator(best_values)
            np.savez(output_dir / "best_simulations.npz", **{f"t_{i}": item[0] for i, item in enumerate(simulations)},
                     **{f"v_{i}": item[1] for i, item in enumerate(simulations)})
            if plotter is not None:
                plotter(best_values, output_dir / "best_fit.png")
        entry = {"generation": optimizer.generation, "best_loss": generation_best,
                 "population_mean_loss": float(np.mean(losses_array)), "global_best_loss": best_loss,
                 "failed_candidates": len(failures)}
        _append_evaluations(output_dir / "evaluations.jsonl", optimizer.generation, candidates,
                            losses_array, evaluated if population_evaluator is not None else None)
        history.append(entry)
        _write_json(output_dir / "history.json", history)
        optimizer.save(state_path)
        _write_json(output_dir / "run_metadata.json", {"config": str(config_path), "parameters": keys,
                                                        "completed_generations": optimizer.generation,
                                                        "population_size": optimizer.population_size,
                                                        "parallel_workers": int(config.get("optimizer", {}).get("workers", 1))
                                                        if population_evaluator is not None else 1})
    if best_values is None:
        raise RuntimeError("Optimization completed without a finite candidate")
    return {"loss": best_loss, "parameters": best_values, "history": history}

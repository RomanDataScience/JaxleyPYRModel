"""Parallel two-stage orchestration and basin generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cma import CMAES
from .config import RunConfig
from .data import Trace, load_protocol_traces
from .objective import depolarizing_objective, hyperpolarizing_objective
from .parameters import ParameterSpace
from .plotting import plot_generation
from .simulator import NeuronSimulator


_WORKER: dict[str, Any] = {}


def _init_worker(stage: str, traces: list[Trace], d_lambda: float,
                 objective_options: dict[str, Any]) -> None:
    _WORKER["stage"] = stage
    _WORKER["traces"] = traces
    _WORKER["simulator"] = NeuronSimulator(d_lambda=d_lambda, quiet=True)
    _WORKER["objective_options"] = objective_options


def _evaluate_worker(normalized: np.ndarray, keys: tuple[str, ...]) -> tuple[float, dict]:
    physical = _WORKER["space"].physical(normalized) if "space" in _WORKER else None
    if physical is None:
        raise RuntimeError("Worker parameter space was not initialized")
    mapping = dict(zip(keys, physical, strict=True))
    try:
        simulations = _WORKER["simulator"].simulate_many(_WORKER["traces"], mapping)
        if _WORKER["stage"] == "hyper":
            result = hyperpolarizing_objective(_WORKER["traces"], simulations, **_WORKER["objective_options"])
        else:
            result = depolarizing_objective(_WORKER["traces"], simulations, **_WORKER["objective_options"])
        return result.value, result.details
    except Exception as exc:
        return 1.0e12, {"error": type(exc).__name__, "message": str(exc)}


def _simulate_worker(normalized: np.ndarray, keys: tuple[str, ...]):
    physical = _WORKER["space"].physical(normalized)
    mapping = dict(zip(keys, physical, strict=True))
    try:
        simulations = _WORKER["simulator"].simulate_many(_WORKER["traces"], mapping)
        return [{"time_ms": simulation.time_ms, "voltage_mV": simulation.voltage_mV}
                for simulation in simulations]
    except Exception:
        return None


def _init_worker_with_space(stage: str, traces: list[Trace], d_lambda: float,
                            objective_options: dict[str, Any], space: ParameterSpace) -> None:
    _init_worker(stage, traces, d_lambda, objective_options)
    _WORKER["space"] = space


def _evaluate_serial(normalized: np.ndarray, *, stage: str, traces: list[Trace],
                     space: ParameterSpace, d_lambda: float,
                     objective_options: dict[str, Any]) -> tuple[float, dict]:
    try:
        simulator = NeuronSimulator(d_lambda=d_lambda, quiet=True)
        simulations = simulator.simulate_many(traces, space.mapping(normalized))
        if stage == "hyper":
            result = hyperpolarizing_objective(traces, simulations, **objective_options)
        else:
            result = depolarizing_objective(traces, simulations, **objective_options)
        return result.value, result.details
    except Exception as exc:
        return 1.0e12, {"error": type(exc).__name__, "message": str(exc)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _objective_options(raw: dict, stage: str) -> dict[str, Any]:
    section = dict(raw["stage1"] if stage == "hyper" else raw["stage2"])
    if stage == "hyper":
        return {"sigma_mV": float(section.get("sigma_hyper_mV", 1.0)),
                "region_weights": section.get("region_weights", {"pre": 1.0, "step": 4.0, "recovery": 3.0}),
                "deflection_weight": float(section.get("deflection_weight", 2.0)),
                "sigma_deflection_mV": float(section.get("sigma_deflection_mV", 1.0)),
                "threshold_mV": float(section.get("threshold_mV", -20.0)),
                "refractory_ms": float(section.get("refractory_ms", 2.0)),
                "prominence_mV": float(section.get("prominence_mV", 5.0)),
                "spike_penalty": float(section.get("hyper_spike_penalty", 1.0e4))}
    return {"sigma_trajectory_mV": float(section.get("sigma_trajectory_mV", 2.0)),
            "sigma_spike_mV": float(section.get("sigma_spike_mV", 2.0)),
            "sigma_plateau_mV": float(section.get("sigma_plateau_mV", 2.0)),
            "sigma_recovery_mV": float(section.get("sigma_recovery_mV", 2.0)),
            "sigma_terminal_mV": float(section.get("sigma_terminal_mV", 2.0)),
            "weights": section.get("weights", {"trajectory": 1.0, "spike_shape": 3.0, "plateau": 2.0, "return_baseline": 3.0}),
            "threshold_mV": float(section.get("threshold_mV", -20.0)),
            "refractory_ms": float(section.get("refractory_ms", 2.0)),
            "prominence_mV": float(section.get("prominence_mV", 5.0)),
            "spike_window_ms": float(section.get("spike_window_ms", 5.0)),
            "plateau_exclusion_ms": float(section.get("plateau_exclusion_ms", 3.0)),
            "unmatched_spike_penalty": float(section.get("unmatched_spike_penalty", 100.0)),
            "invalid_plateau_penalty": float(section.get("invalid_plateau_penalty", 100.0)),
            "extra_spike_penalty": float(section.get("depolarizing_extra_spike_penalty", 1.0e4)),
            "return_alpha": float(section.get("return_alpha", 0.7))}


def _load_traces(config: RunConfig, stage: str) -> list[Trace]:
    if stage == "hyper":
        return load_protocol_traces(config.data_root, cell=config.cell,
                                    protocol="hyperpolarizing_pulse",
                                    trace_names=config.trace_names, pre_ms=800.0,
                                    full_trial=True, center_current=False)
    return load_protocol_traces(config.data_root, cell=config.cell,
                                protocol="depolarizing_step",
                                trace_names=config.trace_names, pre_ms=0.0,
                                full_trial=True)


def _evaluate_population(population: np.ndarray, *, stage: str, traces: list[Trace],
                         space: ParameterSpace, config: RunConfig,
                         options: dict[str, Any]) -> tuple[np.ndarray, list[dict]]:
    if config.workers == 1:
        pairs = [_evaluate_serial(candidate, stage=stage, traces=traces, space=space,
                                  d_lambda=config.d_lambda, objective_options=options)
                 for candidate in population]
    else:
        with ProcessPoolExecutor(max_workers=config.workers,
                                 initializer=_init_worker_with_space,
                                 initargs=(stage, traces, config.d_lambda, options, space)) as pool:
            pairs = list(pool.map(_evaluate_worker, population, [space.keys] * len(population)))
    return np.asarray([pair[0] for pair in pairs], dtype=float), [pair[1] for pair in pairs]


def _simulate_for_plots(population: np.ndarray, *, stage: str, traces: list[Trace],
                        space: ParameterSpace, config: RunConfig,
                        options: dict[str, Any]) -> list[list]:
    if config.workers == 1:
        simulator = NeuronSimulator(d_lambda=config.d_lambda, quiet=True)
        result = []
        for candidate in population:
            try:
                simulations = simulator.simulate_many(traces, space.mapping(candidate))
                result.append([{"time_ms": simulation.time_ms, "voltage_mV": simulation.voltage_mV}
                               for simulation in simulations])
            except Exception:
                result.append(None)
        return result
    with ProcessPoolExecutor(max_workers=config.workers,
                             initializer=_init_worker_with_space,
                             initargs=(stage, traces, config.d_lambda, options, space)) as pool:
        return list(pool.map(_simulate_worker, population, [space.keys] * len(population)))


def _study_manifest(config: RunConfig, stage: str, seed: int, space: ParameterSpace,
                    run_dir: Path, basin_id: str | None = None) -> None:
    section = config.section("stage1" if stage == "hyper" else "stage2")
    _write_json(run_dir / "manifest.json", {
        "stage": stage, "seed": seed, "basin_id": basin_id,
        "config_hash": config.hash(), "parameter_keys": list(space.keys),
        "generations": int(section["generations"]),
        "population_size": int(section["population_size"]),
        "parallel_workers": config.workers,
    })


def run_study(config: RunConfig, *, stage: str, seed: int, run_dir: Path,
              initial_mean: np.ndarray | None = None,
              basin_id: str | None = None) -> dict[str, Any]:
    space = config.parameters
    section = config.section("stage1" if stage == "hyper" else "stage2")
    traces = _load_traces(config, stage)
    run_dir.mkdir(parents=True, exist_ok=True)
    _study_manifest(config, stage, seed, space, run_dir, basin_id)
    compatibility = f"{config.hash()}:{stage}:{seed}:{basin_id}:{space.keys}"
    checkpoint_dir = run_dir / "checkpoint"
    optimizer = CMAES.load(checkpoint_dir, seed=seed, compatibility_hash=compatibility)
    if optimizer is None:
        mean = initial_mean if initial_mean is not None else space.normalize(space.reference)
        optimizer = CMAES(mean, sigma=float(section.get("sigma0", 0.15)), seed=seed,
                          population_size=int(section.get("population_size", 30)),
                          parent_fraction=float(section.get("parent_fraction", 0.5)))
    options = _objective_options(config.raw, stage)
    generations = int(section.get("generations", 200))
    best: tuple[float, np.ndarray, dict] | None = None
    history_path = run_dir / "generations.jsonl"
    while optimizer.state.generation < generations:
        population = optimizer.ask()
        losses, details = _evaluate_population(population, stage=stage, traces=traces,
                                               space=space, config=config, options=options)
        best_index = int(np.argmin(losses))
        if best is None or losses[best_index] < best[0]:
            best = (float(losses[best_index]), population[best_index].copy(), details[best_index])
        generation = optimizer.state.generation + 1
        np.savez_compressed(run_dir / f"population_generation_{generation:04d}.npz",
                            population=population, losses=losses)
        plotting = dict(config.raw.get("plotting", {}))
        if bool(plotting.get("enabled", True)):
            top_k = int(plotting.get("top_k", 10))
            top_indices = np.argsort(losses, kind="stable")[:min(top_k, len(losses))]
            captured = _simulate_for_plots(population[top_indices], stage=stage, traces=traces,
                                           space=space, config=config, options=options)
            from .objective import SimulationOutput
            converted = []
            for item in captured:
                converted.append(None if item is None else
                                 [SimulationOutput(np.asarray(entry["time_ms"]), np.asarray(entry["voltage_mV"]))
                                  for entry in item])
            # Plot only the selected candidates, but retain their original
            # population indices/loss ordering in the metadata.
            plot_population = population[top_indices]
            plot_losses = losses[top_indices]
            plot_simulations = [item for item in converted]
            if all(item is not None for item in plot_simulations):
                plot_generation(output_dir=run_dir / "plots", generation=generation,
                                stage=stage, traces=traces, population=plot_population,
                                losses=plot_losses, simulations=plot_simulations,
                                space=space, top_k=top_k,
                                dpi=int(plotting.get("dpi", 120)))
        optimizer.tell(population, losses)
        optimizer.save(checkpoint_dir, compatibility)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"generation": generation, "best": float(losses[best_index]),
                                     "mean": float(np.mean(losses)), "sigma": optimizer.state.sigma}) + "\n")
    if best is None:
        raise RuntimeError("Study produced no generations")
    result = {"loss": best[0], "normalized": best[1].tolist(),
              "physical": space.physical(best[1]).tolist(), "details": best[2],
              "generation": optimizer.state.generation}
    _write_json(run_dir / "best_parameters.json", {"keys": list(space.keys), **result})
    return result


def make_basins(config: RunConfig, stage1_run: Path, *, output: Path | None = None) -> list[dict]:
    space = config.parameters
    records: list[dict] = []
    candidates: list[dict] = []
    for seed in config.section("stage1").get("seeds", list(range(10))):
        seed_dir = stage1_run / f"seed_{int(seed):03d}"
        generation = int(config.section("stage1")["generations"])
        population_path = seed_dir / f"population_generation_{generation:04d}.npz"
        if not population_path.exists():
            raise FileNotFoundError(population_path)
        with np.load(population_path, allow_pickle=False) as values:
            population, losses = values["population"], values["losses"]
        for rank in np.argsort(losses, kind="stable"):
            candidates.append({"source_seed": int(seed), "source_generation": generation,
                               "rank": int(rank), "loss": float(losses[rank]),
                               "normalized": population[rank].tolist(),
                               "physical": space.physical(population[rank]).tolist()})
    selected: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        if candidate["rank"] >= 10:
            continue
        key = tuple(np.round(candidate["normalized"], 14))
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
    for candidate in candidates:
        if len(selected) >= 100:
            break
        key = tuple(np.round(candidate["normalized"], 14))
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
    if len(selected) < 100:
        raise RuntimeError(f"Expected 100 distinct basins, found {len(selected)}")
    for index, candidate in enumerate(selected[:100]):
        records.append({"basin_id": f"b{index:03d}", **candidate})
    output = output or stage1_run / "basins.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


def load_basins(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_pipeline(config: RunConfig, output_root: Path) -> None:
    stage1_dir = output_root / "stage1_hyper"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    for seed in config.section("stage1").get("seeds", list(range(10))):
        run_study(config, stage="hyper", seed=int(seed), run_dir=stage1_dir / f"seed_{int(seed):03d}")
    basins = make_basins(config, stage1_dir)
    stage2_dir = output_root / "stage2_depolarizing"
    for basin in basins:
        mean = np.asarray(basin["normalized"], dtype=float)
        for seed in config.section("stage2").get("seeds", list(range(10))):
            seed = int(seed)
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(basin["basin_id"][1:])]))
            initial = np.clip(mean + rng.uniform(-0.15, 0.15, size=mean.size), 0.0, 1.0)
            run_study(config, stage="depolarizing", seed=seed,
                      run_dir=stage2_dir / basin["basin_id"] / f"seed_{seed:03d}",
                      initial_mean=initial, basin_id=basin["basin_id"])

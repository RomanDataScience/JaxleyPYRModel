"""Parallel two-stage orchestration and basin generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .cma import CMAES
from .config import RunConfig
from .data import Trace, crop_trace, load_protocol_traces
from .objective import (
    ObjectiveContext,
    build_objective_context,
    depolarizing_objective,
    hyperpolarizing_objective,
)
from .parameters import DEFAULTS, PASSIVE, ParameterSpace, make_parameter_space, passive_model_values
from .plotting import plot_generation
from .simulator import make_simulator


LOGGER = logging.getLogger(__name__)
_WORKER: dict[str, Any] = {}


def _init_worker(stage: str, simulation_traces: list[Trace], fitness_traces: list[Trace],
                 d_lambda: float,
                 objective_options: dict[str, Any], backend: str) -> None:
    _WORKER["stage"] = stage
    _WORKER["simulation_traces"] = simulation_traces
    _WORKER["fitness_traces"] = fitness_traces
    _WORKER["simulator"] = make_simulator(backend, d_lambda=d_lambda, quiet=True)
    _WORKER["objective_options"] = objective_options
    _WORKER["objective_context"] = _build_objective_context(
        fitness_traces, stage, objective_options
    )


def _simulation_payload(simulations) -> list[dict[str, np.ndarray]]:
    """Convert simulator outputs to a small, pickle/NPZ-friendly payload."""
    return [{"time_ms": np.asarray(simulation.time_ms, dtype=float),
             "voltage_mV": np.asarray(simulation.voltage_mV, dtype=float)}
            for simulation in simulations]


def _payload_to_simulations(payload: list[dict[str, np.ndarray]] | None):
    if payload is None:
        return None
    from .objective import SimulationOutput
    return [SimulationOutput(np.asarray(item["time_ms"], dtype=float),
                             np.asarray(item["voltage_mV"], dtype=float))
            for item in payload]


def _build_objective_context(traces: list[Trace], stage: str,
                             options: dict[str, Any]) -> ObjectiveContext:
    return build_objective_context(
        traces,
        stage="depolarizing" if stage == "depolarizing" else "hyper",
        threshold_mV=float(options.get("threshold_mV", -20.0)),
        refractory_ms=float(options.get("refractory_ms", 2.0)),
        prominence_mV=float(options.get("prominence_mV", 5.0)),
    )


def _evaluate_worker(normalized: np.ndarray, keys: tuple[str, ...]):
    physical = _WORKER["space"].physical(normalized) if "space" in _WORKER else None
    if physical is None:
        raise RuntimeError("Worker parameter space was not initialized")
    mapping = dict(zip(keys, physical, strict=True))
    if _WORKER["stage"] == "passive":
        mapping = passive_model_values(mapping)
    try:
        simulations = _WORKER["simulator"].simulate_many(
            _WORKER["simulation_traces"], mapping
        )
        if _WORKER["stage"] in {"passive", "hyper"}:
            result = hyperpolarizing_objective(
                _WORKER["fitness_traces"], simulations,
                context=_WORKER["objective_context"], include_details=False,
                **_WORKER["objective_options"],
            )
        else:
            result = depolarizing_objective(
                _WORKER["fitness_traces"], simulations,
                context=_WORKER["objective_context"], include_details=False,
                **_WORKER["objective_options"],
            )
        return result.value, _simulation_payload(simulations), None
    except Exception as exc:
        return 1.0e12, None, {"error": type(exc).__name__, "message": str(exc)}


def _init_worker_with_space(stage: str, simulation_traces: list[Trace],
                            fitness_traces: list[Trace], d_lambda: float,
                            objective_options: dict[str, Any], backend: str,
                            space: ParameterSpace) -> None:
    _init_worker(stage, simulation_traces, fitness_traces, d_lambda,
                 objective_options, backend)
    _WORKER["space"] = space


def _evaluate_serial(normalized: np.ndarray, *, stage: str,
                     simulation_traces: list[Trace], fitness_traces: list[Trace],
                     space: ParameterSpace, d_lambda: float,
                     objective_options: dict[str, Any], backend: str,
                     simulator=None, objective_context: ObjectiveContext | None = None):
    try:
        simulator = simulator or make_simulator(backend, d_lambda=d_lambda, quiet=True)
        mapping = space.mapping(normalized)
        if stage == "passive":
            mapping = passive_model_values(mapping)
        simulations = simulator.simulate_many(simulation_traces, mapping)
        if stage in {"passive", "hyper"}:
            result = hyperpolarizing_objective(
                fitness_traces, simulations, context=objective_context,
                include_details=False,
                **objective_options,
            )
        else:
            result = depolarizing_objective(
                fitness_traces, simulations, context=objective_context,
                **objective_options,
            )
        return result.value, _simulation_payload(simulations), None
    except Exception as exc:
        return 1.0e12, None, {"error": type(exc).__name__, "message": str(exc)}


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
    section = dict(raw["stage1"] if stage in {"passive", "hyper"} else raw["stage2"])
    if stage in {"passive", "hyper"}:
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


def _config_with_workers(config: RunConfig, workers: int) -> RunConfig:
    raw = copy.deepcopy(config.raw)
    raw.setdefault("runtime", {})["parallel_workers"] = int(workers)
    return RunConfig(raw, config.path)


def _run_study_task(task):
    config, stage, seed, run_dir, initial_mean, basin_id = task
    return run_study(config, stage=stage, seed=seed, run_dir=run_dir,
                     initial_mean=initial_mean, basin_id=basin_id)


def _run_studies(config: RunConfig, tasks: list[tuple]) -> list[dict[str, Any]]:
    """Run independent studies without oversubscribing candidate workers."""
    if config.study_workers == 1:
        return [_run_study_task(task) for task in tasks]
    inner_config = _config_with_workers(config, 1)
    inner_tasks = [(
        inner_config, stage, seed, run_dir, initial_mean, basin_id
    ) for _, stage, seed, run_dir, initial_mean, basin_id in tasks]
    with ProcessPoolExecutor(max_workers=config.study_workers) as pool:
        return list(pool.map(_run_study_task, inner_tasks))


def _load_traces(config: RunConfig, stage: str) -> list[Trace]:
    if stage in {"passive", "hyper"}:
        return load_protocol_traces(config.data_root, cell=config.cell,
                                    protocol="hyperpolarizing_pulse",
                                    trace_names=config.trace_names,
                                    pre_ms=config.simulation_pre_ms,
                                    post_ms=config.simulation_post_ms,
                                    full_trial=True, center_current=False)
    return load_protocol_traces(config.data_root, cell=config.cell,
                                protocol="depolarizing_step",
                                trace_names=config.trace_names,
                                pre_ms=config.simulation_pre_ms,
                                post_ms=config.simulation_post_ms,
                                full_trial=True)


def _fitness_traces(config: RunConfig, stage: str,
                    simulation_traces: list[Trace]) -> list[Trace]:
    """Build objective traces while retaining the simulator time origin."""
    if stage not in {"passive", "hyper"}:
        return simulation_traces
    pre_ms = config.hyperpolarizing_fitness_pre_ms
    return [
        crop_trace(trace, start_ms=trace.epoch_start_ms - pre_ms)
        for trace in simulation_traces
    ]


def validate_current_replay(config: RunConfig, output_path: Path | None = None) -> dict[str, Any]:
    """Run one passive-only trace and record the raw-current replay check."""
    validation = config.raw.get("validation", {})
    trace_name = str(validation.get("current_trace", config.trace_names[0]))
    traces = load_protocol_traces(
        config.data_root,
        cell=config.cell,
        protocol="hyperpolarizing_pulse",
        trace_names=(trace_name,),
        pre_ms=config.simulation_pre_ms,
        post_ms=config.simulation_post_ms,
        full_trial=True,
        center_current=False,
    )
    trace = traces[0]
    passive_space = make_parameter_space(include=PASSIVE)
    reference = dict(zip(PASSIVE, passive_space.reference.tolist(), strict=True))
    values = passive_model_values(reference)
    simulation = make_simulator(config.backend, d_lambda=config.d_lambda, quiet=True).simulate_many(
        traces, values
    )[0]
    simulated = np.interp(trace.time_ms, simulation.time_ms, simulation.voltage_mV)
    pre = trace.time_ms <= trace.epoch_start_ms
    step = ((trace.time_ms >= trace.epoch_start_ms) &
            (trace.time_ms <= trace.epoch_stop_ms))
    current_pre = float(np.median(trace.current_nA[pre]))
    current_step = float(np.median(trace.current_nA[step]))
    current_delta = current_step - current_pre
    voltage_pre = float(np.median(simulated[pre]))
    voltage_step = float(np.median(simulated[step]))
    voltage_min_delta = float(np.min(simulated[step]) - voltage_pre)
    report = {
        "trace": trace.trace,
        "protocol": trace.protocol,
        "current_units": "nA",
        "current_replay": {
            "before_pulse_median_nA": current_pre,
            "during_pulse_median_nA": current_step,
            "step_delta_nA": current_delta,
        },
        "passive_reference": reference,
        "voltage_response": {
            "simulated_pre_median_mV": voltage_pre,
            "simulated_step_median_mV": voltage_step,
            "simulated_step_min_deflection_mV": voltage_min_delta,
        },
        "passed": False,
    }

    expected_pre = validation.get("expected_before_pulse_nA")
    expected_step = validation.get("expected_during_pulse_nA")
    tolerance = float(validation.get("current_tolerance_nA", 1.0e-6))
    if expected_pre is not None and abs(current_pre - float(expected_pre)) > tolerance:
        if output_path is not None:
            _write_json(output_path, report)
        raise RuntimeError(f"Current replay pre-pulse mismatch: {current_pre} nA")
    if expected_step is not None and abs(current_step - float(expected_step)) > tolerance:
        if output_path is not None:
            _write_json(output_path, report)
        raise RuntimeError(f"Current replay step mismatch: {current_step} nA")
    if current_delta < 0.0 and voltage_min_delta >= 0.0:
        if output_path is not None:
            _write_json(output_path, report)
        raise RuntimeError(
            "Raw-current replay did not produce a negative passive deflection "
            f"for a negative current step ({voltage_min_delta:.6g} mV)."
        )
    report["passed"] = True
    if output_path is not None:
        _write_json(output_path, report)
    return report


def _full_initial_from_passive(config: RunConfig, passive_values: dict[str, float],
                               seed: int, variation: float = 0.15) -> np.ndarray:
    """Embed a passive fit in the full space and perturb passive coordinates."""
    space = config.parameters
    physical = dict(DEFAULTS)
    physical.update(passive_values)
    mean = space.normalize([physical[key] for key in space.keys])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x50415353]))
    for index, key in enumerate(space.keys):
        if key in PASSIVE:
            mean[index] = np.clip(mean[index] + rng.uniform(-variation, variation), 0.0, 1.0)
    return mean


def run_passive_precalibration(config: RunConfig, output_root: Path) -> dict[str, float]:
    """Validate raw replay, then fit passive properties before full fitting."""
    passive_root = output_root / "stage0_passive"
    passive_root.mkdir(parents=True, exist_ok=True)
    replay_path = passive_root / "current_replay.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    else:
        replay = None
    if not replay or not replay.get("passed", False):
        replay = validate_current_replay(config, replay_path)
    _write_json(passive_root / "current_replay.json", replay)

    best_path = passive_root / "passive_best.json"
    if best_path.exists():
        return dict(json.loads(best_path.read_text(encoding="utf-8"))["physical_by_name"])

    results = []
    for seed in config.section("passive").get("seeds", [0]):
        seed = int(seed)
        result = run_study(
            config,
            stage="passive",
            seed=seed,
            run_dir=passive_root / f"seed_{seed:03d}",
        )
        results.append((float(result["loss"]), seed, result))
    loss, seed, result = min(results, key=lambda item: item[0])
    best = {
        "stage": "passive",
        "best_seed": seed,
        "loss": loss,
        "physical_by_name": result["physical_by_name"],
    }
    _write_json(best_path, best)
    return dict(result["physical_by_name"])


def _evaluate_population(population: np.ndarray, *, stage: str,
                         simulation_traces: list[Trace], fitness_traces: list[Trace],
                         space: ParameterSpace, config: RunConfig,
                         options: dict[str, Any], executor=None,
                         simulator=None,
                         objective_context: ObjectiveContext | None = None) -> tuple[np.ndarray, list | None, list]:
    if config.workers == 1:
        pairs = [_evaluate_serial(
                                  candidate, stage=stage,
                                  simulation_traces=simulation_traces,
                                  fitness_traces=fitness_traces, space=space,
                                  d_lambda=config.d_lambda, objective_options=options,
                                  backend=config.backend, simulator=simulator,
                                  objective_context=objective_context)
                 for candidate in population]
    else:
        if executor is None:
            with ProcessPoolExecutor(
                    max_workers=config.workers,
                    initializer=_init_worker_with_space,
                    initargs=(stage, simulation_traces, fitness_traces,
                              config.d_lambda, options, config.backend, space),
            ) as pool:
                pairs = list(pool.map(_evaluate_worker, population, [space.keys] * len(population)))
        else:
            pairs = list(executor.map(_evaluate_worker, population, [space.keys] * len(population)))
    return (np.asarray([pair[0] for pair in pairs], dtype=float),
            [pair[1] for pair in pairs],
            [pair[2] for pair in pairs])


def _objective_details(stage: str, fitness_traces: list[Trace], payload,
                       options: dict[str, Any], context: ObjectiveContext,
                       error: dict | None) -> dict:
    if payload is None:
        return error or {}
    simulations = _payload_to_simulations(payload)
    if stage in {"passive", "hyper"}:
        result = hyperpolarizing_objective(
            fitness_traces, simulations, context=context, include_details=True, **options
        )
    else:
        result = depolarizing_objective(
            fitness_traces, simulations, context=context, include_details=True, **options
        )
    return result.details


def _save_generation_simulations(run_dir: Path, generation: int,
                                 population: np.ndarray, losses: np.ndarray,
                                 simulations: list | None) -> None:
    """Persist candidate parameters and simulator outputs for later plotting.

    Trace lengths need not be identical, so each candidate/trace pair is stored
    under its own NPZ key.  Failed candidates are represented in ``valid`` and
    do not prevent the rest of the generation from being cached.
    """
    if simulations is None:
        return
    payload: dict[str, np.ndarray] = {
        "population": np.asarray(population, dtype=float),
        "losses": np.asarray(losses, dtype=float),
        "valid": np.asarray([item is not None for item in simulations], dtype=bool),
    }
    for candidate_index, candidate in enumerate(simulations):
        if candidate is None:
            continue
        for trace_index, simulation in enumerate(candidate):
            payload[f"candidate_{candidate_index:04d}_trace_{trace_index:04d}_time_ms"] = simulation["time_ms"]
            payload[f"candidate_{candidate_index:04d}_trace_{trace_index:04d}_voltage_mV"] = simulation["voltage_mV"]
    np.savez_compressed(run_dir / f"simulations_generation_{generation:04d}.npz", **payload)


def _study_manifest(config: RunConfig, stage: str, seed: int, space: ParameterSpace,
                    run_dir: Path, basin_id: str | None = None,
                    initial_mean: np.ndarray | None = None) -> None:
    section_name = "passive" if stage == "passive" else "stage1" if stage == "hyper" else "stage2"
    section = config.section(section_name)
    payload = {
        "stage": stage, "model_mode": "passive_only" if stage == "passive" else "full",
        "seed": seed, "basin_id": basin_id,
        "config_hash": config.hash(), "parameter_keys": list(space.keys),
        "generations": int(section["generations"]),
        "population_size": int(section["population_size"]),
        "backend": config.backend,
        "parallel_workers": config.workers,
    }
    if initial_mean is not None:
        payload["initial_mean_normalized"] = np.asarray(initial_mean, dtype=float).tolist()
    _write_json(run_dir / "manifest.json", payload)


def run_study(config: RunConfig, *, stage: str, seed: int, run_dir: Path,
              initial_mean: np.ndarray | None = None,
              basin_id: str | None = None) -> dict[str, Any]:
    space = make_parameter_space(include=PASSIVE) if stage == "passive" else config.parameters
    section_name = "passive" if stage == "passive" else "stage1" if stage == "hyper" else "stage2"
    section = config.section(section_name)
    simulation_traces = _load_traces(config, stage)
    fitness_traces = _fitness_traces(config, stage, simulation_traces)
    run_dir.mkdir(parents=True, exist_ok=True)
    _study_manifest(config, stage, seed, space, run_dir, basin_id, initial_mean)
    compatibility = (f"{config.hash()}:{stage}:{seed}:{basin_id}:{space.keys}:"
                     "fitness-window-v2")
    checkpoint_dir = run_dir / "checkpoint"
    optimizer = CMAES.load(checkpoint_dir, seed=seed, compatibility_hash=compatibility)
    resumed = optimizer is not None
    if optimizer is None:
        mean = initial_mean if initial_mean is not None else space.normalize(space.reference)
        optimizer = CMAES(mean, sigma=float(section.get("sigma0", 0.15)), seed=seed,
                          population_size=int(section.get("population_size", 30)),
                          parent_fraction=float(section.get("parent_fraction", 0.5)))
    options = _objective_options(config.raw, stage)
    objective_context = _build_objective_context(fitness_traces, stage, options)
    generations = int(section.get("generations", 200))
    checkpoint_every = max(1, int(section.get("checkpoint_every", 1)))
    completed_best_path = run_dir / "best_parameters.json"
    if resumed and optimizer.state.generation >= generations and completed_best_path.exists():
        return json.loads(completed_best_path.read_text(encoding="utf-8"))
    best_state_path = run_dir / "best_so_far.json"
    best: tuple[float, np.ndarray, dict] | None = None
    if resumed and best_state_path.exists():
        saved_best = json.loads(best_state_path.read_text(encoding="utf-8"))
        best = (float(saved_best["loss"]),
                np.asarray(saved_best["normalized"], dtype=float),
                dict(saved_best.get("details", {})))
    history_path = run_dir / "generations.jsonl"
    # Keep one worker pool for the complete study.  More importantly, the
    # plotting path consumes the outputs returned by the evaluation below;
    # it never starts a second simulation pool for the same candidates.
    executor = None
    serial_simulator = None
    if config.workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_init_worker_with_space,
            initargs=(stage, simulation_traces, fitness_traces, config.d_lambda,
                      options, config.backend, space),
        )
    else:
        serial_simulator = make_simulator(config.backend, d_lambda=config.d_lambda, quiet=True)
    try:
        while optimizer.state.generation < generations:
            population = optimizer.ask()
            losses, simulations, errors = _evaluate_population(
                population, stage=stage, simulation_traces=simulation_traces,
                fitness_traces=fitness_traces, space=space, config=config,
                options=options, executor=executor, simulator=serial_simulator,
                objective_context=objective_context,
            )
            best_index = int(np.argmin(losses))
            if best is None or losses[best_index] < best[0]:
                details = _objective_details(
                    stage, fitness_traces, simulations[best_index], options,
                    objective_context, errors[best_index],
                )
                best = (float(losses[best_index]), population[best_index].copy(), details)
                _write_json(best_state_path, {
                    "loss": best[0], "normalized": best[1], "details": best[2],
                    "generation": optimizer.state.generation + 1,
                })
            generation = optimizer.state.generation + 1
            np.savez_compressed(run_dir / f"population_generation_{generation:04d}.npz",
                                population=population, losses=losses)
            _save_generation_simulations(run_dir, generation, population, losses, simulations)
            # Commit the optimizer state before rendering.  Figures are
            # diagnostics and must not make an expensive evaluated generation
            # need to be rerun if a plotting backend fails.
            optimizer.tell(population, losses)
            if generation % checkpoint_every == 0 or generation == generations:
                optimizer.save(checkpoint_dir, compatibility)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"generation": generation, "best": float(losses[best_index]),
                                         "mean": float(np.mean(losses)), "sigma": optimizer.state.sigma}) + "\n")
            plotting = dict(config.raw.get("plotting", {}))
            plot_every = max(1, int(plotting.get("every", 1)))
            if (bool(plotting.get("enabled", True)) and
                    (generation % plot_every == 0 or generation == generations)):
                try:
                    top_k = int(plotting.get("top_k", 10))
                    top_indices = np.argsort(losses, kind="stable")[:min(top_k, len(losses))]
                    converted = [_payload_to_simulations(simulations[index]) for index in top_indices]
                    # Plot only the selected candidates, but retain their original
                    # population indices/loss ordering in the metadata.
                    plot_population = population[top_indices]
                    plot_losses = losses[top_indices]
                    if all(item is not None for item in converted):
                        plot_generation(output_dir=run_dir / "plots", generation=generation,
                                        stage=stage, traces=fitness_traces,
                                        population=plot_population,
                                        losses=plot_losses, simulations=converted,
                                        space=space, top_k=top_k,
                                        population_indices=top_indices,
                                        dpi=int(plotting.get("dpi", 120)))
                except Exception:
                    LOGGER.exception("Plotting failed for generation %04d; continuing", generation)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if best is None:
        raise RuntimeError("Study produced no generations")
    physical = space.physical(best[1])
    physical_by_name = dict(zip(space.keys, physical.tolist(), strict=True))
    result = {"loss": best[0], "normalized": best[1].tolist(),
              "physical": physical.tolist(), "physical_by_name": physical_by_name,
              "details": best[2], "generation": optimizer.state.generation}
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
        if len(selected) >= min(100, len(candidates)):
            break
        key = tuple(np.round(candidate["normalized"], 14))
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
    target_count = min(100, len(candidates))
    if len(selected) < target_count:
        raise RuntimeError(f"Expected {target_count} distinct basins, found {len(selected)}")
    for index, candidate in enumerate(selected[:target_count]):
        records.append({"basin_id": f"b{index:03d}", **candidate})
    output = output or stage1_run / "basins.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


def load_basins(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_hyper_stage(config: RunConfig, output_root: Path) -> None:
    """Run passive pre-calibration followed by full-model hyper fitting."""
    passive_values = run_passive_precalibration(config, output_root)
    stage1_dir = output_root / "stage1_hyper"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in config.section("stage1").get("seeds", range(10)):
        seed = int(seed)
        initial = _full_initial_from_passive(config, passive_values, seed)
        tasks.append((config, "hyper", seed,
                      stage1_dir / f"seed_{seed:03d}", initial, None))
    _run_studies(config, tasks)
    make_basins(config, stage1_dir)


def run_depolarizing_stage(config: RunConfig, output_root: Path,
                           basins: list[dict] | None = None) -> None:
    stage1_dir = output_root / "stage1_hyper"
    basins = basins if basins is not None else load_basins(stage1_dir / "basins.jsonl")
    stage2_dir = output_root / "stage2_depolarizing"
    tasks = []
    for basin in basins:
        mean = np.asarray(basin["normalized"], dtype=float)
        for seed in config.section("stage2").get("seeds", list(range(10))):
            seed = int(seed)
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(basin["basin_id"][1:])])
            )
            initial = np.clip(mean + rng.uniform(-0.15, 0.15, size=mean.size), 0.0, 1.0)
            tasks.append((config, "depolarizing", seed,
                          stage2_dir / basin["basin_id"] / f"seed_{seed:03d}",
                          initial, basin["basin_id"]))
    _run_studies(config, tasks)


def run_pipeline(config: RunConfig, output_root: Path) -> None:
    run_hyper_stage(config, output_root)
    run_depolarizing_stage(config, output_root)

"""Jaxley simulation and Optuna MOCMA orchestration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import OBJECTIVE_LABELS, PipelineConfig
from .features import FEATURE_LABELS as SCALAR_FEATURE_LABELS, extract_features
from .objectives import evaluate_objectives
from .pareto import annotate_front, trial_row, write_front_csv, write_jsonl
from .plotting import plot_solution


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_experimental(path: Path, records: tuple[Any, ...], values: dict[str, Any]) -> None:
    rows = []
    for record in records:
        item = values[record.trace_key]
        row = {"trace_key": record.trace_key, "protocol": record.protocol, **item["features"]}
        rows.append(row)
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(_json(values), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["trace_key", "protocol", *SCALAR_FEATURE_LABELS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json(row.get(key)) for key in fields})


class JaxleyEvaluator:
    """Build Jaxley once and evaluate arbitrary physical parameter vectors."""

    def __init__(self, config: PipelineConfig):
        import jax.numpy as jnp

        from jaxley_refactored.data import SegmentedTraceLoader, bucket_records
        from jaxley_refactored.models import default_builder
        from jaxley_refactored.simulation import InitialStateFactory, SimulationKernel

        self.config = config
        self.records = tuple(SegmentedTraceLoader().load(config.app_config.dataset))
        if not self.records:
            raise ValueError("No traces were selected")
        self.buckets = bucket_records(self.records, pad_to_longest=False)
        self.model = default_builder().build(config.app_config.model)
        self.jnp = jnp
        self.kernels = []
        self.states = []
        for bucket in self.buckets:
            states = InitialStateFactory(self.model.cell, bucket.dt_ms).build(bucket.initial_voltage_mV)
            kernel = SimulationKernel(
                self.model.cell,
                self.model.parameterizer,
                config.app_config.protocol,
                config.app_config.runtime,
                bucket.dt_ms,
                bucket.n_steps,
            )
            self.states.append(states)
            self.kernels.append(kernel)
        self.experimental = {
            record.trace_key: extract_features(record, record.voltage_mV, config.feature)
            for record in self.records
        }

    def simulate(self, parameters: dict[str, float]) -> dict[tuple[float, int], np.ndarray]:
        vector = self.jnp.asarray([parameters[name] for name in self.model.parameterizer.keys])
        predictions = {}
        for bucket, kernel, states in zip(self.buckets, self.kernels, self.states, strict=True):
            predictions[bucket.key] = np.asarray(
                kernel.simulate_batch(vector, self.jnp.asarray(bucket.currents_nA), states)
            )
        return predictions

    def evaluate(self, parameters: dict[str, float]) -> tuple[tuple[float, ...], dict[str, Any], dict[tuple[float, int], np.ndarray]]:
        predictions = self.simulate(parameters)
        values, details = evaluate_objectives(
            self.records,
            predictions,
            self.buckets,
            self.experimental,
            self.config.feature,
            self.config.objectives,
        )
        return values, details, predictions


def _write_solution(output: Path, evaluator: JaxleyEvaluator, trial: Any, row: dict[str, Any]) -> None:
    trial_dir = output / "solutions" / f"trial_{trial.number:06d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    values, details, predictions = evaluator.evaluate({key: float(value) for key, value in trial.params.items()})
    objective_values = {label: float(value) for label, value in zip(OBJECTIVE_LABELS, values, strict=True)}
    with (trial_dir / "objective_values.json").open("w", encoding="utf-8") as handle:
        json.dump(_json(objective_values), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (trial_dir / "features_simulated.json").open("w", encoding="utf-8") as handle:
        json.dump(_json(details), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (trial_dir / "features_simulated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trace_key", "protocol", *SCALAR_FEATURE_LABELS])
        writer.writeheader()
        for record in evaluator.records:
            feature_values = details["features_by_trace"][record.trace_key]["simulated"]["features"]
            writer.writerow({"trace_key": record.trace_key, "protocol": record.protocol, **{label: _json(feature_values.get(label)) for label in SCALAR_FEATURE_LABELS}})
    with (trial_dir / "objective_ranks.json").open("w", encoding="utf-8") as handle:
        json.dump(_json({"objective_ranks": row["objective_ranks"], "objective_gaps_from_front_best": row["objective_gaps_from_front_best"], "objective_percentiles": row["objective_percentiles"], "best_for_objectives": row["best_for_objectives"]}), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (trial_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key, value in trial.params.items():
            writer.writerow([key, float(value)])
    arrays = {}
    index = {}
    for bucket in evaluator.buckets:
        array = predictions[bucket.key]
        for item, record in enumerate(bucket.records):
            key = f"trace_{len(index):04d}"
            arrays[key] = array[item, : len(record.time_ms)]
            index[key] = record.trace_key
    np.savez_compressed(trial_dir / "predictions.npz", **arrays)
    with (trial_dir / "prediction_index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
    plot_solution(
        trial_dir / "solution.png",
        evaluator.records,
        predictions,
        evaluator.buckets,
        details,
        trial.number,
        objective_values,
        row["best_for_objectives"],
    )


def _storage_url(storage: str, output: Path) -> str:
    if storage in {"", "auto"}:
        database = (output / "study.db").resolve()
        return f"sqlite:///{database.as_posix()}"
    return storage


def _run_hash(config: PipelineConfig, app_hash: str) -> str:
    from dataclasses import asdict

    from jaxley_refactored.config.hashing import stable_hash

    return stable_hash(
        {
            "app_config": app_hash,
            "feature": asdict(config.feature),
            "objectives": {
                "labels": config.objectives.labels,
                "scales": dict(config.objectives.scales),
                "invalid_feature_penalty": config.objectives.invalid_feature_penalty,
                "overlap_sigma_mV": config.objectives.overlap_sigma_mV,
                "overlap_window_weights": dict(config.objectives.overlap_window_weights),
            },
            "seed": config.seed,
            "population_size": config.population_size,
        }
    )


def _write_run_state(output: Path, **values: Any) -> None:
    with (output / "run_state.json").open("w", encoding="utf-8") as handle:
        json.dump(_json(values), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _completed_trials(study: Any) -> int:
    return sum(1 for trial in study.trials if str(trial.state).endswith("COMPLETE"))


def run_pipeline(config: PipelineConfig) -> Path:
    """Run MOCMA and write the complete annotated Pareto front."""
    import optuna
    import optunahub

    from jaxley_refactored.config.hashing import config_as_dict, stable_hash
    from jaxley_refactored.runtime import collect_provenance, validate_device

    evaluator = JaxleyEvaluator(config)
    output = config.output_root / f"{config.study_name}-{config.app_config.dataset.cell_id}-seed{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "solutions").mkdir(exist_ok=True)
    _write_experimental(output / "experimental_features.csv", evaluator.records, evaluator.experimental)

    parameter_specs = tuple(evaluator.model.parameterizer.specs)
    search_space = {
        spec.name: optuna.distributions.FloatDistribution(float(spec.bounds[0]), float(spec.bounds[1]))
        for spec in parameter_specs
    }
    module = optunahub.load_module("samplers/mocma")
    sampler = module.MoCmaSampler(search_space=search_space, popsize=config.population_size, seed=config.seed)
    storage_url = _storage_url(config.storage, output)
    storage = optuna.storages.RDBStorage(
        storage_url,
        heartbeat_interval=config.heartbeat_interval_s,
        grace_period=config.grace_period_s,
    )
    app_hash = stable_hash(config_as_dict(config.app_config))
    run_hash = _run_hash(config, app_hash)
    study = optuna.create_study(
        study_name=config.study_name,
        directions=["minimize"] * len(OBJECTIVE_LABELS),
        sampler=sampler,
        storage=storage,
        load_if_exists=config.resume,
    )
    stored_hash = study.user_attrs.get("run_hash")
    if stored_hash is not None and stored_hash != run_hash:
        raise ValueError(
            "Existing study configuration does not match this run. "
            "Use a new study_name/output directory or keep the original settings."
        )
    study.set_user_attr("run_hash", run_hash)
    study.set_user_attr("population_size", config.population_size)
    study.set_user_attr("objective_labels", list(OBJECTIVE_LABELS))
    optuna.storages.fail_stale_trials(study)
    initial_trial_count = _completed_trials(study)

    def objective(trial: Any) -> tuple[float, ...]:
        parameters = {spec.name: trial.suggest_float(spec.name, float(spec.bounds[0]), float(spec.bounds[1])) for spec in parameter_specs}
        started = time.perf_counter()
        try:
            values, _details, _predictions = evaluator.evaluate(parameters)
            trial.set_user_attr("evaluation_seconds", time.perf_counter() - started)
            return values
        except Exception as error:
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            return tuple([config.objectives.invalid_feature_penalty] * len(OBJECTIVE_LABELS))

    remaining_trials = max(0, config.trials - _completed_trials(study))
    _write_run_state(
        output,
        status="running" if remaining_trials else "complete",
        target_trials=config.trials,
        trials_recorded=len(study.trials),
        completed_trials=_completed_trials(study),
        remaining_trials=remaining_trials,
        storage=storage_url,
    )
    try:
        if remaining_trials:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                gc_after_trial=True,
                callbacks=[
                    lambda current_study, trial: _write_run_state(
                        output,
                        status="running",
                        target_trials=config.trials,
                        trials_recorded=len(current_study.trials),
                        completed_trials=_completed_trials(current_study),
                        remaining_trials=max(0, config.trials - _completed_trials(current_study)),
                        last_trial=int(trial.number),
                        storage=storage_url,
                    )
                ],
            )
    except KeyboardInterrupt:
        _write_run_state(
            output,
            status="interrupted",
            target_trials=config.trials,
            trials_recorded=len(study.trials),
            completed_trials=_completed_trials(study),
            remaining_trials=max(0, config.trials - _completed_trials(study)),
            storage=storage_url,
        )
        raise
    except Exception as error:
        _write_run_state(
            output,
            status="failed",
            target_trials=config.trials,
            trials_recorded=len(study.trials),
            completed_trials=_completed_trials(study),
            remaining_trials=max(0, config.trials - _completed_trials(study)),
            error=f"{type(error).__name__}: {error}",
            storage=storage_url,
        )
        raise
    _write_run_state(
        output,
        status="complete",
        target_trials=config.trials,
        trials_recorded=len(study.trials),
        completed_trials=_completed_trials(study),
        remaining_trials=max(0, config.trials - _completed_trials(study)),
        storage=storage_url,
    )
    labels = OBJECTIVE_LABELS
    all_rows = [trial_row(trial, labels) for trial in study.trials if trial.values is not None]
    write_jsonl(output / "all_trials.jsonl", all_rows)
    front = annotate_front(study.best_trials, labels, config.objectives.tie_tolerance)
    write_jsonl(output / "pareto_front.jsonl", front)
    write_front_csv(output / "pareto_front.csv", front, labels)

    with (output / "objective_definitions.json").open("w", encoding="utf-8") as handle:
        json.dump(_json({"labels": labels, "directions": ["minimize"] * len(labels), "scales": config.objectives.scales, "overlap_sigma_mV": config.objectives.overlap_sigma_mV, "overlap_window_weights": config.objectives.overlap_window_weights, "tie_tolerance": config.objectives.tie_tolerance}), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        import yaml

        with config.source_path.open(encoding="utf-8") as source:
            yaml.safe_dump(yaml.safe_load(source), handle, sort_keys=False)
    device = validate_device(config.app_config.runtime)
    manifest = {
        "study_name": config.study_name,
        "sampler": "mocma",
        "seed": config.seed,
        "trials_requested": config.trials,
        "trials_completed": _completed_trials(study),
        "trials_recorded": len(study.trials),
        "storage": storage_url,
        "resumed": initial_trial_count > 0,
        "population_size": config.population_size,
        "model_signature": evaluator.model.signature,
        "parameter_names": list(evaluator.model.parameterizer.keys),
        "trace_keys": [record.trace_key for record in evaluator.records],
        "config_hash": stable_hash(config_as_dict(config.app_config)),
        **collect_provenance(Path(__file__).resolve().parents[2], device),
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    for trial, row in zip(study.best_trials, front, strict=True):
        _write_solution(output, evaluator, trial, row)
    return output

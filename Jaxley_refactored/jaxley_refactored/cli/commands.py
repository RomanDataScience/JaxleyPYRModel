"""User-facing orchestration; domain logic remains in dedicated services."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from jaxley_refactored.config.hashing import config_as_dict, stable_hash
from jaxley_refactored.data import (
    SegmentedTraceLoader,
    TraceBucket,
    bucket_records,
    weight_records,
)
from jaxley_refactored.runtime import collect_provenance, validate_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaxley-refactored")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "Validate YAML without building a cell."),
        ("inspect-data", "Load and summarize selected experimental traces."),
        ("inspect-model", "Build and summarize the configured model."),
        ("export-hoc-artifact", "Export a portable exact-HOC artifact."),
        ("simulate", "Run one selected trace or a short smoke simulation."),
        ("fit", "Fit shared parameters to every selected trace."),
        ("hybrid-fit", "Run serial CMA-ES, Adam exploration, and backtracking refinement."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        if name == "export-hoc-artifact":
            command.add_argument("--destination", required=True, type=Path)
        if name == "simulate":
            command.add_argument("--trace")
            command.add_argument("--protocol")
            command.add_argument("--max-steps", type=int)
            command.add_argument("--output", type=Path)
        if name in {"fit", "hybrid-fit"}:
            command.add_argument("--epochs", type=int)
            command.add_argument("--max-steps", type=int)
            command.add_argument("--seed", type=int)
            command.add_argument("--cell-id")
            command.add_argument("--run-name")
            command.add_argument("--dry-run", action="store_true")
    return parser


def run(config, argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        print(
            json.dumps(
                {
                    "valid": True,
                    "config_hash": stable_hash(config_as_dict(config)),
                    "source": str(config.source_path),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "inspect-data":
        return _inspect_data(config)
    if args.command == "inspect-model":
        return _inspect_model(config)
    if args.command == "export-hoc-artifact":
        return _export_artifact(config, args.destination)
    if args.command == "simulate":
        return _simulate(config, args)
    if args.command == "fit":
        return _fit(config, args)
    if args.command == "hybrid-fit":
        return _hybrid_fit(config, args)
    raise AssertionError(args.command)


def _records(config):
    records = SegmentedTraceLoader().load(config.dataset)
    return weight_records(
        records,
        aggregation=config.fit.aggregation,
        protocol_weights=config.fit.protocol_weights,
    )


def _inspect_data(config) -> int:
    records = _records(config)
    buckets = bucket_records(
        records, pad_to_longest=config.fit.pad_to_longest
    )
    print(
        json.dumps(
            {
                "cell_id": config.dataset.cell_id,
                "n_records": len(records),
                "weight_sum": sum(record.weight for record in records),
                "records": [
                    {
                        "trace": record.trace_id,
                        "protocol": record.protocol,
                        "shape": list(record.voltage_mV.shape),
                        "dt_ms": record.dt_ms,
                        "score_samples": int(record.score_mask.sum()),
                        "weight": record.weight,
                    }
                    for record in records
                ],
                "buckets": [
                    {
                        "dt_ms": bucket.dt_ms,
                        "n_steps": bucket.n_steps,
                        "n_traces": len(bucket.records),
                    }
                    for bucket in buckets
                ],
            },
            indent=2,
        )
    )
    return 0


def _build_model(config):
    validate_device(config.runtime)
    from jaxley_refactored.models import default_builder

    return default_builder().build(config.model)


def _inspect_model(config) -> int:
    model = _build_model(config)
    group_counts = {
        group: int(mask.sum()) for group, mask in model.features.group_masks.items()
    }
    print(
        json.dumps(
            {
                "signature": model.signature,
                "branches": len(model.cell.xyzr),
                "compartments": len(model.cell.nodes),
                "groups": group_counts,
                "enabled_mechanisms": sorted(model.enabled_mechanisms),
                "fit_parameters": list(model.parameterizer.keys),
                "provider": model.provenance,
            },
            indent=2,
        )
    )
    return 0


def _export_artifact(config, destination: Path) -> int:
    if config.model.morphology.provider != "hoc_live":
        raise ValueError("Artifact export requires morphology.provider=hoc_live.")
    model = _build_model(config)
    from jaxley_refactored.morphology import export_hoc_artifact

    manifest = export_hoc_artifact(
        model.cell,
        destination,
        provenance={
            "model_signature": model.signature,
            "source_config": str(config.source_path),
            **model.provenance,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _selected_record(config, args):
    records = _records(config)
    matching = [
        record
        for record in records
        if (args.trace is None or record.trace_id == args.trace)
        and (args.protocol is None or record.protocol == args.protocol)
    ]
    if not matching:
        raise ValueError("No record matches --trace/--protocol.")
    return matching[0]


def _simulate(config, args) -> int:
    import jax.numpy as jnp

    from jaxley_refactored.simulation import InitialStateFactory, SimulationKernel

    record = _selected_record(config, args)
    steps = len(record.time_ms)
    if args.max_steps is not None:
        if args.max_steps <= 1:
            raise ValueError("--max-steps must be greater than one.")
        steps = min(steps, args.max_steps)
    model = _build_model(config)
    current = jnp.asarray(record.current_nA[:steps])[None, :]
    initial_voltage = (
        record.initial_voltage_mV
        if config.protocol.initial_state_mode == "observed_first_sample"
        else config.protocol.fixed_voltage_mV
    )
    states = InitialStateFactory(model.cell, record.dt_ms).build([initial_voltage])
    kernel = SimulationKernel(
        model.cell,
        model.parameterizer,
        config.protocol,
        config.runtime,
        record.dt_ms,
        steps,
    )
    parameters = jnp.asarray(model.reference_values)
    voltage = np.asarray(kernel.simulate_batch(parameters, current, states)[0])
    result = {
        "trace": record.trace_key,
        "samples": len(voltage),
        "finite": bool(np.isfinite(voltage).all()),
        "min_mV": float(voltage.min()),
        "max_mV": float(voltage.max()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            voltage_mV=voltage,
            current_nA=np.asarray(current[0]),
            time_ms=np.arange(steps) * record.dt_ms,
        )
        result["output"] = str(args.output)
    print(json.dumps(result, indent=2))
    return 0


def _fit(config, args) -> int:
    from jaxley_refactored.fitting.checkpoints import CheckpointManager
    from jaxley_refactored.fitting.initialization import initial_physical_values
    from jaxley_refactored.fitting.trainer import Trainer
    from jaxley_refactored.reporting import RunDirectory, plot_epoch_traces

    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        optimizer = replace(config.fit.optimizer, epochs=args.epochs)
        config = replace(config, fit=replace(config.fit, optimizer=optimizer))
    if args.max_steps is not None and args.max_steps <= 1:
        raise ValueError("--max-steps must be greater than one.")
    if args.seed is not None:
        config = replace(config, runtime=replace(config.runtime, seed=args.seed))
    if args.cell_id is not None:
        if not args.cell_id.strip():
            raise ValueError("--cell-id cannot be empty.")
        config = replace(
            config, dataset=replace(config.dataset, cell_id=args.cell_id)
        )
    if args.run_name is not None:
        if not args.run_name.strip():
            raise ValueError("--run-name cannot be empty.")
        config = replace(
            config, output=replace(config.output, run_name=args.run_name)
        )

    records = _records(config)
    if args.max_steps is not None:
        records = tuple(
            record.with_max_steps(args.max_steps) for record in records
        )
    buckets = bucket_records(
        records, pad_to_longest=config.fit.pad_to_longest
    )
    print(
        f"fit_setup cell={config.dataset.cell_id} traces={len(records)} "
        f"buckets={[(len(bucket.records), bucket.n_steps) for bucket in buckets]} "
        f"max_steps={args.max_steps}",
        flush=True,
    )
    print("building_model", flush=True)
    model = _build_model(config)
    compatibility_hash = stable_hash(
        {
            "config": config_as_dict(config),
            "model": model.signature,
            "inputs": {
                record.trace_key: record.checksums for record in records
            },
            "max_steps": args.max_steps,
        }
    )
    run_id = (
        config.output.run_name
        if config.output.run_name != "auto"
        else (
            f"{config.model.model_id}-{config.dataset.cell_id}-"
            f"{model.signature[:8]}-{compatibility_hash[:8]}-"
            f"seed{config.runtime.seed}"
        )
    )
    output = RunDirectory(config.output.root, run_id)
    resolved_config = config_as_dict(config)
    resolved_config["run_overrides"] = {"max_steps": args.max_steps}
    output.write_yaml("resolved_config.yaml", resolved_config)
    device = validate_device(config.runtime)
    output.write_json(
        "run_manifest.json",
        {
            "compatibility_hash": compatibility_hash,
            "model_signature": model.signature,
            "model_provenance": model.provenance,
            "input_checksums": {
                record.trace_key: record.checksums for record in records
            },
            "max_steps": args.max_steps,
            "trace_shapes": {
                record.trace_key: len(record.time_ms) for record in records
            },
            **collect_provenance(Path(__file__).resolve().parents[3], device),
        },
    )
    output.write_parameters(
        "parameters_initial.csv",
        model.parameterizer.specs,
        initial_physical_values(model, config.fit, config.runtime),
    )
    if args.dry_run:
        output.write_json("status.json", {"status": "validated", "dry_run": True})
        print(
            json.dumps(
                {
                    "run": str(output.path),
                    "dry_run": True,
                    "max_steps": args.max_steps,
                },
                indent=2,
            )
        )
        return 0

    checkpoints = CheckpointManager(
        output.path / "checkpoints", compatibility_hash
    )
    trainer = Trainer(
        model,
        buckets,
        config.protocol,
        config.fit,
        config.runtime,
        checkpoints,
    )
    trainer.install_signal_handlers()
    print(
        "starting_optimization "
        "(the first epoch includes JAX forward/backward compilation)",
        flush=True,
    )

    def report(metrics, predictions):
        output.append_metrics(metrics)
        plot_path = None
        if config.output.plot_every_epochs and (
            (metrics["epoch"] + 1) % config.output.plot_every_epochs == 0
        ):
            plot_path = plot_epoch_traces(
                output.path / "plots",
                buckets,
                predictions,
                epoch=metrics["epoch"],
                loss=metrics["loss"],
            )
        message = (
            f"epoch={metrics['epoch']:04d} loss={metrics['loss']:.8g} "
            f"rmse_mV={metrics['rmse_mV']:.6g} "
            f"grad_norm={metrics['gradient_norm']:.6g} "
            f"lr={metrics['learning_rate']:.6g}"
        )
        if config.fit.optimizer.line_search.enabled:
            message += (
                f" accepted={str(metrics['step_accepted']).lower()}"
                f" trials={metrics['line_search_trials']}"
            )
        penalty_metrics = metrics["penalty_metrics"]
        if penalty_metrics:
            counts = ",".join(
                f"{label}:{value:.4g}"
                for label, value in penalty_metrics[
                    "soft_spike_counts"
                ].items()
            )
            message += (
                f" loss_multiplier={penalty_metrics['loss_multiplier']:.6g}"
                f" outside_soft_spikes={counts}"
            )
        if plot_path is not None:
            message += f" plot={plot_path}"
        print(message, flush=True)

    output.write_json("status.json", {"status": "running"})
    result = trainer.train(report)
    output.write_parameters(
        "parameters_best.csv",
        model.parameterizer.specs,
        np.asarray(result.best_parameters),
    )
    output.write_parameters(
        "parameters_final.csv",
        model.parameterizer.specs,
        np.asarray(result.final_parameters),
    )
    output.write_json(
        "status.json",
        {
            "status": "interrupted" if result.stopped_by_signal else "complete",
            "epochs_completed": result.epochs_completed,
            "best_loss": result.best_loss,
        },
    )
    return 0 if not result.stopped_by_signal else 75


def _hybrid_fit(config, args) -> int:
    from jaxley_refactored.fitting.hybrid import run_hybrid
    from jaxley_refactored.reporting import RunDirectory

    if config.search.strategy != "hybrid":
        raise ValueError("hybrid-fit requires search.strategy: hybrid.")
    if args.seed is not None:
        config = replace(config, runtime=replace(config.runtime, seed=args.seed))
    if args.cell_id is not None:
        config = replace(config, dataset=replace(config.dataset, cell_id=args.cell_id))
    if args.run_name is not None:
        config = replace(config, output=replace(config.output, run_name=args.run_name))

    training_records = _records(config)
    validation_dataset = replace(
        config.dataset, traces=(), trace_indices=(2, 4)
    )
    validation_config = replace(config, dataset=validation_dataset)
    validation_records = _records(validation_config)
    if args.max_steps is not None:
        training_records = tuple(
            record.with_max_steps(args.max_steps) for record in training_records
        )
        validation_records = tuple(
            record.with_max_steps(args.max_steps) for record in validation_records
        )
    training_buckets = bucket_records(
        training_records, pad_to_longest=config.fit.pad_to_longest
    )
    validation_buckets = bucket_records(
        validation_records, pad_to_longest=config.fit.pad_to_longest
    )
    model = _build_model(config)
    compatibility_hash = stable_hash(
        {
            "config": config_as_dict(config),
            "model": model.signature,
            "training_inputs": {
                record.trace_key: record.checksums for record in training_records
            },
            "validation_inputs": {
                record.trace_key: record.checksums for record in validation_records
            },
            "max_steps": args.max_steps,
        }
    )
    run_id = (
        config.output.run_name
        if config.output.run_name != "auto"
        else (
            f"hybrid-{config.dataset.cell_id}-{compatibility_hash[:8]}-"
            f"seed{config.runtime.seed}"
        )
    )
    output = RunDirectory(config.output.root, run_id)
    resolved = config_as_dict(config)
    resolved["run_overrides"] = {"max_steps": args.max_steps}
    output.write_yaml("resolved_config.yaml", resolved)
    device = validate_device(config.runtime)
    output.write_json(
        "run_manifest.json",
        {
            "compatibility_hash": compatibility_hash,
            "model_signature": model.signature,
            "training_trace_keys": [record.trace_key for record in training_records],
            "validation_trace_keys": [record.trace_key for record in validation_records],
            "training_input_checksums": {
                record.trace_key: record.checksums for record in training_records
            },
            "validation_input_checksums": {
                record.trace_key: record.checksums for record in validation_records
            },
            **collect_provenance(Path(__file__).resolve().parents[3], device),
        },
    )
    if args.dry_run:
        output.write_json("status.json", {"status": "validated", "dry_run": True})
        print(json.dumps({"run": str(output.path), "dry_run": True}, indent=2))
        return 0
    output.write_json("status.json", {"status": "running"})
    try:
        selected = run_hybrid(
            model=model,
            training_buckets=training_buckets,
            validation_buckets=validation_buckets,
            config=config,
            output=output,
            compatibility_hash=compatibility_hash,
        )
    except Exception as error:
        output.write_json(
            "status.json",
            {
                "status": (
                    "interrupted"
                    if isinstance(error, InterruptedError)
                    else "failed"
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    output.write_json(
        "status.json",
        {
            "status": "complete",
            "selected_candidate": selected["candidate_id"],
            "training_loss": selected["training_loss"],
            "validation_loss": selected["validation_loss"],
        },
    )
    print(
        json.dumps(
            {
                "run": str(output.path),
                "selected_candidate": selected["candidate_id"],
                "validation_loss": selected["validation_loss"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit("Use jaxley_refactored.cli.bootstrap so runtime is configured first.")

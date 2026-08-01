#!/usr/bin/env python3
"""Evaluate one fixed fitted candidate at several simulation timesteps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

# Permit direct execution from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxley_refactored.config import load_config
from jaxley_refactored.data import SegmentedTraceLoader, bucket_records, weight_records
from jaxley_refactored.runtime import validate_device
from jaxley_refactored.runtime.bootstrap import configure_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--candidate-id",
        help="Candidate to evaluate; defaults to the lowest-loss successful candidate.",
    )
    parser.add_argument(
        "--dt-ms", nargs="+", type=float, default=[0.1, 0.05, 0.025]
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_candidate(path: Path, candidate_id: str | None) -> dict:
    candidates = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate.get("status") != "ok":
                continue
            if candidate_id is None or candidate.get("candidate_id") == candidate_id:
                candidates.append(candidate)
    if not candidates:
        requested = candidate_id or "a successful candidate"
        raise ValueError(f"Could not find {requested} in {path}.")
    if candidate_id is not None:
        return candidates[-1]
    return min(candidates, key=lambda item: item["training_loss"])


def records_for(config):
    records = SegmentedTraceLoader().load(config.dataset)
    return weight_records(
        records,
        aggregation=config.fit.aggregation,
        protocol_weights=config.fit.protocol_weights,
    )


def main() -> int:
    args = parse_args()
    if any(dt <= 0.0 for dt in args.dt_ms):
        raise ValueError("Every --dt-ms value must be positive.")
    config = load_config(args.config)
    configure_environment(config.runtime)
    validate_device(config.runtime)

    import jax.numpy as jnp

    from jaxley_refactored.fitting.trainer import Trainer
    from jaxley_refactored.models import default_builder

    candidate = load_candidate(args.candidates, args.candidate_id)
    normalized = jnp.asarray(candidate["normalized"])
    evaluations = []
    predictions_by_dt = {}

    for dt in args.dt_ms:
        dt_config = replace(
            config,
            dataset=replace(config.dataset, target_dt_ms=float(dt)),
        )
        records = records_for(dt_config)
        buckets = bucket_records(records, pad_to_longest=dt_config.fit.pad_to_longest)
        model = default_builder().build(dt_config.model)
        trainer = Trainer(
            model, buckets, dt_config.protocol, dt_config.fit, dt_config.runtime
        )
        result = trainer.evaluate(normalized, gradient=False)
        loss, _, bucket_losses, predictions, components, mse, penalties = result
        trace_predictions = {}
        for prepared in trainer._prepared:
            values = predictions[prepared.bucket.key]
            for index, record in enumerate(prepared.bucket.records):
                trace_predictions[record.trace_key] = {
                    "time_ms": np.asarray(record.time_ms),
                    "voltage_mV": np.asarray(values[index, : len(record.time_ms)]),
                }
        predictions_by_dt[float(dt)] = trace_predictions
        evaluations.append(
            {
                "dt_ms": float(dt),
                "loss": float(loss),
                "evaluation_mse": float(mse),
                "component_losses": components,
                "bucket_losses": bucket_losses,
                "penalty_metrics": penalties,
            }
        )
        print(f"dt_ms={dt:g} loss={float(loss):.10g}", flush=True)

    reference_dt = min(predictions_by_dt)
    reference = predictions_by_dt[reference_dt]
    for evaluation in evaluations:
        dt = evaluation["dt_ms"]
        comparisons = {}
        for trace_key, reference_trace in reference.items():
            trace = predictions_by_dt[dt][trace_key]
            aligned = np.interp(
                reference_trace["time_ms"], trace["time_ms"], trace["voltage_mV"]
            )
            difference = aligned - reference_trace["voltage_mV"]
            comparisons[trace_key] = {
                "rmse_mV": float(np.sqrt(np.mean(difference**2))),
                "max_abs_mV": float(np.max(np.abs(difference))),
            }
        evaluation["versus_finest_dt_ms"] = reference_dt
        evaluation["voltage_difference"] = comparisons

    report = {
        "candidate_id": candidate["candidate_id"],
        "candidate_source_loss": candidate["training_loss"],
        "config": str(args.config),
        "candidate_file": str(args.candidates),
        "evaluations": evaluations,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

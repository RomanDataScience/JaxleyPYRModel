"""Compare two morphology discretizations with one fixed parameter vector."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


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
    return value


def _parameters(path: Path, keys: tuple[str, ...]) -> dict[str, float] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        values = {row["parameter"]: float(row["value"]) for row in csv.DictReader(handle)}
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"Parameter file is missing {len(missing)} model parameters")
    return {key: values[key] for key in keys}


def _prediction_by_trace(evaluator: Any, predictions: dict[Any, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for bucket in evaluator.buckets:
        values = predictions[bucket.key]
        for row, record in enumerate(bucket.records):
            result[record.trace_key] = values[row, : len(record.time_ms)]
    return result


def _run_at_resolution(config: Any, d_lambda: float, parameters: dict[str, float] | None):
    from newjaxley_multiobjective.runner import JaxleyEvaluator

    app_config = replace(
        config.app_config,
        model=replace(
            config.app_config.model,
            morphology=replace(config.app_config.model.morphology, d_lambda=d_lambda),
        ),
    )
    evaluator = JaxleyEvaluator(replace(config, app_config=app_config))
    if parameters is None:
        parameters = dict(zip(evaluator.model.parameterizer.keys, evaluator.model.reference_values, strict=True))
    values, details, predictions = evaluator.evaluate(parameters)
    return evaluator, values, details, _prediction_by_trace(evaluator, predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coarse", type=float, default=0.3)
    parser.add_argument("--fine", type=float, default=0.1)
    parser.add_argument("--parameters", type=Path, help="Optional solution parameters.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxley_mocma_mpl")

    from newjaxley_multiobjective.config import load_pipeline_config
    from jaxley_refactored.runtime.bootstrap import configure_environment

    config = load_pipeline_config(args.config)
    configure_environment(config.app_config.runtime)

    coarse_evaluator, coarse_values, coarse_details, coarse_predictions = _run_at_resolution(
        config, args.coarse, None
    )
    parameters = _parameters(args.parameters, coarse_evaluator.model.parameterizer.keys)
    if parameters is not None:
        coarse_evaluator, coarse_values, coarse_details, coarse_predictions = _run_at_resolution(
            config, args.coarse, parameters
        )
    fine_evaluator, fine_values, fine_details, fine_predictions = _run_at_resolution(
        config, args.fine, parameters
    )

    coarse_records = {record.trace_key: record for record in coarse_evaluator.records}
    fine_records = {record.trace_key: record for record in fine_evaluator.records}
    if set(coarse_records) != set(fine_records):
        raise ValueError("The two discretizations selected different trace keys")

    comparison = {
        "config": str(args.config.resolve()),
        "coarse_d_lambda": args.coarse,
        "fine_d_lambda": args.fine,
        "parameter_source": str(args.parameters.resolve()) if args.parameters else "model_reference_values",
        "objective_values": {
            "coarse": dict(zip(coarse_evaluator.objective_labels, coarse_values, strict=True)),
            "fine": dict(zip(fine_evaluator.objective_labels, fine_values, strict=True)),
        },
        "traces": {},
    }
    for key, record in coarse_records.items():
        fine_record = fine_records[key]
        if not np.array_equal(record.time_ms, fine_record.time_ms):
            raise ValueError(f"Time grids differ for {key}")
        difference = fine_predictions[key] - coarse_predictions[key]
        coarse_features = coarse_details["features_by_trace"][key]["simulated"]["features"]
        fine_features = fine_details["features_by_trace"][key]["simulated"]["features"]
        comparison["traces"][key] = {
            "mean_abs_voltage_difference_mV": float(np.mean(np.abs(difference))),
            "rmse_voltage_difference_mV": float(np.sqrt(np.mean(difference**2))),
            "max_abs_voltage_difference_mV": float(np.max(np.abs(difference))),
            "coarse_features": coarse_features,
            "fine_features": fine_features,
            "feature_differences": {
                name: float(fine_features[name] - coarse_features[name])
                for name in coarse_features
                if np.isfinite(coarse_features[name]) and np.isfinite(fine_features[name])
            },
        }

    output = args.output or (
        config.output_root / f"discretization_comparison-{config.app_config.dataset.cell_id}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "comparison.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json(comparison), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(path)
    for key, values in comparison["traces"].items():
        print(
            f"{key}: RMSE={values['rmse_voltage_difference_mV']:.6g} mV, "
            f"max={values['max_abs_voltage_difference_mV']:.6g} mV"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

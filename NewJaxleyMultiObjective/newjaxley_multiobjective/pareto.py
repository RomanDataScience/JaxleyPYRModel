"""Reporting helpers for Optuna's nondominated trials."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def trial_row(trial: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    values = {} if trial.values is None else {
        label: float(value) for label, value in zip(labels, trial.values, strict=True)
    }
    return {
        "trial": int(trial.number),
        "state": str(trial.state),
        "values": values,
        "parameters": {str(key): float(value) for key, value in trial.params.items()},
        "user_attrs": dict(getattr(trial, "user_attrs", {})),
    }


def annotate_front(trials: Iterable[Any], labels: tuple[str, ...], tie_tolerance: float) -> list[dict[str, Any]]:
    trials = tuple(trials)
    if not trials:
        return []
    matrix = np.asarray([[float(value) for value in trial.values] for trial in trials], dtype=float)
    minimum = np.min(matrix, axis=0)
    result = []
    for row_index, trial in enumerate(trials):
        values = matrix[row_index]
        ranks = 1 + np.sum(matrix < (values[None, :] - tie_tolerance), axis=0)
        gaps = values - minimum
        percentiles = 100.0 * np.mean(matrix <= values[None, :], axis=0)
        winners = [label for index, label in enumerate(labels) if abs(gaps[index]) <= tie_tolerance]
        row = trial_row(trial, labels)
        row.update({"objective_ranks": {label: int(ranks[index]) for index, label in enumerate(labels)}, "objective_gaps_from_front_best": {label: float(gaps[index]) for index, label in enumerate(labels)}, "objective_percentiles": {label: float(percentiles[index]) for index, label in enumerate(labels)}, "best_for_objectives": winners})
        result.append(row)
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_front_csv(path: Path, rows: list[dict[str, Any]], labels: tuple[str, ...]) -> None:
    import csv

    fields = ["trial", "best_for_objectives"] + list(labels)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = row["values"]
            writer.writerow({"trial": row["trial"], "best_for_objectives": ";".join(row["best_for_objectives"]), **values})

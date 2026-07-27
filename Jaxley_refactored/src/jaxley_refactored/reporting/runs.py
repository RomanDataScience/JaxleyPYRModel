"""Collision-safe run directories and small durable outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


class RunDirectory:
    def __init__(self, root: Path, run_id: str):
        self.path = root / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        claim = self.path / ".claim"
        try:
            claim.touch(exist_ok=False)
        except FileExistsError:
            # A completed/resumable run owns its directory already. Only the
            # same process/config should proceed, checked by checkpoint hashes.
            pass
        (self.path / "checkpoints").mkdir(exist_ok=True)

    def write_yaml(self, name: str, value: Any) -> None:
        with (self.path / name).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False)

    def write_json(self, name: str, value: Any) -> None:
        with (self.path / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def append_metrics(self, value: Any) -> None:
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")

    def write_parameters(
        self,
        name: str,
        specifications: Iterable,
        values,
    ) -> None:
        with (self.path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["parameter", "value", "lower", "upper", "units"],
            )
            writer.writeheader()
            for spec, value in zip(specifications, values, strict=True):
                writer.writerow(
                    {
                        "parameter": spec.name,
                        "value": float(value),
                        "lower": spec.bounds[0],
                        "upper": spec.bounds[1],
                        "units": spec.units,
                    }
                )


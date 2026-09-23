"""Command-line entry points for the NEURON optimization pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .stages import (
    load_basins,
    make_basins,
    run_hyper_stage,
    run_depolarizing_stage,
    run_passive_precalibration,
    run_pipeline,
    run_study,
    run_stage3,
    validate_current_replay,
)


def _config(path: str, workers: int | None):
    config = load_config(path)
    if workers is not None:
        config.raw.setdefault("runtime", {})["parallel_workers"] = workers
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEURON Combe staged CMA-ES optimization")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)

    for command in ("run-passive", "run-hyper", "run-depolarizing", "run-stage3", "run-all"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        item.add_argument("--output-dir", type=Path, default=Path("runs"))
        item.add_argument("--workers", type=int)

    current = sub.add_parser("check-current")
    current.add_argument("--config", required=True)
    current.add_argument("--output-dir", type=Path, default=Path("runs"))

    basins = sub.add_parser("make-basins")
    basins.add_argument("--config", required=True)
    basins.add_argument("--stage1-run", type=Path, required=True)
    basins.add_argument("--output", type=Path)

    report = sub.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config)
        errors = config.validate()
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 2
        print(json.dumps({"valid": True, "config_hash": config.hash(),
                          "workers": config.workers,
                          "parameters": len(config.parameters.keys)}, indent=2))
        return 0

    if args.command == "make-basins":
        config = load_config(args.config)
        records = make_basins(config, args.stage1_run, output=args.output)
        print(f"wrote {len(records)} basins")
        return 0

    config = _config(args.config, getattr(args, "workers", None))
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    output = args.output_dir.resolve()
    if args.command == "check-current":
        report = validate_current_replay(config, output / "stage0_passive" / "current_replay.json")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "run-passive":
        values = run_passive_precalibration(config, output)
        print(json.dumps({"completed": "passive", "physical_by_name": values}, indent=2))
        return 0
    if args.command == "run-hyper":
        run_hyper_stage(config, output)
    elif args.command == "run-depolarizing":
        basins_path = output / "stage1_hyper" / "basins.jsonl"
        basins = load_basins(basins_path)
        run_depolarizing_stage(config, output, basins)
    elif args.command == "run-stage3":
        run_stage3(config, output)
    else:
        run_pipeline(config, output)
    print(f"completed {args.command}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

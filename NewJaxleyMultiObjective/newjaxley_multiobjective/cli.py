"""Command-line entry point for the feature-fitting pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jaxley-mocma")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument(
            "--seed",
            type=int,
            help="Override multi_objective.seed for this run.",
        )
        command.add_argument(
            "--d-lambda",
            type=float,
            help="Override morphology discretization d_lambda for this run.",
        )
        command.add_argument(
            "--trials",
            type=int,
            help="Override the requested number of trials for this run.",
        )
    args = parser.parse_args(argv)

    from .config import load_pipeline_config
    from .objectives import objective_labels_for_trace_ids

    config = load_pipeline_config(args.config)
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be nonnegative")
        config = replace(
            config,
            seed=args.seed,
            app_config=replace(
                config.app_config,
                runtime=replace(config.app_config.runtime, seed=args.seed),
            ),
        )
    if args.d_lambda is not None:
        if args.d_lambda <= 0:
            parser.error("--d-lambda must be positive")
        config = replace(
            config,
            study_name=f"{config.study_name}-dlambda{args.d_lambda:g}".replace(".", "p"),
            app_config=replace(
                config.app_config,
                model=replace(
                    config.app_config.model,
                    morphology=replace(
                        config.app_config.model.morphology,
                        d_lambda=args.d_lambda,
                    ),
                ),
            ),
        )
    if args.trials is not None:
        if args.trials <= 0:
            parser.error("--trials must be positive")
        config = replace(config, trials=args.trials)
    if args.command == "validate":
        labels = objective_labels_for_trace_ids(
            config.objectives,
            config.optimization_trace_indices,
        )
        print(
            f"valid sampler={config.sampler} objectives={len(labels)} parameters=from-model "
            f"cell={config.app_config.dataset.cell_id} "
            f"optimization_traces={config.optimization_trace_indices} "
            f"test_traces={config.test_trace_indices} protocols={config.protocols} "
            f"d_lambda={config.app_config.model.morphology.d_lambda} "
            f"parallel_workers={config.parallel_workers}"
        )
        return 0

    # Runtime variables must be configured before importing JAX/Jaxley.
    from jaxley_refactored.runtime.bootstrap import configure_environment

    configure_environment(config.app_config.runtime)
    from .runner import run_pipeline

    output = run_pipeline(config)
    print(f"completed run={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

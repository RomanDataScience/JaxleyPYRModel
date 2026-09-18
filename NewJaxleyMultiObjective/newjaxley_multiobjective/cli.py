"""Command-line entry point for the MOCMA pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jaxley-mocma")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    from .config import load_pipeline_config

    config = load_pipeline_config(args.config)
    if args.command == "validate":
        print(
            f"valid sampler=mocma objectives=11 parameters=from-model "
            f"cell={config.app_config.dataset.cell_id} "
            f"optimization_traces={config.optimization_trace_indices} "
            f"test_traces={config.test_trace_indices} "
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

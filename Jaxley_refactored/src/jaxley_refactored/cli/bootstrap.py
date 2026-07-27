"""JAX-free CLI bootstrap.

This module is the console entry point. It reads runtime configuration and sets
environment variables before importing any numerical command implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from jaxley_refactored.config import load_config
from jaxley_refactored.runtime.bootstrap import configure_environment


def _config_path(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command")
    parser.add_argument("--config", required=True, type=Path)
    known, _ = parser.parse_known_args(argv)
    return known.config


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config(_config_path(argv))
    configure_environment(config.runtime)
    from .commands import run

    return run(config, argv)


if __name__ == "__main__":
    raise SystemExit(main())


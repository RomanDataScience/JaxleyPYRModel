#!/usr/bin/env python3
"""CMA-ES fit of the CA1 pyramidal cell model to one recorded cell.

The model is ``ca1_model.py`` with the parameter catalog ``ca1_parameters.py``;
the bounded CMA-ES (``fitting/optimizer.py``), the joint objective over four
depolarizing steps and one hyperpolarizing pulse (``fitting/objective.py``),
the data loader and the plots are in ``fitting/``.

    python run_cma.py --config configs/cma_m20240527cd.yaml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ca1_model import apply_parameters, build_cell, build_mechanisms, simulate  # noqa: E402
from ca1_parameters import ALL_KEYS, BOUNDS, CATALOG_KEYS, DEFAULTS  # noqa: E402
from fitting.data import load_dataset  # noqa: E402
from fitting.objective import joint_score  # noqa: E402
from fitting.optimizer import run_optimization  # noqa: E402
from fitting.plotting import fit_plotter, generation_plotter  # noqa: E402


class Evaluator:
    """Builds the cell lazily and scores complete parameter sets."""

    def __init__(self, traces, base_values: dict[str, float], simulation: dict,
                 hyper_weight: float, objective_options: dict):
        self.traces = traces
        self.base_values = dict(base_values)
        self.simulation = dict(simulation)
        self.hyper_weight = float(hyper_weight)
        self.objective_options = dict(objective_options)
        self._h = None

    def complete(self, values: dict[str, float]) -> dict[str, float]:
        return {**self.base_values, **{key: float(value) for key, value in values.items()}}

    def simulate(self, values: dict[str, float]) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._h is None:
            self._h = build_cell()
        apply_parameters(self._h, self.complete(values))
        outputs = []
        for trace in self.traces:
            time, soma_v, _ = simulate(self._h, trace, **self.simulation)
            outputs.append((time, soma_v))
        return outputs

    def evaluate(self, values: dict[str, float]):
        simulations = self.simulate(values)
        loss, details = joint_score(self.traces[:-1], self.traces[-1], simulations,
                                    hyper_weight=self.hyper_weight, **self.objective_options)
        return loss, details, simulations


_WORKER: Evaluator | None = None


def _initialize_worker(evaluator: Evaluator) -> None:
    global _WORKER
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    _WORKER = evaluator


def _evaluate_worker(values: dict[str, float]):
    if _WORKER is None:
        raise RuntimeError("Candidate worker was not initialized")
    try:
        return _WORKER.evaluate(values)
    except Exception as exc:  # a failed candidate must not stop the generation
        return 1.0e6, {"error": f"{type(exc).__name__}: {exc}"}, []


def _resolve_parameters(config: dict) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    section = config.get("parameters", {}) or {}
    include = list(section.get("include") or ALL_KEYS)
    exclude = set(section.get("exclude") or [])
    fixed = {key: float(value) for key, value in (section.get("fixed") or {}).items()}
    unknown = sorted((set(include) | exclude | set(fixed)) - set(CATALOG_KEYS))
    if unknown:
        raise ValueError(f"Unknown CA1 parameter(s): {unknown}")
    keys = [key for key in include if key not in exclude and key not in fixed]
    if not keys:
        raise ValueError("No free parameters remain after applying include/exclude/fixed")
    overrides = section.get("overrides", {}) or {}
    resolved = {}
    for key in keys:
        default = float(overrides.get(key, {}).get("default", DEFAULTS[key]))
        lower = float(overrides.get(key, {}).get("lower", BOUNDS[key][0]))
        upper = float(overrides.get(key, {}).get("upper", BOUNDS[key][1]))
        if not np.isfinite([default, lower, upper]).all() or not lower <= default <= upper:
            raise ValueError(f"Invalid bounds/default for {key}: {lower}, {default}, {upper}")
        resolved[key] = {"default": default, "lower": lower, "upper": upper}
    config["parameters"] = resolved
    config["fixed_parameters"] = fixed
    return resolved, fixed


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "cma_m20240527cd.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="override optimizer.seed; output goes to <output_dir>_seed<N>")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="one generation of two candidates in one process")
    parser.add_argument("--validate-only", action="store_true",
                        help="build the cell, score the default parameters and exit")
    parser.add_argument("--list-parameters", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.list_parameters:
        for key in CATALOG_KEYS:
            print(f"{key}: default={DEFAULTS[key]} bounds={BOUNDS[key]}"
                  + ("" if key in ALL_KEYS else "  (optional)"))
        return 0

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    optimizer = config.setdefault("optimizer", {})
    for key, value in (("generations", args.generations), ("seed", args.seed),
                       ("population_size", args.population_size), ("workers", args.workers)):
        if value is not None:
            optimizer[key] = value
    if args.smoke_test:
        optimizer.update({"generations": 1, "population_size": 2, "workers": 1})
    parameters, fixed = _resolve_parameters(config)

    base_dir = config_path.parent
    depolarizing, hyperpolarizing = load_dataset(
        (base_dir / config["data_root"]).resolve(), config["cell"],
        list(config["depolarizing_traces"]), config["hyperpolarizing_trace"],
    )
    traces = [*depolarizing, hyperpolarizing]
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.smoke_test:
        output_dir = (HERE / "runs" / "smoke_test").resolve()
    else:
        output_dir = (base_dir / config["output_dir"]).resolve()
        if args.seed is not None:
            output_dir = output_dir.with_name(f"{output_dir.name}_seed{args.seed}")

    build_mechanisms()
    evaluator = Evaluator(
        traces, {**DEFAULTS, **fixed}, config.get("simulation", {}) or {},
        float(config.get("hyper_weight", 1.0)), dict(config.get("objective", {}) or {}),
    )

    if args.validate_only:
        loss, details, _ = evaluator.evaluate({})
        print(f"Cell {config['cell']}: {len(depolarizing)} depolarizing + 1 hyperpolarizing trace")
        print(f"Free parameters: {len(parameters)}; fixed: {fixed}")
        print(f"Default-parameter joint loss: {loss:.4g}")
        for item in details["traces"]:
            print(f"  {item['trace']:8s} {item['kind']:15s} loss={item['loss']:.4g} "
                  f"baseline exp/sim={item['baseline_mV']['experimental']:.2f}/"
                  f"{item['baseline_mV']['simulated']:.2f} mV")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                                     encoding="utf-8")
    (output_dir / "parameter_catalog.json").write_text(
        json.dumps({"source": "CA1_pyramidal_cell_model/ca1_parameters.py",
                    "parameters": parameters, "fixed": fixed}, indent=2) + "\n", encoding="utf-8")

    workers = int(optimizer.get("workers", 1))
    if workers < 1:
        raise ValueError("optimizer.workers must be at least 1")
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                       initializer=_initialize_worker, initargs=(evaluator,))
        population_evaluator = lambda candidates: list(executor.map(_evaluate_worker, candidates))
    else:
        population_evaluator = lambda candidates: [evaluator.evaluate(c) for c in candidates]
    plotting = config.get("plotting", {}) or {}
    try:
        result = run_optimization(
            config, config_path=config_path, output_dir=output_dir,
            evaluator=lambda values: evaluator.evaluate(values)[:2],
            simulator=evaluator.simulate,
            plotter=fit_plotter(depolarizing, hyperpolarizing, evaluator.simulate),
            population_evaluator=population_evaluator,
            generation_plotter=generation_plotter(
                depolarizing, hyperpolarizing, output_dir,
                top_k=int(plotting.get("top_k", 6)), dpi=int(plotting.get("dpi", 110)),
                title=f"CA1 model CMA-ES ({config['cell']})",
            ),
            resume=not args.no_resume and not args.smoke_test,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    print(f"Best joint loss: {result['loss']:.6g}")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

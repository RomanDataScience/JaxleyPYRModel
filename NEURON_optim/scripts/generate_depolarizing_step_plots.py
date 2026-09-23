"""Generate short depolarizing-step plots from saved Stage 2 archives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neuron_optim.config import load_config
from neuron_optim.data import load_protocol_traces
from neuron_optim.objective import SimulationOutput
from neuron_optim.plotting import plot_depolarizing_step_generation


def _saved_simulations(archive, population_size: int, trace_count: int):
    valid = np.asarray(archive["valid"], dtype=bool)
    simulations = []
    for candidate_index in range(population_size):
        if not valid[candidate_index]:
            simulations.append(None)
            continue
        candidate = []
        for trace_index in range(trace_count):
            candidate.append(SimulationOutput(
                np.asarray(archive[
                    f"candidate_{candidate_index:04d}_trace_{trace_index:04d}_time_ms"
                ], dtype=float),
                np.asarray(archive[
                    f"candidate_{candidate_index:04d}_trace_{trace_index:04d}_voltage_mV"
                ], dtype=float),
            ))
        simulations.append(candidate)
    return simulations


def generate(config_path: Path, output_root: Path) -> int:
    config = load_config(config_path)
    traces = load_protocol_traces(
        config.data_root,
        cell=config.cell,
        protocol="depolarizing_step",
        trace_names=config.trace_names,
        pre_ms=config.simulation_pre_ms,
        post_ms=config.simulation_post_ms,
        full_trial=True,
    )
    plotting = dict(config.raw.get("plotting", {}))
    top_k = int(plotting.get("top_k", 10))
    dpi = int(plotting.get("dpi", 120))
    generated = 0

    stage2_root = output_root / "stage2_depolarizing"
    for run_dir in sorted(stage2_root.glob("b*/seed_*")):
        for population_path in sorted(run_dir.glob("population_generation_*.npz")):
            generation = int(population_path.stem.rsplit("_", 1)[1])
            simulation_path = run_dir / f"simulations_generation_{generation:04d}.npz"
            if not simulation_path.exists():
                continue
            with np.load(population_path, allow_pickle=False) as values:
                population = np.asarray(values["population"], dtype=float)
                losses = np.asarray(values["losses"], dtype=float)
            with np.load(simulation_path, allow_pickle=False) as archive:
                simulations = _saved_simulations(
                    archive, len(population), len(traces)
                )
            top_indices = np.argsort(losses, kind="stable")[:min(top_k, len(losses))]
            selected = [simulations[index] for index in top_indices]
            if not all(item is not None for item in selected):
                continue
            plot_depolarizing_step_generation(
                output_dir=run_dir / "plots",
                generation=generation,
                traces=traces,
                population=population[top_indices],
                losses=losses[top_indices],
                simulations=selected,
                space=config.parameters,
                top_k=top_k,
                dpi=dpi,
                population_indices=top_indices,
            )
            generated += 1
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generated = generate(args.config.resolve(), args.output_dir.resolve())
    print(f"generated {generated} depolarizing step figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

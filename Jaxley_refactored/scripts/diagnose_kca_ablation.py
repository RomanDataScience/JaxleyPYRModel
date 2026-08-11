"""Compare KCa ablations for one saved CMA-ES candidate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_refactored.cli.commands import _build_model, _records
from jaxley_refactored.config import load_config
from jaxley_refactored.simulation import InitialStateFactory, SimulationKernel


def _candidate(path: Path, generation: int) -> dict:
    candidates = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("generation") == generation and item.get("status") == "ok":
                candidates.append(item)
    return min(candidates, key=lambda item: item["training_loss"])


def _slope(time_ms, voltage, start_ms, end_ms):
    mask = (time_ms >= start_ms) & (time_ms <= end_ms)
    return float(np.polyfit(time_ms[mask], voltage[mask], 1)[0] * 100.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--equilibrate-ms", type=float, default=None)
    args = parser.parse_args()

    config = load_config(Path("configs/LSU_1_cma_adam.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, cell_id="m20240527cd"),
    )
    if args.equilibrate_ms is not None:
        config = replace(
            config,
            protocol=replace(
                config.protocol, equilibrate_ms=args.equilibrate_ms
            ),
        )
    model = _build_model(config)
    records = tuple(
        record for record in _records(config)
        if record.protocol == "hyperpolarizing_pulse"
        and record.trace_key.split("/")[1] in {"v75ctrl", "v77ctrl"}
    )
    parameters = np.asarray(
        _candidate(args.run / "global/candidates.jsonl", args.generation)["physical"]
    )
    keys = model.parameterizer.keys
    currents = jnp.asarray(np.stack([record.current_nA for record in records]))
    voltages = np.asarray([record.initial_voltage_mV for record in records])
    states = InitialStateFactory(model.cell, records[0].dt_ms).build(voltages)
    kernel = SimulationKernel(
        model.cell,
        model.parameterizer,
        config.protocol,
        config.runtime,
        records[0].dt_ms,
        len(records[0].time_ms),
    )

    variants = {
        "original": {},
        "nap_off": {"nap_gnabar": 0.0},
        "mykca_off": {"mykca_init": 0.0},
        "soma_kca_off": {"soma_kca": 0.0},
        "both_off": {"mykca_init": 0.0, "soma_kca": 0.0},
    }
    output = {}
    for label, replacements in variants.items():
        values = parameters.copy()
        for name, value in replacements.items():
            values[keys.index(name)] = value
        predicted = np.asarray(
            kernel.simulate_batch(jnp.asarray(values), currents, states)
        )
        output[label] = {
            record.trace_key: {
                "pre_slope_mV_per_100ms": _slope(
                    record.time_ms, trace, 400.0, 500.0
                ),
                "recovery_slope_mV_per_100ms": _slope(
                    record.time_ms, trace, 550.0, 650.0
                ),
                "pre_delta_mV": float(
                    np.interp(500.0, record.time_ms, trace)
                    - np.interp(400.0, record.time_ms, trace)
                ),
                "recovery_delta_mV": float(
                    np.interp(650.0, record.time_ms, trace)
                    - np.interp(550.0, record.time_ms, trace)
                ),
            }
            for record, trace in zip(records, predicted, strict=True)
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

# Combe staged optimization

This package implements the staged CMA-ES plan using the Combe2023
model. The checked-in configurations select the repository's NEURON backend;
the Jaxley backend remains available with `runtime.backend: jaxley`.

The checked-in hyperpolarizing settings use 3 seeds, 100 generations, 20
offspring, and 1 process worker for the passive and stage-1 searches. The
depolarizing Stage 2 retains 10 seeds, 200 generations, and 30 offspring and
optimizes the complete active-current parameter set in one simulation stage.

See [STAGES.md](STAGES.md) for the complete Stage 0 → Stage 1 → Stage 2
solution handoff.

For Optuna-style importance analysis of the calibrated parameters in each
stage, see [hyperparameter_analyses/README.md](hyperparameter_analyses/README.md).

Hyperpolarizing trials are simulated over the existing 500 ms pre-step and
600 ms post-step window, subject to the available recorded data. Their fitness
and plots use the final 100 ms before the hyperpolarizing step onward. The
hyperpolarizing loss combines baseline-centered ΔV waveform matching with an
explicit absolute pre-step voltage-offset term; both component weights are
configurable in the stage-1 section.
`runtime.hyperpolarizing_fitness_pre_ms` controls that scoring window without
changing the simulation duration. Depolarizing trials retain the full prepared
trial window. The checked-in hyperpolarizing configs use trace index `0` as the
representative trace; depolarizing trials still use all four configured traces.

## Environment

Use the existing `Jaxley` environment:

```bash
conda run --name Jaxley python -m pip install -e NEURON_optim
conda run --name Jaxley python -c "import neuron; print(neuron.__version__)"
```

The NEURON backend uses the repository's Combe morphology and compiled MOD
mechanisms. Check the NEURON import in the same environment:

```bash
conda run --name Jaxley python -c "import neuron; print(neuron.__version__)"
```

The Jaxley backend uses the HOC-derived compartment layout and the
`NEURON_optim` parameter-application rules when `runtime.backend` is set to
`jaxley`, so it can be compared directly with NEURON. The SWC morphology is
still available as an explicit approximate diagnostic via
`make_simulator("jaxley", morphology_source="swc")`.

`Combe2023.zip` is extracted automatically into `.cache/Combe2023` on the first
simulation if an extracted `Combe2023/` directory is not present.

## Validate and run

```bash
conda run --name Jaxley \
  combe-neuron-optim validate \
  --config NEURON_optim/configs/hyperpolarizing.yaml
```

The hyperpolarizing workflow begins with a passive-only pre-calibration. It
uses one representative raw current trace and records the pre-pulse and pulse
current medians, then fits only passive parameters with all active
conductances disabled. The subsequent full-model hyperpolarizing fits start
from the best passive solution, with deterministic ±15% perturbations applied
to the passive coordinates. The raw current waveform is preserved; the
holding current is not centered away.

To run only the replay check:

```bash
conda run --name Jaxley \
  combe-neuron-optim check-current \
  --config NEURON_optim/configs/hyperpolarizing.yaml \
  --output-dir NEURON_optim/runs/passive_check
```

To run the passive pre-calibration by itself:

```bash
conda run --name Jaxley \
  combe-neuron-optim run-passive \
  --config NEURON_optim/configs/hyperpolarizing.yaml \
  --output-dir NEURON_optim/runs/passive
```

For a different smoke or production budget, make a temporary copy of the YAML
and adjust `generations`, `population_size`, and `seeds`.

Run stage 1 and create the basins:

```bash
conda run --name Jaxley \
  combe-neuron-optim run-hyper \
  --config NEURON_optim/configs/hyperpolarizing.yaml \
  --output-dir NEURON_optim/runs/example
```

Run stage 2 from those basins:

```bash
conda run --name Jaxley \
  combe-neuron-optim run-depolarizing \
  --config NEURON_optim/configs/depolarizing.yaml \
  --output-dir NEURON_optim/runs/example
```

The checked-in configs use one candidate worker; `--workers` can override this
for process-parallel runs. Independent seeds/basins can additionally be
parallelized with `runtime.study_workers`; when that value is greater than one,
candidate workers are reduced to one inside each study to avoid
oversubscription. Backend state is kept inside each worker process. A
generation advances only after all offspring have returned finite losses and
the checkpoint has been written.

## Outputs

Stage 1 writes one directory per seed, a checkpoint and generation history, and
`basins.jsonl` containing selected candidates from the complete Stage 1
history. With the checked-in Stage 1 budget, the basin cap produces up to 100
basins. Each
generation also writes `population_generation_XXXX.npz` and
`simulations_generation_XXXX.npz`, containing the evaluated candidate
parameters/losses and the corresponding per-trace simulation arrays. Plotting
uses those saved evaluation results; it does not rerun simulations for the
selected candidates. Both stages write
`plots/generation_XXXX/candidates.png` for the ten lowest-loss candidates after
every generation. The single figure uses one row per candidate and one panel
per trace, showing measured and simulated voltages together.
Plotting is diagnostic and a figure-rendering failure does not discard a
checkpointed generation. Set `plotting.every` to a value greater than one to
render every Nth generation (the final generation is always rendered when
plotting is enabled). Stage 2 writes one directory per basin and seed with
its perturbed initial point, checkpoint, generation history, simulation
archives, plots, and final objective breakdown. All active-current parameters
are optimized in these Stage 2 studies; there is no subsequent Stage 3
optimization. Set `plotting.enabled: false`
when running a diagnostic search without figures.

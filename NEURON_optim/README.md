# Combe two-stage optimization

This package implements the reviewed two-stage CMA-ES plan using the Combe2023
model. The checked-in configurations select the repository's Jaxley backend;
the original NEURON backend remains available with `runtime.backend: neuron`.

The checked-in hyperpolarizing settings use 3 seeds, 100 generations, 20
offspring, and 1 process worker for the passive and stage-1 searches. Stage 2
retains 10 seeds, 200 generations, and 30 offspring; its study count depends
on the number of basins produced by stage 1.

Trace windows are capped to 500 ms before the current step (or time zero) and
600 ms after it ends, subject to the available recorded data.

## Environment

Use the existing `Jaxley` environment:

```bash
conda run --name Jaxley python -m pip install -e NEURON_optim
conda run --name Jaxley python -c "import jaxley; print(jaxley.__version__)"
```

The Jaxley backend uses the repository's SWC morphology and does not require
NEURON or compiled MOD mechanisms. If `runtime.backend` is changed to
`neuron`, compile the mechanisms and check the NEURON import in the same
environment:

```bash
conda run --name Jaxley python -c "import neuron; print(neuron.__version__)"
```

`Combe2023.zip` is extracted automatically into `.cache/Combe2023` on the first
simulation if an extracted `Combe2023/` directory is not present.

## Validate and run

```bash
conda run --name Jaxley \
  combe-neuron-optim validate \
  --config NEURON_optim/configs/hyperpolarizing.yaml
```

The hyperpolarizing workflow begins with a passive-only pre-calibration. It
first replays one raw current trace and records the pre-pulse and pulse
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

The checked-in configs use one worker; `--workers` can override this for
process-parallel runs. Backend state is kept inside each worker process. A
generation advances only after all offspring have returned finite losses and
the checkpoint has been written.

## Outputs

Stage 1 writes one directory per seed, a checkpoint and generation history, and
`basins.jsonl` containing the best final-generation candidates. With the
checked-in 3 × 20 stage-1 budget, this produces up to 60 basins. Both stages
also write `plots/generation_XXXX/rank_XX.png` for the ten
lowest-loss candidates after every generation, showing measured and simulated
voltages for all four traces. Stage 2 writes one directory per basin and seed
with its perturbed initial point, checkpoint, generation history, plots, and
final objective breakdown. Set `plotting.enabled: false` when running a
diagnostic search without figures.

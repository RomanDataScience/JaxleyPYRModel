# NEURON Combe two-stage optimization

This package implements the reviewed two-stage CMA-ES plan using the original
Combe2023 NEURON model. It runs in the existing Jaxley environment; it does not
replace the NEURON simulation with Jaxley.

The default production settings are 10 seeds, 200 generations, 30 offspring,
and 6 process workers. The full stage-2 design contains 1,000 independent
studies, so use the smoke overrides before launching production.

## Environment

Use the existing `Jaxley` environment:

```bash
conda run --name Jaxley python -m pip install -e NEURON_optim
conda run --name Jaxley nrnivmodl channels_converted/mod
```

If the environment already has the compiled mechanisms in
`channels_converted/mod/arm64` or another platform directory, the second
command is only needed after changing MOD sources. Check that NEURON imports in
the same environment:

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

For a small smoke run, make a temporary copy of the YAML and reduce both
`generations` and `population_size`. Production configs intentionally retain
the requested 200/30 budget.

Run stage 1 and create the 100 basins:

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

Use `--workers 1` for deterministic debugging. The default is six process
workers; NEURON state is never shared between worker processes. A generation
advances only after all offspring have returned finite losses and the checkpoint
has been written.

## Outputs

Stage 1 writes one directory per seed, a checkpoint and generation history, and
`basins.jsonl` containing the ten best final-generation candidates from each
seed. Both stages also write `plots/generation_XXXX/rank_XX.png` for the ten
lowest-loss candidates after every generation, showing measured and simulated
voltages for all four traces. Stage 2 writes one directory per basin and seed
with its perturbed initial point, checkpoint, generation history, plots, and
final objective breakdown. Set `plotting.enabled: false` when running a
diagnostic search without figures.

# Jaxley refactored

This is an executable, configuration-driven refactor of the Combe2023 cell
model. It keeps the validated HOC-compatible compartment-property behavior
while separating morphology, mechanisms, spatial profiles, parameters,
experimental data, simulation, optimization, and compute policy.

The default configuration fits one shared parameter vector to all eight
segmented current-clamp records for `m20240527cd`:

- four 24,000-sample depolarizing-step records;
- four 13,000-sample hyperpolarizing-pulse records;
- protocol-balanced weights, so each protocol contributes 50% of the loss.

The simulation backbone follows Jaxley's documented
[`jit(vmap(...))` approach](https://jaxley.readthedocs.io/en/latest/tutorials/04_jit_and_vmap.html).
Traces are grouped into natural static shapes, initialized independently, and
simulated through jitted `vmap` kernels. Every bucket gradient is summed before
one update to the parameter vector shared by all recordings.

## Repository layout

```text
Jaxley_refactored/
├── jaxley_refactored/   # model, data, simulation, fitting, and reporting code
├── configs/             # model, runtime, dataset, and sweep configurations
├── scripts/             # local fitting launchers
├── slurm/               # cluster launchers
├── tests/               # unit and integration tests
└── pyproject.toml
```

## Quick start

Use the existing Jaxley environment or install the package in editable mode:

```bash
cd Jaxley_refactored
python -m pip install -e .
```

The console entry point configures JAX before importing it. Use that entry point
or the bootstrap module, not `cli.commands` directly:

```bash
jaxley-refactored validate-config \
  --config configs/runtimes/cpu_x64.yaml

jaxley-refactored inspect-data \
  --config configs/runtimes/cpu_x64.yaml

jaxley-refactored inspect-model \
  --config configs/runtimes/cpu_x64.yaml

jaxley-refactored simulate \
  --config configs/runtimes/cpu_x64.yaml \
  --trace v75ctrl --protocol depolarizing_step --max-steps 100

jaxley-refactored fit \
  --config configs/runtimes/cpu_x64.yaml
```

Run `fit --dry-run` to build the model, load and hash every input, and write a
run manifest without compiling the optimizer.

The existing `fit` command remains the standard local optimizer. The additive
hybrid pipeline runs bounded CMA-ES, fixed-step Adam exploration, backtracking
Adam refinement, and validation on the second/fourth traces:

```bash
bash scripts/run_hybrid_fitting.sh \
  --config configs/search/LSU_1_cma_adam.yaml \
  --cell m20260331b \
  --seed 1234
```

The multi-cell launcher discovers valid cell IDs from the trace manifest:

```bash
# Show the available selectors.
bash scripts/run_hybrid_cells.sh --list-cells

# Run one cell.
bash scripts/run_hybrid_cells.sh --cells m20260331b --seed 1234

# Run selected cells, sequentially.
bash scripts/run_hybrid_cells.sh \
  --cells m20240527cd,m20260331b \
  --seed 1234

# Run every manifest cell. Use parallelism only when CPU RAM permits it.
bash scripts/run_hybrid_cells.sh --cells all --max-parallel-cells 2
```

The default is `--cells all --max-parallel-cells 1`, so complete hybrid
pipelines run one at a time and do not compete for memory.

Use `configs/search/LSU_1_cma_adam_smoke.yaml` only for short end-to-end
correctness checks; it is not a scientific optimization budget.

Hybrid reporting defaults to:

- the best training candidate after every CMA generation;
- each Adam candidate every 10 epochs, with a stage-local `latest.png`;
- training and validation plots for every final candidate;
- a selected-model validation plot.

Configure these under `search.reporting`. Experimental traces use `alpha=0.6`.

### Fit every recorded cell locally

The local launcher discovers cell IDs from `segment_metadata.csv` and performs
one complete fit per cell:

```bash
scripts/run_full_fitting.sh
```

Within each cell, traces are grouped by `(dt_ms, n_steps)` and simulated with
jitted `vmap` kernels. All bucket losses and gradients are accumulated before
one optimizer update, so all eight recordings share exactly one parameter
vector without wasting integration steps on padding. Cell fits run sequentially
by default because each process can use substantial GPU memory or CPU RAM. A
machine with enough memory can also run independent cells concurrently:

```bash
scripts/run_full_fitting.sh --max-parallel-cells 2
```

Useful first checks are:

```bash
# Build and validate every cell without optimization.
scripts/run_full_fitting.sh --dry-run

# Run one epoch for one cell.
scripts/run_full_fitting.sh --cells m20240527cd --epochs 1

# Fast end-to-end gradient, logging, checkpoint, and plotting test.
scripts/run_full_fitting.sh \
  --cells m20240527cd --epochs 1 --max-steps 100
```

Set `PYTHON_EXECUTABLE=/path/to/python` if the desired environment is not the
shell's default Python. Epoch metrics are printed live and also written to
per-cell logs under `runs/launcher_logs/<timestamp>/`. After every epoch, an
eight-panel simulated-versus-experimental figure is saved under the run's
`plots/` directory; `plots/latest.png` always points to the newest result.

To use loss-decreasing adaptive Adam steps:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/optimizers/adam_backtracking.yaml \
  --cells m20240527cd
```

The log prints the tested learning rate, whether the step was accepted, and the
number of forward trials. See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

## Configuration knobs

The main executable configuration is
`configs/fits/combe_m20240527cd_all.yaml`. Smaller runtime files inherit from it.
For a complete explanation of LSU_1—including hyperpolarizing waveform MSE,
depolarizing firing rate, interspike-minimum voltage, spike and recovery
metrics, and the outside-step spike penalty—see
[`configs/losses/README_LSU_1.md`](configs/losses/README_LSU_1.md).

For a separate hyperpolarization-only fit using point-by-point voltage MSE:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/losses/hyperpolarizing_only.yaml \
  --cells m20240527cd \
  --seed 1234
```

This configuration writes fitted runs to `runs_hyper/` instead of `runs/`.

To run the full hyperpolarization-only CMA-ES → Adam hybrid pipeline:

```bash
./scripts/run_hybrid_cells.sh \
  --config configs/search/hyperpolarizing_only_cma_adam.yaml \
  --cells m20240527cd \
  --seed 1234
```

- `model.morphology.provider`: `hoc_live`, `hoc_artifact`, or `swc`.
- `model.morphology.path`: artifact directory or any compatible SWC file.
- `model.morphology.discretization`: d-lambda and frequency.
- `model.mechanisms.include/exclude`: statically enable a valid channel subset.
- `model.distributions.overrides`: change spatial-profile coefficients.
- `model.parameters.fit`: select fitted values by tags or names.
- `dataset.selection`: choose cells, traces, and segment types.
- `dataset.simulation_window`: optionally end simulations a fixed time after
  stimulus offset, with `post_stimulus_ms_by_protocol` overrides when protocols
  need different horizons. The loader fails explicitly if a recording does not
  contain the requested interval.
- `fit.objective`: choose trace/protocol aggregation and weights.
- `fit.batching.strategy`: `vmap` for throughput or `serial` as a lower-memory
  reference path.
- `runtime`: CPU/GPU, precision, JIT, solver, rematerialization, and memory.

A distribution override uses the same canonical 44-parameter catalog as
fitting: the original 28 conductance and 12 passive parameters, followed by
four shared kinetic time scales. If the coefficient is selected for fitting,
the override changes its initial value. If it is excluded, the override remains
fixed during fitting. This keeps one implementation of every Combe profile.
See [docs/KINETIC_PARAMETERS.md](docs/KINETIC_PARAMETERS.md) for the kinetic
equations, regional targets, bounds, and interpretation caveats.

Changing morphology, discretization, enabled mechanisms, or profile family is
a static change and causes a new model signature/JAX compilation. Parameter
values, stimuli, masks, observations, and initial voltages remain dynamic.

## Exact HOC compatibility and GPUs

`hoc_live` is the CPU reference mode. It builds the original HOC morphology,
preserves the final NEURON segment grid, and uses reference-delta updates for
compartment properties. Reapplying defaults is an identity operation.

GPU nodes should use a portable HOC artifact:

```bash
jaxley-refactored export-hoc-artifact \
  --config configs/runtimes/cpu_x64.yaml \
  --destination assets/morphologies/combe_pc2b_dlambda_0p3

RUN_SCRATCH=/tmp/jaxley \
jaxley-refactored inspect-model \
  --config configs/runtimes/gpu_x64.yaml
```

The artifact is JSON plus compressed NumPy arrays—never pickle—and includes a
checksum, schema version, source provenance, topology, groups, geometry, and
all numeric node properties. It can be loaded without NEURON on a GPU node.

The GPU runtime requests float64 for first-line scientific parity. Use float32
only after comparing losses, voltages, and gradients for the intended model.

## SLURM

One fit uses one GPU. Independent seeds, morphologies, channel sets, or folds
scale across a SLURM array:

```bash
cd Jaxley_refactored
sbatch --array=0-3 slurm/fit_array.sbatch \
  configs/sweeps/combe_multistart_manifest.tsv
```

Set `PYTHON_EXECUTABLE` when the cluster environment does not expose the correct
Python as `python`. Edit the resource directives for the cluster. Each TSV row
contains `config`, `seed`, and `run_name`; the CLI overrides are included in the
resolved run configuration and compatibility hash.

Generate the portable artifact on a CPU partition with:

```bash
sbatch slurm/export_hoc_artifact.sbatch \
  configs/runtimes/cpu_x64.yaml \
  assets/morphologies/combe_pc2b_dlambda_0p3
```

The trainer writes atomic latest/best checkpoints and handles SLURM's `USR1`
preemption warning by checkpointing and returning exit code 75.

## Documentation and validation

- `docs/ARCHITECTURE.md`: module responsibilities and extension points.
- `docs/LOSS_LIBRARY.md`: registered losses, windows, scaling, and composition.
- `COMPARTMENT_PROPERTY_COMPATIBILITY.md`: exact HOC update semantics.
- `REFACTORING_PLAN.md`: audit, decisions, and migration history.
- `CONFIG_BLUEPRINT.yaml`: broader schema roadmap; executable configs live
  under `configs/`.

Run the fast suite with:

```bash
pytest
```

The real-model smoke checks are intentionally explicit because they compile a
974-compartment cell:

```bash
jaxley-refactored simulate \
  --config configs/runtimes/cpu_x64.yaml --max-steps 11
```

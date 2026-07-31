# Jaxley refactored

This is an executable, configuration-driven refactor of the Combe2023 cell
model. It keeps the validated HOC-compatible compartment-property behavior
while separating morphology, mechanisms, spatial profiles, parameters,
experimental data, simulation, optimization, and compute policy.

The configuration directory contains exactly two runnable YAML files. Both are
standalone: each records the complete model, data, simulation, loss, optimizer,
runtime, search, and output policy without YAML inheritance.

- `configs/LSU_1_cma_adam.yaml` trains on the first and third depolarizing and
  hyperpolarizing traces and validates on the second and fourth traces.
- `configs/hyperpolarizing_only_cma_adam.yaml` uses the same split but selects
  only hyperpolarizing traces and writes results under `runs_hyper/`.

The simulation backbone follows Jaxley's documented
[`jit(vmap(...))` approach](https://jaxley.readthedocs.io/en/latest/tutorials/04_jit_and_vmap.html).
Traces are grouped into natural static shapes, initialized independently, and
simulated through jitted `vmap` kernels. Every bucket gradient is summed before
one update to the parameter vector shared by all selected recordings.

## Repository layout

```text
Jaxley_refactored/
├── jaxley_refactored/   # model, data, simulation, fitting, and reporting code
├── configs/             # two complete standalone run configurations
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
  --config configs/LSU_1_cma_adam.yaml

jaxley-refactored inspect-data \
  --config configs/LSU_1_cma_adam.yaml

jaxley-refactored inspect-model \
  --config configs/LSU_1_cma_adam.yaml

jaxley-refactored simulate \
  --config configs/LSU_1_cma_adam.yaml \
  --trace v75ctrl --protocol depolarizing_step --max-steps 100

jaxley-refactored fit \
  --config configs/LSU_1_cma_adam.yaml
```

Run `fit --dry-run` to build the model, load and hash every input, and write a
run manifest without compiling the optimizer.

The existing `fit` command remains the standard local optimizer. The additive
hybrid pipeline runs bounded CMA-ES, fixed-step Adam exploration, backtracking
Adam refinement, and validation on the second/fourth traces.

The complete full-protocol hybrid configuration is contained in
`configs/LSU_1_cma_adam.yaml`; it has no inherited YAML dependencies.
Training traces `[1, 3]` and held-out validation traces `[2, 4]` are both
declared under `dataset.selection`.

Each 40-candidate CMA generation recombines its best 12 candidates
(`parent_fraction: 0.30`). After the global stage, the best 10 candidates
overall continue into Adam.

```bash
bash scripts/run_hybrid_fitting.sh \
  --config configs/LSU_1_cma_adam.yaml \
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
one optimizer update, so all selected recordings share exactly one parameter
vector without wasting integration steps on padding. Cell fits run sequentially
by default because each process can use substantial CPU RAM. A
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
per-cell logs under `runs/launcher_logs/<timestamp>/`. After every epoch, a
simulated-versus-experimental panel for each selected trace is saved under the
run's `plots/` directory; `plots/latest.png` always points to the newest result.

To use loss-decreasing adaptive Adam steps:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/LSU_1_cma_adam.yaml \
  --cells m20240527cd
```

The log prints the tested learning rate, whether the step was accepted, and the
number of forward trials. See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

## Configuration knobs

The configuration directory intentionally ships only these two standalone
files:

- `configs/LSU_1_cma_adam.yaml`: full depolarizing and hyperpolarizing LSU_1
  hybrid calibration;
- `configs/hyperpolarizing_only_cma_adam.yaml`: hyperpolarizing-only hybrid
  calibration with trough depth as its primary feature.

Both can also be passed to the ordinary `fit` command, which uses the base Adam
section and ignores the additive hybrid stages.

For a complete explanation of LSU_1—including hyperpolarizing waveform MSE,
depolarizing firing rate, interspike-minimum voltage, spike and recovery
metrics, and the outside-step spike penalty—see
[`docs/LSU_1.md`](docs/LSU_1.md).

For a separate hyperpolarization-only fit using dominant trough-depth
matching, point-by-point voltage MSE, and first-derivative MSE:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/hyperpolarizing_only_cma_adam.yaml \
  --cells m20240527cd \
  --seed 1234
```

This configuration writes fitted runs to `runs_hyper/` instead of `runs/`.

To run the full hyperpolarization-only CMA-ES → Adam hybrid pipeline:

```bash
./scripts/run_hybrid_cells.sh \
  --config configs/hyperpolarizing_only_cma_adam.yaml \
  --cells m20240527cd \
  --seed 1234
```

- `model.morphology.provider`: `hoc_live`, `hoc_artifact`, or `swc`.
- `model.morphology.path`: artifact directory or any compatible SWC file.
- `model.morphology.discretization`: d-lambda and frequency.
- `model.mechanisms.include/exclude`: statically enable a valid channel subset.
- `model.distributions.overrides`: change spatial-profile coefficients.
- `model.parameters.fit`: select fitted values by tags or names.
- `dataset.selection`: choose training traces, held-out hybrid validation
  traces, and segment types.
- `dataset.simulation_window`: optionally end simulations a fixed time after
  stimulus offset, with `post_stimulus_ms_by_protocol` overrides when protocols
  need different horizons. The loader fails explicitly if a recording does not
  contain the requested interval.
- `fit.objective`: choose trace/protocol aggregation and weights.
- `fit.batching.strategy`: `vmap` for throughput or `serial` as a lower-memory
  reference path.
- `runtime`: backend, precision, JIT, solver, rematerialization, and memory. The
  two shipped configurations select CPU and float64.

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

## Exact HOC compatibility and portability

`hoc_live` is the CPU reference mode. It builds the original HOC morphology,
preserves the final NEURON segment grid, and uses reference-delta updates for
compartment properties. Reapplying defaults is an identity operation.

The portable HOC-artifact exporter can serialize the exact morphology and
compartment properties:

```bash
jaxley-refactored export-hoc-artifact \
  --config configs/LSU_1_cma_adam.yaml \
  --destination assets/morphologies/combe_pc2b_dlambda_0p3
```

The artifact is JSON plus compressed NumPy arrays—never pickle—and includes a
checksum, schema version, source provenance, topology, groups, geometry, and
all numeric node properties. The two shipped configurations use `hoc_live` on
CPU; no separate artifact or GPU runtime preset is shipped.

## SLURM

The shipped hybrid SLURM entry point runs the serial CPU population baseline.
Select the cell and seed through environment variables:

```bash
cd Jaxley_refactored
CELL_ID=m20260331b SEED=1234 sbatch slurm/hybrid_lsu1_cpu.sbatch
```

Set `PYTHON_EXECUTABLE` when the cluster environment does not expose the correct
Python as `python`. Edit the resource directives for the cluster. CLI cell,
seed, run-name, and maximum-step overrides are included in run identity and
provenance where applicable.

The trainer writes atomic latest/best checkpoints and handles SLURM's `USR1`
preemption warning by checkpointing and returning exit code 75.

## Documentation and validation

- `docs/ARCHITECTURE.md`: module responsibilities and extension points.
- `docs/LOSS_LIBRARY.md`: registered losses, windows, scaling, and composition.
- `COMPARTMENT_PROPERTY_COMPATIBILITY.md`: exact HOC update semantics.
- `REFACTORING_PLAN.md`: audit, decisions, and migration history.
- `CONFIG_BLUEPRINT.yaml`: historical schema roadmap; the two executable
  standalone configs live under `configs/`.

Run the fast suite with:

```bash
pytest
```

The real-model smoke checks are intentionally explicit because they compile a
974-compartment cell:

```bash
jaxley-refactored simulate \
  --config configs/LSU_1_cma_adam.yaml --max-steps 11
```

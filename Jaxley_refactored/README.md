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
Traces with the same `(dt, n_steps)` are stacked into static-shape buckets,
initialized independently, and simulated in parallel.

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

## Configuration knobs

The main executable configuration is
`configs/fits/combe_m20240527cd_all.yaml`. Smaller runtime files inherit from it.

- `model.morphology.provider`: `hoc_live`, `hoc_artifact`, or `swc`.
- `model.morphology.path`: artifact directory or any compatible SWC file.
- `model.morphology.discretization`: d-lambda and frequency.
- `model.mechanisms.include/exclude`: statically enable a valid channel subset.
- `model.distributions.overrides`: change spatial-profile coefficients.
- `model.parameters.fit`: select fitted values by tags or names.
- `dataset.selection`: choose cells, traces, and segment types.
- `fit.objective`: choose trace/protocol aggregation and weights.
- `fit.batching.strategy`: `vmap` for throughput or `serial` as a lower-memory
  reference path.
- `runtime`: CPU/GPU, precision, JIT, solver, rematerialization, and memory.

A distribution override uses the same canonical 40-parameter catalog as
fitting. If the coefficient is selected for fitting, the override changes its
initial value. If it is excluded, the override remains fixed during fitting.
This keeps one implementation of every Combe profile.

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

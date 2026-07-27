# Jaxley refactoring plan

Status: the package refactor remains a plan and does not yet replace
`JaxleyModel/model/model_Combe.py`. The critical HOC-compatible compartment
update fix has been implemented in the current model as a safe precursor.

The target is a configuration-driven Jaxley package in which morphology,
mechanism selection, spatial channel distributions, fitted parameters, data
selection, and compute policy are independent knobs. The default fit will use
every segmented trace for cell `m20240527cd`.

Start with:

- [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for the audited design, migration
  sequence, test gates, GPU strategy, and SLURM strategy.
- [CONFIG_BLUEPRINT.yaml](CONFIG_BLUEPRINT.yaml) for the proposed user-facing
  configuration contract. It is illustrative and is not executable yet.
- [COMPARTMENT_PROPERTY_COMPATIBILITY.md](COMPARTMENT_PROPERTY_COMPATIBILITY.md)
  for the implemented HOC frozen-grid update semantics, validation results, and
  the boundary between differentiable fitting and CPU-only rediscretization.

## Intended end state

The implemented package should support workflows resembling:

```bash
jaxley-refactored validate-config \
  --config configs/fits/combe_m20240527cd_all.yaml

jaxley-refactored inspect-model \
  --config configs/fits/combe_m20240527cd_all.yaml

jaxley-refactored fit \
  --config configs/fits/combe_m20240527cd_all.yaml

sbatch slurm/fit_array.sbatch \
  configs/sweeps/combe_multistart_manifest.tsv
```

A user should be able to change the following without editing Python:

- HOC-reference, SWC, or another registered morphology provider.
- Morphology file, group mapping, root, and discretization.
- Enabled channels and their anatomical placement.
- Channel and passive-property distribution profiles.
- Parameter defaults, bounds, transforms, fit tags, and explicit exclusions.
- Cell, traces, segment types, loss windows, and trace/protocol weights.
- CPU/GPU backend, precision, JIT, trace microbatching, and solver
  rematerialization.
- Optimizer, seed, checkpoint cadence, output location, and SLURM sweep axis.

## Decisions already made

1. Exact HOC reproduction and reuse on a new morphology are separate scientific
   modes. They will never be selected implicitly by comparing parameter values.
2. Applying no parameter updates, or reapplying defaults, must leave the built
   baseline unchanged.
3. Morphology topology, compartment count, enabled channels, placement, and
   distribution kind are static choices that rebuild and recompile the model.
   Fitted values, stimuli, observations, masks, weights, and initial voltage are
   dynamic JAX inputs.
4. Parallel trace execution will follow Jaxley's documented
   [`data_set` + `data_stimulate` + `jit(vmap(...))` pattern](https://jaxley.readthedocs.io/en/latest/tutorials/04_jit_and_vmap.html).
5. The default `m20240527cd` dataset forms two compiled batches: four
   depolarizing traces with 24,000 samples and four hyperpolarizing traces with
   13,000 samples.
6. One fit runs on one GPU first. SLURM arrays parallelize independent seeds,
   model configurations, morphologies, and folds. Multi-GPU optimization of one
   shared fit is a later, measured extension.
7. CPU float64 remains the scientific reference. GPU float64 is the first GPU
   target; float32 is enabled only after stability and parity tests.

# Architecture and extension guide

The package follows SOLID by keeping each module responsible for one axis of
change and putting replaceable behavior behind small interfaces.

```text
YAML -> AppConfig -> ModelBuilder -> BuiltModel
                       |    |    |
            MorphologyProvider  MechanismRegistry  ParameterCatalog
                       |
DatasetProvider -> TraceBucket -> SimulationKernel -> Loss -> Optimizer
                                      |
                                jit(vmap(trace))
```

## Dependency direction

`cli` and `fitting` orchestrate use cases. They depend on stable records and
protocols in `config`, `models`, `data`, `simulation`, and `parameters`.
Model-specific legacy calls point inward through
`compatibility.LegacyCombeBackend`; no other layer imports `model_Combe.py`.

| Package | Owns | Extend by |
| --- | --- | --- |
| `config` | Frozen validated value objects, YAML inheritance, path resolution | Add a value object and validate at the boundary |
| `morphology` | Live HOC, portable artifact, and SWC construction | Implement `MorphologyProvider` and register it |
| `mechanisms` | Channel inventory, dependencies, static selection | Add `MechanismInfo` or a new registry factory |
| `parameters` | Metadata, bounds, transforms, JAX parameter state | Add catalog entries or another `ParameterStateBackend` |
| `data` | Immutable traces, validation, resampling, weights, buckets | Implement another loader returning `TraceRecord` |
| `simulation` | Sites, trace-specific states, JIT/vmap kernel | Add a kernel without changing data or optimization |
| `fitting` | Loss primitives, projected Adam, checkpoints, trainer | Inject another loss/optimizer implementation |
| `runtime` | Pre-import JAX environment and device validation | Add backend policy without touching model code |
| `reporting` | Atomic run outputs and provenance | Add reporters around immutable metrics |

## Static and dynamic boundaries

JAX compilation works best when topology and Python structure are static.
Morphology, compartment count, mechanism set, profile family, recording sites,
solver, time step, and trace length therefore define an executable variant.

The fitted parameter vector, fixed profile coefficients, input current,
initial-state pytree, observations, masks, and weights are arrays passed through
the numerical boundary. The parameter key tuple is static; values are dynamic.

Records are bucketed by `(dt_ms, n_steps)`. One `SimulationKernel` is compiled
per natural shape. `jax.vmap` maps a pure single-trace simulation over currents
and initial states in each bucket. Bucket losses and gradients are summed before
one optimizer step, so every recording shares exactly one parameter vector.

## HOC frozen-grid semantics

The exact-HOC mode stores both final Jaxley compartment centers and the original
HOC segment-assignment distance. During fitting, passive and active fields use:

```text
HOC reference field + rule(theta) - rule(default theta)
```

Only fields dependent on selected parameters are emitted into Jaxley's
`param_state`. This makes default updates an exact identity and prevents
unselected parameters from overwriting HOC values. Topology and `ncomp` remain
frozen during JIT; changing d-lambda rebuilds the static model.

The SWC mode intentionally uses rule-based values on final compartment centers.
It is a reusable scientific model, not a claim of exact HOC segment parity.

## Adding a morphology

1. Implement `MorphologyProvider.build(ModelSpec) -> MorphologyResult`.
2. Produce the required anatomical group columns.
3. Extract immutable `StaticFeatures`.
4. Include every topology-affecting input in the provider fingerprint.
5. Register the provider in `models.default_builder`.
6. Add a config and parity/validation test.

## Adding or removing mechanisms

Mechanisms are chosen before compilation. Dependencies are checked first; for
example calcium-dependent currents require `d3` and `cal4`. The builder then
checks that every selected fitted or fixed parameter still has a live target.
This turns incompatible channel/parameter configurations into early errors.

## Runs and restart safety

A compatibility hash covers the resolved configuration, model signature, and
input file hashes. Checkpoints with a different hash are rejected. Run outputs
include initial/best/final parameter CSV files, metrics, resolved configuration,
device/environment provenance, and atomic latest/best checkpoints.

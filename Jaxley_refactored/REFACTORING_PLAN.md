# Refactoring plan: modular Jaxley models, all-trace fitting, GPU, and SLURM

Date: 2026-07-27

Status: implementation plan; no production code has been moved yet

## 1. Outcome

Refactor the current Combe implementation into an installable,
configuration-driven package that can:

1. Build the Combe model, or another registered model, from interchangeable
   morphology providers.
2. Enable, disable, or replace channel mechanisms without editing the model
   builder.
3. Configure channel placement and spatial distributions independently.
4. Fit a shared parameter set against every selected segment from one cell,
   defaulting to all eight segments under
   `JaxleyModel/Experimental_currentClamp_Analysis/Segmented_Traces/m20240527cd`.
5. Use Jaxley's documented `data_set`, `data_stimulate`, `jit`, and `vmap`
   workflow as the single-GPU parallel execution backbone.
6. Run reproducibly on a workstation or as collision-free, resumable SLURM
   jobs.

The refactor must preserve a scientifically meaningful reference. It must not
silently treat the HOC-derived model and an SWC-based rule-driven model as the
same morphology.

## 2. Scope and non-goals

### In scope

- `JaxleyModel/model/model_Combe.py` and `fit_model_Combe.py`.
- The converted Combe channel registry and placement rules.
- HOC-reference and SWC morphology construction.
- Segmented current-clamp loading, validation, batching, and multi-trace loss.
- Differentiable parameter application.
- Runtime/device configuration, JIT compilation, `vmap`, microbatching, and
  solver rematerialization.
- Persistent checkpoints, run provenance, GPU execution, and SLURM launchers.
- Thin compatibility adapters for current imports and command-line usage.

### Not in the first production milestone

- Rewriting the translated channel equations for biological changes.
- Differentiably adding/removing channels or changing morphology topology
  during an optimizer step. These are discrete outer-loop configuration choices.
- Optimizing raw SWC node coordinates.
- Multi-node synchronous fitting before a single GPU is correct and measured.
- Bitwise equality between CPU and GPU. CPU float64 is the reference; GPU uses
  calibrated numerical tolerances.

## 3. Current-state audit

### 3.1 `model_Combe.py` is a monolith

The 973-line module currently owns all of the following:

- Process-global JAX precision and CPU selection
  (`JaxleyModel/model/model_Combe.py:9-12`).
- Global warning policy, `sys.path` mutation, and a Jaxley monkeypatch
  (`:24-28`, `:63-67`).
- Forty-eight model values, two manually maintained fit groups, and bounds
  (`:70-215`).
- HOC and SWC morphology acquisition (`:218-432`, `:914-935`).
- Discretization, group masks, and distance features (`:445-481`).
- Channel registration and anatomical placement (`:267-309`, `:484-525`).
- Passive and active spatial distributions (`:528-699`).
- A second, duplicated JAX implementation of those distributions for fitting
  (`:702-894`).
- Model construction, calcium diffusion, and initial voltage (`:904-948`).
- A fixed somatic step protocol (`:951-970`).

That coupling means a channel choice, morphology path, distribution breakpoint,
device, or fitting policy requires source changes.

### 3.2 Correctness blocker: fitting changes the baseline before optimizing

The default HOC-backed build copies final NEURON per-segment passive and active
values at `model_Combe.py:941-943`. The fitting entry point then calls
`set_fitted_parameters()`, which recomputes *all* passive and active profiles
from analytic formulas at `:897-900`, whether or not a corresponding parameter
was selected.

A characterization run at `d_lambda=0.3` found that even:

```python
set_fitted_parameters(cell, [], [])
```

changed:

- 293 of 974 capacitance entries.
- 286 of 974 axial-resistivity entries.
- 338 of 974 leak-conductance entries.
- Several active channel distributions.

This conflicts with the repository's own finding that final HOC segment values,
not re-evaluated formulas, are required for exact reproduction
(`channels_converted/modelComparison/HOC_IMPORT_NOTES.md:58-112`).

The first refactoring invariant is therefore:

> Applying no updates, or applying the configured default parameter values,
> must reproduce the built baseline.

Every parameter must declare the exact target rules it can update. Unselected
parameters must remain untouched.

**Implemented precursor (2026-07-27).** The current `model_Combe.py` now
enforces this invariant before the package refactor:

- The HOC importer exports the original per-section assignment coordinate,
  `hoc_assignment_distance_um`.
- Empty updates return the incoming state unchanged.
- Only targets whose dependency sets intersect the selected keys are written.
- Exact-HOC updates use the immutable imported value plus the complete
  endpoint-rule difference from the reference parameters.
- SWC/rule-mode updates continue to use final compartment-center rules.
- Default values are bitwise-identical to the imported profile.

The regression suite and the remaining fixed-grid limitation are recorded in
`COMPARTMENT_PROPERTY_COMPATIBILITY.md`.

### 3.3 Exact HOC and SWC are different scientific modes

At `d_lambda=0.3`, the inspected structures are:

| Source | Branches | Compartments | Soma | Axon | Basal | Apical |
|---|---:|---:|---:|---:|---:|---:|
| HOC section tree | 144 | 974 | 5 | 7 | 373 | 589 |
| SWC | 140 | 950 | — | — | — | — |

The HOC notes explain why a standard SWC cannot preserve every zero-length
logical section connection (`HOC_IMPORT_NOTES.md:32-56`). The target architecture
must expose explicit modes:

- `exact_hoc_frozen_grid`: preserve final HOC topology and segment tables, then
  fit declared endpoint-rule deltas without changing array shapes.
- `rule_based_final_centers`: apply portable Combe distribution rules after
  discretizing a supplied morphology.
- `hoc_rebuild`: optionally rebuild through NEURON for CPU validation when
  passive parameters alter `nseg`; this mode is not differentiable or JIT-safe.

Mode selection must be configuration, never an equality comparison such as
`params == COMBE_PARAMS`.

### 3.4 Channel selection and distributions are hard-coded

Current placement is:

| Region | Inserted mechanisms |
|---|---|
| Whole cell | D3, Leak, Cal4 |
| Soma + apical | Icand, Nav16A, Kd, Kv2like, H, Kap, Km, Kca, MyKca |
| Soma | Nap, Cal, Cat, Car |
| Apical | Car, CalH, Cat, Kad, Kir |
| Axon | Nax, Kd, Km, Kap, Kv2like |
| Basal | Na3Dend, Nap, Kap, H, Kd, Kv2like, Kir |

This is embedded at `model_Combe.py:484-525`. Dependencies are implicit:

- Calcium conductances provide the shared calcium current.
- Cal4 consumes calcium current and owns `CaCon_i`.
- Kca and MyKca consume `CaCon_i`.
- Longitudinal calcium diffusion is optional and applies to `CaCon_i`.

Spatial rules contain hard-coded breakpoints and caps at 40, 50, 100, 126,
200, 300, 350, 394, and 500 micrometres. They exist twice, in NumPy and JAX
forms, which permits drift.

### 3.5 Current fitting is single-trace

`fit_model_Combe.py` accepts one `--trace-name` and one `--segment-name`
(`:55-108`). It:

- Loads only `v` and `i`, ignores `t_ms` and the manifest (`:115-137`).
- Assumes a caller-provided source `dt`.
- Uses floor-index resampling for both voltage and current.
- Silently truncates current and voltage to their shorter length (`:325-327`).
- Sets one observed initial voltage on one mutable cell (`:358-367`).
- Minimizes one masked voltage MSE (`:377-391`).
- Computes `v_final` for plotting, not for optimization (`:349-356`).
- Returns and copies a full prediction to the host every epoch (`:393-410`).
- Produces two plots per epoch by default (`:418-441`).
- Writes results only after training and cannot resume (`:460-531`).

The default selects all 40 bounded parameters: 28 conductance-related and 12
passive. Eight `CombeParameters` values are not selectable. Several selected
parameters are confounded, including `gna * scale_Na_conduct` and
`gkv2 * gkv2scale`.

Seven conductances start exactly at zero. The current bounded sigmoid path moves
them to tiny positive values near sigmoid saturation. The new catalog must
choose a boundary-safe transform explicitly; it must not silently modify exact
zero defaults.

### 3.6 Default all-trace dataset

The on-disk contract is:

```text
<root>/<cell>/<segment>/<trace>_<segment>_{v,i,t_ms}.txt
```

For `m20240527cd`:

| Protocol | Traces | Array shape | Duration | Step/pulse delta |
|---|---|---:|---:|---|
| `depolarizing_step` | v75ctrl-v78ctrl | 4 x 24,000 | 1,200 ms | +290.61, +387.48, +484.35, +581.22 pA |
| `hyperpolarizing_pulse` | v75ctrl-v78ctrl | 4 x 13,000 | 650 ms | -31.248 pA each |

All time vectors are uniform at 0.05 ms. All four hyperpolarizing records are
clipped at the end of the original recording. The manifest contains useful
epoch and clipping metadata, but its output paths are absolute and nonportable.

The current redetected loss windows include about 21,999 samples per
depolarizing trace and 4,971-4,998 samples per hyperpolarizing trace. A naive
concatenated sample MSE would therefore give the depolarizing protocol about
81.5% of the total weight.

### 3.7 GPU and clean-cluster blockers

- Both model and fit modules force CPU during import.
- XLA memory settings are applied after importing JAX.
- `/private/tmp` is hard-coded for plotting.
- The default HOC build invokes NEURON on every construction.
- `Combe2023/` and compiled `channels_converted/mod/` are ignored by Git.
  `Combe2023.zip` is tracked, while local compiled MOD artifacts are
  architecture-specific.
- `jaxley-models` is a gitlink, but there is no `.gitmodules` entry.
- There is no root environment definition, lock file, test suite, SLURM file,
  preemption checkpoint, or rank-safe output scheme.

The characterized local environment is Python 3.14.6, JAX/JAXlib 0.10.2,
Jaxley 0.13.0, jaxley-mech 0.3.1, NEURON 9.0.1, and NumPy 2.5.1. These values are
evidence for reproducing the baseline, not a decision to allow unbounded
dependency upgrades.

## 4. Design principles

1. **Configuration describes composition.** Python implements providers,
   registries, and rule primitives; users select them in validated YAML.
2. **Static structure and dynamic arrays are separate.** Structural changes
   rebuild and recompile. Numeric fitting inputs remain JAX values.
3. **One numerical rule implementation.** Spatial rules use `jax.numpy`; normal
   construction obtains host arrays with `np.asarray()` when needed.
4. **Reference values are immutable.** Exact HOC arrays are preserved, and
   fitting applies only declared deltas/scales.
5. **The inner simulation is pure.** No path access, pandas mutation, plotting,
   logging, or host conversion inside JIT/gradient functions.
6. **Data fails loudly.** Length, time, units, finiteness, metadata, and file
   triplets are validated; no silent truncation.
7. **Parallelism is layered.** `vmap` is the fine-grained single-GPU layer;
   SLURM arrays are the first cluster layer.
8. **Every run is reconstructable.** Resolved config, code, environment, inputs,
   model signature, device, and optimizer state are recorded.
9. **Migration is characterization-first.** The monolith remains available
   through adapters until the new path passes scientific and compute gates.

## 5. Static/dynamic boundary

JAX compilation is most predictable when the following boundary is explicit:

| Static; changes rebuild/recompile | Dynamic; remains an array input |
|---|---|
| Morphology topology/provider/artifact | Fitted parameter values |
| Compartment count and `d_lambda` | Current stimulus |
| Anatomical group mapping and root | Observed voltage |
| Enabled channel set | Scoring mask and trace weight |
| Channel placement selectors | Per-trace initial voltage/state |
| Distribution rule *kind* | Differentiable distribution coefficients |
| Injection/recording sites | Random key where required |
| Solver, precision, `dt`, time length | Optimizer state |
| Rematerialization layout | |
| Trace microbatch size | |

Morphology variants, channel subsets, or different distribution families belong
in separate configurations and usually separate SLURM array tasks. A fitted
conductance scale or sigmoid half-distance belongs in `theta`.

## 6. Target package layout

```text
Jaxley_refactored/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── models/
│   │   ├── combe2023_exact_hoc.yaml
│   │   └── combe2023_rule_based.yaml
│   ├── datasets/
│   │   └── m20240527cd.yaml
│   ├── fits/
│   │   └── combe_m20240527cd_all.yaml
│   ├── runtimes/
│   │   ├── cpu_x64.yaml
│   │   └── gpu_x64.yaml
│   └── sweeps/
├── assets/
│   └── morphologies/
├── src/jaxley_refactored/
│   ├── config/
│   │   ├── schema.py
│   │   ├── loading.py
│   │   └── hashing.py
│   ├── morphology/
│   │   ├── protocol.py
│   │   ├── hoc_live.py
│   │   ├── hoc_artifact.py
│   │   ├── swc.py
│   │   ├── grouping.py
│   │   ├── discretization.py
│   │   └── features.py
│   ├── mechanisms/
│   │   ├── registry.py
│   │   ├── dependencies.py
│   │   └── combe/
│   ├── distributions/
│   │   ├── protocol.py
│   │   ├── primitives.py
│   │   └── combe2023.py
│   ├── parameters/
│   │   ├── catalog.py
│   │   ├── transforms.py
│   │   └── parameterizer.py
│   ├── models/
│   │   ├── recipe.py
│   │   ├── builder.py
│   │   └── combe2023.py
│   ├── data/
│   │   ├── records.py
│   │   ├── segmented_traces.py
│   │   ├── windows.py
│   │   └── batching.py
│   ├── simulation/
│   │   ├── initialization.py
│   │   ├── kernel.py
│   │   ├── mapped.py
│   │   └── cache.py
│   ├── fitting/
│   │   ├── losses.py
│   │   ├── objective.py
│   │   ├── optimizer.py
│   │   ├── trainer.py
│   │   └── checkpoints.py
│   ├── runtime/
│   │   ├── bootstrap.py
│   │   ├── device.py
│   │   └── provenance.py
│   ├── reporting/
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── compatibility/
│   │   └── model_combe.py
│   └── cli/
│       ├── validate_config.py
│       ├── inspect_model.py
│       ├── export_hoc.py
│       ├── simulate.py
│       ├── fit.py
│       └── summarize.py
├── slurm/
│   ├── prepare_hoc_artifact.sbatch
│   ├── fit_array.sbatch
│   └── summarize_array.sbatch
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── gpu/
```

Use `pydantic` models to validate the configuration and PyYAML for input. Use
Optax for optimizers, consistent with Jaxley's official training tutorial.
Checkpoints should use a versioned JAX-pytree-capable format; Orbax is the
preferred implementation after a short compatibility spike. Never use
unversioned Python pickle as the only durable format.

## 7. Core interfaces

### 7.1 Morphology provider

```python
class MorphologyProvider(Protocol):
    def build(self, spec: MorphologySpec) -> MorphologyResult: ...
```

`MorphologyResult` contains:

- Initialized structural `jx.Cell`.
- Validated group masks.
- Path-distance and other static compartment features.
- Source and discretization provenance.
- A structural fingerprint.

Providers:

- `HocLiveProvider`: CPU/reference preprocessing only; may import NEURON.
- `HocArtifactProvider`: default exact-mode worker provider; must not import
  NEURON.
- `SwcProvider`: accepts a caller path and explicit group classifier.
- A registry hook for future morphology formats.

Group classification must fail on unknown sections when
`reject_unclassified=true`. The current behavior of treating every unknown HOC
prefix as apical is not acceptable for new morphologies.

### 7.2 Portable HOC artifact

Export the final HOC model once on a prepared CPU node into:

```text
assets/morphologies/<artifact_id>/
  manifest.json
  arrays.npz
```

The non-pickle, schema-versioned artifact contains:

- Parent array, per-branch `ncomp`, and ragged `xyzr`.
- Radius, length, area, volume, and resistive loads.
- Group labels and root information.
- Final passive values, ion reversals, temperature, and mechanism parameters.
- HOC-to-Jaxley coverage report.
- Source archive, HOC, MOD, registry, NEURON, `d_lambda`, and exporter hashes.

The artifact ID includes the source checksum and discretization, because final
HOC RANGE values can change with `nseg`. Add an explicit `.gitignore` exception
if the canonical artifact is versioned in Git.

### 7.3 Mechanism registry

One registry replaces both `HOC_CHANNEL_CLASSES` and the separate validation
registry:

```python
@dataclass(frozen=True)
class MechanismSpec:
    key: str
    factory: Callable
    instance_name: str
    aliases: tuple[str, ...]
    requires_states: tuple[str, ...]
    provides_states: tuple[str, ...]
    current_name: str | None
    hoc_mechanism: str | None
    validation_status: str
```

The same registry drives:

- Model insertion.
- HOC parameter mapping.
- Dependency checks.
- CLI listing.
- Isolated channel tests.
- Documentation.

Enabling/disabling a channel is a static model choice. The builder validates
state dependencies before initialization. A disabled conductance should remove
the mechanism when requested, not merely set `gbar=0`, because mechanism states
and compute cost otherwise remain.

### 7.4 Placement and distribution rules

```python
@dataclass(frozen=True)
class PlacementSpec:
    mechanism: str
    selector: str
    enabled: bool = True

class DistributionRule(Protocol):
    def evaluate(self, features, parameters) -> jax.Array: ...
```

Reusable JAX primitives include:

- Constant.
- Parameter product.
- Linear and capped-linear ramps.
- Sigmoid.
- Threshold and piecewise selection.
- Region mask.
- Reference scale/additive delta.

No arbitrary `eval()` expressions are accepted from YAML. New behavior is added
by registering a typed rule in Python. Rule configuration supplies inspectable
breakpoints, caps, parameter names, units, and targets.

The Combe recipe composes these primitives. There is only one JAX evaluator;
build-time assignment converts its result to NumPy if the Jaxley setup API
requires host data.

### 7.5 Parameter catalog and selective parameterizer

```python
@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: float
    bounds: tuple[float, float] | None
    units: str
    tags: tuple[str, ...]
    target_rules: tuple[str, ...]
    transform: str
    aliases: tuple[str, ...] = ()

class Parameterizer:
    def state(self, updates, base_state=None): ...
```

This catalog is the only source for defaults, bounds, transforms, fit groups,
CLI listings, and output tables.

Rules:

1. Empty updates return the input state unchanged.
2. A fit-specific `Parameterizer` is compiled for the statically selected
   parameter set.
3. Only distribution targets depending on selected parameters are written.
4. The parameterizer is built from the exact `ModelConfig`, never global
   `COMBE_PARAMS`.
5. Exact HOC mode fits reference scales/deltas unless a rule-based override is
   explicitly requested.
6. Interior bounded values may use sigmoid transforms. Exact boundary values
   use a boundary-safe policy such as projected physical-space updates; they
   are never silently nudged into the interval.
7. The catalog reports derived products and warns about simultaneously selected
   confounded factors.

Keep current short Combe names as aliases during migration so existing parameter
files remain readable.

For an exact reference with a portable analytic rule, the general
baseline-preserving wrapper is:

```text
target(theta)
  = hoc_reference
    + analytic_rule(features, theta)
    - analytic_rule(features, theta_default)
```

At `theta_default` this is exactly the HOC reference. Simpler strictly positive
conductance targets may use a multiplicative reference scale. Every target
declares which wrapper it uses and validates physical constraints such as
non-negativity.

### 7.6 Model recipe and builder

```python
@dataclass
class BuiltModel:
    cell: Any
    parameterizer: Parameterizer
    features: StaticFeatures
    signature: str
    provenance: ModelProvenance
```

Assembly order:

1. Load morphology.
2. Finalize compartment count.
3. Initialize geometry.
4. Validate groups and compute path-distance features.
5. Resolve and dependency-check enabled mechanisms.
6. Insert mechanisms.
7. Apply exact profiles or rule-based passive/active distributions.
8. Enable configured pumps/diffusion.
9. Set reference initial state.
10. Audit structure, profile coverage, and finiteness.
11. Freeze the static signature.

Assigning distributions after final discretization avoids the value-remapping
ambiguity documented in the HOC import notes. NEURON copies values from old
segments when `nseg` changes; it does not numerically interpolate a
nonconstant RANGE profile.

## 8. JIT and `vmap` parallelization backbone

The backbone follows Jaxley's official
[Speeding up simulations](https://jaxley.readthedocs.io/en/latest/tutorials/04_jit_and_vmap.html)
tutorial:

- Use `cell.data_set(..., param_state)` for values that vary across calls.
- Use `data_stimulate(..., data_stimuli)` for stimuli that vary across traces.
- Pass both into `jx.integrate`.
- Batch simulations with `vmap`.
- Compile the mapped function with `jit`.

Jaxley's
[training tutorial](https://jaxley.readthedocs.io/en/latest/tutorials/07_gradient_descent.html)
shows the shared-parameter pattern directly with
`vmap(simulate, in_axes=(None, 0))`.

The target kernel shape is:

```python
def simulate_one(theta, current, initial_state):
    param_state = parameterizer.state(theta)
    data_stimuli = injection_site.data_stimulate(current, None)
    voltage = jx.integrate(
        cell,
        param_state=param_state,
        data_stimuli=data_stimuli,
        all_states=initial_state,
        delta_t=dt,
        solver=solver,
        voltage_solver=voltage_solver,
        checkpoint_lengths=checkpoint_lengths,
    )
    return align_recording_to_observed_samples(voltage[recording_index])

def loss_one(theta, current, observed, score_mask, initial_state):
    predicted = simulate_one(theta, current, initial_state)
    squared_error = (predicted - observed) ** 2
    return jnp.sum(score_mask * squared_error) / jnp.sum(score_mask)

mapped_loss = jax.vmap(
    loss_one,
    in_axes=(None, 0, 0, 0, 0),
)
compiled_bucket_loss = jax.jit(weighted_reduce(mapped_loss))
```

The exact initial-state pytree API must be proven in a focused spike. If
per-trace state batching cannot be made transformation-safe, the fallback policy
must be explicit: use a shared configured initial voltage or map independently.
Do not silently reuse the first trace's voltage.

`align_recording_to_observed_samples` makes Jaxley's returned initial sample and
integration-step convention explicit. Its time vector must match the observed
`t_ms` vector exactly after resampling; implicit slicing such as
`voltage[:observed.size]` is retained only as a characterized legacy policy.

### 8.1 Shape buckets

Compile one kernel per static signature:

```text
(
  model fingerprint,
  solver and precision,
  dt and number of time steps,
  rematerialization layout,
  trace microbatch size,
)
```

The default data naturally requires two time buckets:

- `(4, 24_000)` depolarizing.
- `(4, 13_000)` hyperpolarizing.

Do not pad all traces to 24,000 by default. The two compiled executables avoid
11,000 unnecessary integration steps for each hyperpolarizing trace.

### 8.2 Trace-axis memory control

Full `vmap` is the preferred fast path. Reverse-mode simulation of four detailed
cells may exceed some GPUs, so support:

- `vmap` for the whole bucket.
- `jax.lax.map(..., batch_size=k)` as a memory-controlled mapped path; the
  [official JAX API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)
  describes it as a memory-efficient form of `vmap`.
- Serial mapping as the correctness reference.

Pad only the trace-count axis to a fixed microbatch size and use zero trace
weights for padded records. This prevents a final short microbatch from creating
another compiled shape.

Autotune `k` from `{1, 2, 4}` for this dataset, then write the chosen value into
the resolved configuration. Never autotune silently on resume.

### 8.3 Time-axis memory control

`checkpoint_lengths` controls Jaxley integration rematerialization, not training
resume. Keep it distinct from trace microbatching.

The current square-root heuristic is a baseline. Jaxley may simulate up to the
product of checkpoint lengths, so benchmark layouts separately for 13,000 and
24,000 steps and record both requested and actual padded steps.

### 8.4 Gradient aggregation

Each trace shares `theta`. Compute a normalized loss per trace, reduce within a
bucket, and add bucket gradients with their exact configured global weights
before one optimizer update. This is mathematically the full-dataset gradient,
not stochastic minibatching, unless the user selects stochastic mode.

Training functions return scalar loss and small metric arrays only. Full voltage
predictions are generated by a separate sparse evaluation function, preventing
an epoch-by-epoch device synchronization and host transfer.

### 8.5 Parallel axes

Use one vectorized axis at a time initially:

1. **Inside one GPU:** traces in a same-shape bucket.
2. **Across SLURM tasks:** seeds, model/channel/distribution configurations,
   morphologies, folds, or parameter initializations.
3. **Optional later:** nested `vmap` over parameter candidates after memory
   benchmarks.
4. **Optional later:** shard traces across GPUs and all-reduce the small shared
   parameter gradient.

Static model variants must never be placed inside the trace `vmap`.

## 9. All-segment dataset and objective

### 9.1 `TraceRecord`

An immutable record contains:

- `cell_id`, `trace_id`, and canonical `segment_id`.
- `voltage_mV`, `current_nA`, and `time_ms`.
- An explicit simulation-to-observation sample-alignment policy.
- Inferred and validated `dt_ms`.
- Trace-specific initial voltage/state.
- Original and segment-relative epoch timing.
- Score mask and normalized weight.
- Current amplitude, clipping flag, and input checksums.

The loader:

1. Selects every valid trace/segment for `m20240527cd` by default.
2. Reads all `v/i/t_ms` triplets.
3. Resolves files relative to the dataset root; it does not trust absolute
   manifest output paths.
4. Validates aligned lengths, finiteness, monotonic uniform time, units, and
   metadata.
5. Sorts deterministically by protocol and trace.
6. Uses manifest epoch timing to build score masks. Current-based detection is
   an explicit fallback when metadata is unavailable.
7. Resamples voltage linearly and current with zero-order hold when requested.
8. Rejects a `max_steps` truncation that removes the stimulus.

Keep aliases such as `depolarizing_pulse -> depolarizing_step` and
`hyperpolarizing_step -> hyperpolarizing_pulse` at the input boundary only.

### 9.2 Default loss

For protocol `p`, trace `r`, mask `m`, and parameter vector `theta`:

```text
trace_mse(r, theta)
  = sum_t m[r,t] * (V_model(theta, I_r, v0_r)[t] - V_r[t])^2
    / sum_t m[r,t]

protocol_loss(p, theta)
  = mean_{r in p}(trace_mse(r, theta))

total_loss(theta)
  = sum_p protocol_weight[p] * protocol_loss(p, theta)
```

Default protocol weights are 0.5 depolarizing and 0.5 hyperpolarizing. With four
traces in each protocol, every trace has global weight `1/8`. This ensures that a
long trace does not dominate solely because it has more samples.

Expose:

- `protocol_mean` as the default.
- `trace_mean`.
- `sample_mean` for explicit legacy-like behavior.
- User-specified trace/protocol weights.

Always report per-trace, per-protocol, and total values so aggregation is
auditable.

### 9.3 Optional objective components

The first parity implementation uses masked voltage MSE only. The architecture
supports later weighted components:

- Baseline/resting voltage.
- Final step voltage.
- Sag/rebound.
- Spike count.
- Spike timing.
- Voltage derivative or spectral terms.
- Priors/regularization around reference parameters.

Hard threshold spike metrics are evaluation outputs unless a differentiable
surrogate is explicitly selected. Do not accidentally put nondifferentiable
reporting code into the training loss.

### 9.4 Parameter identifiability

Forty free values from eight somatic traces are likely underconstrained, and
some values appear only as products. Provide presets:

- `legacy_joint_40`: compatibility/characterization only.
- `passive_then_active`: fit a reviewed passive subset, then active subset, then
  a joint refinement over all traces.
- `reviewed_joint`: omit one factor from known confounded products and apply
  priors.

Every stage records its selected parameters and trace weights. The default
dataset remains all eight traces; a staged preset may emphasize a protocol only
when that choice is explicit in its resolved configuration.

## 10. Runtime and GPU design

### 10.1 Bootstrap before importing JAX

Importable model modules never set `jax_platform_name`, warning filters, XLA
memory variables, plotting paths, or global monkeypatches.

The executable bootstrap:

1. Reads the runtime section with a JAX-free loader.
2. Sets process environment needed before JAX import.
3. Imports JAX.
4. Applies precision policy.
5. Validates available devices and requested backend.
6. Imports the model/simulation package.
7. Records backend, device names, precision, and versions.

`backend=auto` chooses an available GPU, otherwise CPU. An explicit unavailable
GPU request fails rather than falling back silently.

### 10.2 Precision

CPU float64 is the scientific regression reference. GPU float64 is the first
accelerated target. Jaxley's speed tutorial warns that detailed morphologies can
be unstable in float32; float32 therefore requires:

- Finite state and gradient tests.
- Trace and spike parity against float64.
- A documented performance benefit.
- An explicit config selection.

### 10.3 Solver

Keep `bwd_euler` and `voltage_solver=jaxley.dhs` as the initial reference.
Jaxley can choose the device-specific DHS implementation.

The tutorial notes that `exp_euler` can help some GPU models below roughly
1,000 compartments. This model is near that size, but it has custom calcium
dynamics and diffusion. Treat `exp_euler` as a benchmark-gated option, not a new
default.

### 10.4 Performance hygiene

- Cache compilations by static signature.
- Put a per-task persistent compilation cache in configured scratch.
- Avoid plotting and full prediction transfer in train steps.
- Evaluate/plot only at configured intervals and after training.
- Log compile time, steady-state step time, peak device memory, padded
  integration steps, and microbatch size.
- Profile one forward pass and one value-and-gradient pass independently.
- Keep compatibility patches version-gated and tested in one runtime module;
  prefer dependency pins or an upstream fix over global mutation.

## 11. SLURM architecture

### 11.1 First production tier: one fit per GPU

`fit_array.sbatch` requests:

- One task.
- One GPU.
- Explicit CPUs, memory, time, and log paths.
- A preemption signal such as `USR1` before wall time.

A versioned TSV/JSON manifest maps `SLURM_ARRAY_TASK_ID` to:

- Resolved base config.
- Overrides.
- Seed.
- Output root.
- Optional cell/fold/morphology/channel variant.

The task:

1. Activates a locked environment or Apptainer image.
2. Copies small immutable inputs/artifacts to `$SLURM_TMPDIR`.
3. Places temporary files and compilation cache in task-local scratch.
4. Starts through `srun`.
5. Validates exactly one visible GPU when GPU mode is requested.
6. Resumes a compatible checkpoint if present.
7. Writes durable artifacts to a unique run directory.

SLURM arrays are independent optimizations. They do not collectively optimize
one shared parameter vector.

### 11.2 Run identity and collision avoidance

Use:

```text
<model>-<cell>-<structural_hash>-<fit_hash>-seed<seed>
```

Record SLURM job, array, task, node, and restart IDs in provenance. Creation uses
an atomic claim file; two tasks cannot write the same run directory.

### 11.3 Preemption

The Python process handles `USR1`/`TERM` by:

1. Marking a checkpoint request.
2. Finishing the current safe epoch/bucket boundary.
3. Atomically saving latest state.
4. Flushing metrics.
5. Exiting with the documented requeue behavior.

Checkpoint contents:

- Epoch and bucket cursor.
- Unconstrained and constrained parameters.
- Optimizer state.
- Best parameters, loss, and epoch.
- Random key.
- History cursor.
- Resolved config and hash.
- Dataset manifest and input hashes.
- Model/artifact signature.
- Code revision and dirty-state digest.
- Package versions, backend, device, precision, and solver settings.

Resume rejects changed structural, data, or fit hashes unless a separate
fine-tune command explicitly starts a new run.

### 11.4 Later multi-GPU tier

Only after single-GPU correctness:

- Shard same-length traces over local GPUs.
- Compute local weighted loss/gradient numerators.
- All-reduce the shared gradient.
- Update parameters identically on each rank.
- Let rank zero write durable outputs.

With only eight traces, useful scaling is limited and collective overhead may
dominate. Multi-host JAX is a later benchmark-driven option, not part of the
initial definition of done.

## 12. Reproducible outputs

```text
runs/<run_id>/
  resolved_config.yaml
  run_manifest.json
  status.json
  metrics.jsonl
  parameters_initial.csv
  parameters_best.csv
  parameters_final.csv
  checkpoints/
    latest/
    best/
  predictions/
    best.npz
    final.npz
  plots/
  logs/
```

`run_manifest.json` includes:

- Git commit and dirty diff hash.
- Exact package versions and environment/container digest.
- Device/backend/precision.
- Seed and SLURM identity.
- Model structural signature and morphology artifact provenance.
- Enabled mechanisms, placements, and distribution rules.
- Dataset selection, file hashes, units, resampling, and final trace weights.
- Solver, checkpoint layout, microbatch size, and compile-cache policy.

Metrics are appended during training. Large predictions and plots are separate
from checkpoints and are written only by the owning process.

## 13. Migration plan and gates

Each milestone is intended to be a reviewable, independently testable change.
Do not start the next compute milestone before the previous scientific gate is
green.

### M0 — Freeze and explain current behavior

Tasks:

- Capture exact HOC and SWC structural/profile audits at `d_lambda=0.3`.
- Freeze two current traces/objectives:
  - Built exact HOC model with no fit `param_state`.
  - Legacy formula-based fit state at nominal values.
- Retain the now-passing regression test that prevents the characterized
  no-op mutation from returning.
- Capture current NEURON/Jaxley comparison metrics at 0.3 and 0.9 nA.
- Hash default input traces and manifest.
- Record the characterized environment.

Gate:

- A clean, documented command reproduces both legacy baselines.
- The intentional choice of exact-HOC baseline for the new default is approved.

### M1 — Package, configuration, and runtime boundary

Tasks:

- Create `pyproject.toml`, lock file, CPU profile, GPU profile, and test extras.
- Remove `sys.path` dependence through package imports.
- Add typed config schemas and config hashing.
- Add a JAX-free CLI bootstrap and device validation.
- Move compatibility patches into a version-checked runtime module.
- Keep current model functions available through imports.

Gate:

- Importing the library does not force CPU or initialize plotting policy.
- Config validation catches unknown providers, mechanisms, parameters, units,
  and invalid bounds.
- CPU smoke tests run from an installed clean environment.

### M2 — Morphology providers and portable exact-HOC artifact

Tasks:

- Implement provider protocol, group validation, features, and fingerprints.
- Extract current HOC live importer without changing results.
- Implement SWC provider with caller-supplied path/classifier/root.
- Implement artifact export/load and coverage reports.
- Add deterministic extraction/target compilation instructions for
  `Combe2023.zip`.
- Ensure GPU worker model construction does not import NEURON.

Gate:

- Exact artifact reproduces 144 branches, 974 compartments, group counts, and
  audited passive/ion/mechanism columns at `d_lambda=0.3`.
- SWC remains explicitly distinct and reproduces its own frozen fingerprint.
- Missing groups, ambiguous root, artifact hash mismatch, or missing required
  HOC targets fail clearly.

### M3 — Unified mechanism registry and distribution engine

Tasks:

- Promote one mechanism registry for build and validation.
- Encode placement as data.
- Add dependency validation for calcium channels, Cal4, Kca/MyKca, and
  diffusion.
- Implement shared JAX distribution primitives.
- Port Combe passive and active rules one target at a time.
- Keep translated channel equations unchanged.

Gate:

- Placement audit matches the frozen current channel/region table.
- Every rule matches frozen arrays in rule-based mode.
- Required channel mapping omissions no longer pass silently.
- Isolated mechanism and calcium-state tests are finite.

### M4 — Parameter catalog and baseline-preserving parameterizer

Tasks:

- Replace dataclass/group/bounds duplication with `ParameterSpec`.
- Define stable ordering, aliases, units, tags, targets, and transforms.
- Implement exact-HOC reference scale/delta parameterization.
- Implement selective rule invalidation/update.
- Add confounding and zero-bound diagnostics.
- Add a legacy adapter for `params`, `bounds`, and `set_fitted_parameters`.

Gate:

- `state({})` is an identity.
- Reapplying defaults gives the same trace as no `param_state`.
- Updating one parameter changes only declared targets.
- Default simulation matches the M0 exact-HOC baseline within frozen CPU x64
  tolerance.

### M5 — Manifest-driven trace dataset

Tasks:

- Implement `TraceRecord`, loader, unit conversion, validation, masks, and
  deterministic sorting.
- Prefer relative reconstruction over absolute manifest paths.
- Use `t_ms` to validate/infer `dt`.
- Implement voltage interpolation and current zero-order hold.
- Create the default dataset config.

Gate:

- Default loader returns exactly eight records:
  - Four `(24_000,)` depolarizing.
  - Four `(13_000,)` hyperpolarizing.
  - All at 0.05 ms with expected amplitudes/clipping flags.
- Missing triplets, mismatched lengths, nonfinite values, invalid time, or unit
  ambiguity fail with trace-specific messages.

### M6 — Serial, `vmap`, and microbatched simulation

Tasks:

- Implement pure `simulate_one`.
- Implement trace-specific initial states.
- Implement and test explicit initial-sample/time alignment.
- Add serial reference, full `vmap`, and `lax.map` strategies.
- Build the two shape buckets and fixed trace-count padding.
- Compile/cache by static signature.
- Separate training scalar path from prediction/evaluation path.

Gate:

- Serial, `vmap`, and microbatched predictions match.
- Their per-trace losses and parameter gradients match within frozen x64
  tolerances.
- The default dataset produces two time-shape compilations, not one per trace.
- A synthetic 1 mV residual yields exactly 1 mV² per trace and in the aggregate,
  independent of trace length.

### M7 — Multi-trace trainer and durable checkpoints

Tasks:

- Implement hierarchical objective and metric reporting.
- Add Optax optimizer, clipping, optional schedules, and fit presets.
- Add atomic latest/best checkpoints and compatibility validation.
- Evaluate and plot sparsely outside the train step.
- Add status and provenance files.

Gate:

- All eight records have positive weights summing to one.
- Aggregate loss/gradient equals an explicit weighted sum of eight serial
  results.
- Interrupted `N + M` epochs match an uninterrupted `N+M` run.
- Best and final parameters are evaluated and distinguishable.
- No full trace is copied to the host on ordinary train epochs.

### M8 — GPU qualification

Tasks:

- Remove remaining CPU pins.
- Add GPU environment/container profile.
- Benchmark trace microbatch sizes and rematerialization layouts.
- Add persistent compilation cache and performance logging.
- Test float64 first; evaluate float32 and `exp_euler` only as optional variants.

Gate:

- Runtime reports the requested GPU and never silently falls back.
- Short forward/loss/gradient smoke tests are finite.
- GPU x64 agrees with CPU x64 within calibrated tolerance.
- Full default loss fits in configured device memory.
- A benchmark report identifies compile time, steady-state time, memory, and the
  selected microbatch/checkpoint policy.

### M9 — SLURM production and compatibility migration

Tasks:

- Add artifact-preparation and fit-array scripts.
- Add deterministic array manifest generation.
- Add unique output claims, signal checkpointing, and resume/requeue.
- Add dry-run/validate-only modes.
- Redirect old Combe factory/fitter through compatibility adapters.
- Document deprecation and remove duplicated fit scripts only after callers
  migrate.

Gate:

- A two-task cluster smoke array creates two collision-free reproducible runs.
- Simulated preemption leaves a valid checkpoint and resumes correctly.
- A clean Linux worker can fit from an exact HOC artifact without NEURON.
- Current comparison runner works through the compatibility factory.

## 14. Acceptance matrix

| Area | Required check |
|---|---|
| Baseline identity | Empty/default parameter update matches built exact-HOC model |
| Structure | HOC branch/compartment/group counts exactly match M0 |
| Profiles | Passive, ion, and mechanism audits match M0 |
| Scientific comparison | Preserve documented HOC-vs-Jaxley results: about 0.011 mV RMSE at 0.3 nA and 0.128 mV at 0.9 nA; first 0 mV crossing within one 0.025 ms step |
| Dataset | Exactly eight default records with two expected shapes |
| Units/time | pA-to-nA conversion and time-vector validation are explicit |
| Sample alignment | Predicted time zero and integration steps map explicitly to observed `t_ms` |
| Objective | Weighted aggregate equals manual serial calculation |
| Ordering | Dataset order does not change loss |
| Mapping | Serial, `vmap`, and microbatch predictions/loss/gradients agree |
| Gradients | Finite; short finite-difference checks agree with autodiff |
| Bounds | Every fitted value remains in bounds; exact zero policy is tested |
| Compilation | At most one executable per default static time/microbatch signature |
| Resume | Interrupted and uninterrupted runs agree |
| GPU | Requested backend is honored; CPU/GPU x64 parity is within calibrated tolerance |
| SLURM | Unique outputs, rank-safe writes, valid preemption checkpoint |
| Provenance | Config/data/code/model/environment/device hashes are sufficient to reconstruct the run |

Numerical tolerances for refactor-vs-current, serial-vs-mapped, and CPU-vs-GPU
must be measured and frozen in M0/M8. Do not reuse the much looser
NEURON-vs-Jaxley tolerance for internal refactor parity.

## 15. Risk register

| Risk | Mitigation |
|---|---|
| Fitter starts from a different model than the builder | M0 characterization; M4 identity invariant |
| HOC topology is lost through SWC | Separate explicit modes and fingerprints |
| Unknown morphology sections become apical | Complete classifier with fail-on-unknown |
| HOC mapping silently skips values | Coverage report and required-target failures |
| Duplicated NumPy/JAX rules drift | One JAX rule evaluator |
| Forty parameters are underidentified | Fit presets, priors, confounding diagnostics, multistart |
| Zero conductances cannot activate through sigmoid | Explicit boundary-safe transform/projection |
| `vmap` exhausts GPU memory | Fixed microbatch plus `lax.map`/serial fallback |
| Long traces make reverse mode too large | Independent time rematerialization tuning |
| Shape variation causes compile storms | Bucket by `dt/n_steps`; pad trace-count axis only |
| Float32 destabilizes detailed morphology | Float64 default and explicit qualification |
| HOC/NEURON unavailable on GPU nodes | Portable exact-HOC artifact |
| Native MOD files differ by architecture | Compile preprocessing inputs on target; worker uses artifact |
| Global monkeypatch changes other jobs | Version-gated runtime compatibility layer |
| Epoch plotting overwhelms shared storage | Off by default; sparse rank-zero evaluation |
| SLURM preemption loses work | Atomic optimizer checkpoints and signals |
| Concurrent runs overwrite outputs | Config-hashed run IDs and atomic claims |
| Dependency drift changes numerics | Exact lock, environment/container digest, regression CI |
| Calcium translation remains incompletely validated | Dedicated Cal4/diffusion/state/gradient tests |

## 16. Mapping from current functions

| Current code | Target owner |
|---|---|
| `CombeParameters`, `params`, `bounds` | `parameters/catalog.py` + configs |
| `morphology_path` | morphology provider configuration |
| `build_hoc_section_cell` | `morphology/hoc_live.py` |
| `apply_hoc_*_profile` | HOC artifact exporter/loader |
| `_hoc_group_name` | explicit grouping strategy |
| `HOC_CHANNEL_CLASSES` | unified mechanism registry |
| `update_number_compartments` | discretization strategy |
| `set_distances_from_soma` | static morphology features |
| `insert_combe_channels` | placement specs + mechanism registry |
| `set_passive_properties` | distribution rules |
| `set_soma/apical/axon/basal_channels` | distribution rules |
| `set_fitted_*` | selective `Parameterizer` |
| `Combe2023` | Combe recipe + generic builder |
| `add_step_stimuli` | protocol/simulation layer |
| `segment_file`, `load_segment` | `TraceDataset` |
| inner `simulate` | pure single-trace kernel |
| inner `loss` | composable objective |
| epoch loop and CSV/plots | trainer, checkpoints, reporting |

## 17. Compatibility surface

Until downstream code migrates, retain thin adapters for:

- `Combe2023(d_lambda, enable_calcium_diffusion, params, morphology_source)`.
- `L5PC_Combe`.
- `params`, `bounds`, and `SEGMENTED_TRACES_DIR`.
- `set_fitted_parameters(cell, keys, values, state=None)`.
- Existing Combe comparison entry points.
- Existing parameter key names and ordering.
- Groups `soma`, `axon`, `basal`, and `apical`.
- Default somatic injection/recording at branch 0, location 0.5.
- Calcium diffusion coefficient 0.22.
- Model units: mV, ms, nA, and micrometres.

Expose a named `legacy_formula` profile only if reproducing the current
fit-start discontinuity is required for old result comparison. It must never be
the new default.

## 18. Definition of done

The refactor is complete when:

- The folder is an installable, locked package with automated tests.
- Morphology, channel set, placement, and distributions are configuration
  knobs.
- Exact-HOC and rule-based morphology modes are explicit and audited.
- Empty/default parameter application is an identity.
- The default fit discovers and minimizes over all eight `m20240527cd` segments.
- The full-dataset loss uses normalized, reported trace/protocol weights.
- The compute path follows Jaxley's `data_set`/`data_stimulate` and
  `jit(vmap(...))` model, with a tested memory-controlled fallback.
- No importable library module forces CPU.
- CPU and GPU regressions pass.
- Training checkpoints are durable and resumable.
- SLURM arrays can run independent configurations without collisions.
- Legacy imports continue through a documented adapter during deprecation.

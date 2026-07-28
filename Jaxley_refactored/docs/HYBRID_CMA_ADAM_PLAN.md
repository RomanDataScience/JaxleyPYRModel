# Hybrid CMA-ES and gradient fitting plan

## Objective

Add a reproducible three-stage optimizer that uses the same bounded parameter
space and LSU_1 objective throughout:

```text
CMA-ES global search
        |
        v
fixed-step Adam exploration
        |
        v
Adam with backtracking refinement
        |
        v
held-out-trace selection
```

CMA-ES should locate promising basins without gradients. Fixed-step Adam should
use Jaxley gradients to move rapidly within each selected basin and may accept
temporarily worse steps. Backtracking Adam should perform the final stable
refinement. The hybrid result must be compared with randomized multistart Adam
under a matched simulation-evaluation budget.

## Scientific scope

### Training and validation split

- Training: first and third depolarizing and hyperpolarizing traces.
- Validation: second and fourth depolarizing and hyperpolarizing traces.
- CMA-ES and both Adam stages must see training traces only.
- Candidate selection during optimization uses training loss.
- Final model selection uses validation metrics after all optimization has
  stopped.
- Report both the continuous LSU_1 objective and discrete post-hoc metrics:
  spike count, firing rate, spike peak, spike width, plateau voltage, AHP depth,
  recovery time, and hyperpolarizing RMSE.

### Initial parameter subset

Do not begin with full-covariance CMA-ES over all 40 parameters. Start with a
mechanistically relevant subset of approximately 12–20 active parameters:

- sodium: `AXNa`, `gna`, `gnadend`, `scale_Na_conduct`, `nap_gnabar`;
- delayed rectifier/Kv2: `gkdrsoma`, `gkdrdend`, `axongkdr`,
  `gkdrapical`, `gkv2soma`, `gkv2`, `gkv2axon`, `gkv2scale`;
- A-type potassium: `soma_kap`, `axon_kap`, `basal_kap`, `soma_kad`;
- slow/recovery currents as needed: `soma_hbar`, `KirGbar`, `soma_km`,
  `mykca_init`, `soma_kca`;
- selected passive values only if passive fitting has not already been staged:
  `RmSoma`, `Epas`, and `CmSoma`.

The first implementation should make this list configurable. Parameters outside
the CMA subset remain at the supplied starting vector and become trainable
during the later Adam stages if selected by the fit configuration.

## Optimization coordinates

Use the existing `ProjectedBoxSpace`; CMA-ES operates in normalized `[0, 1]`
coordinates. This avoids mixing conductances, resistances, voltages, and
dimensionless coefficients in one raw physical scale.

- CMA mean: configured reference, randomized start, or imported candidate.
- Initial CMA standard deviation: `0.15` normalized units.
- Bounds: `[0, 1]` for every searched coordinate.
- Boundary handling: resample out-of-bounds proposals instead of clipping them.
  Clipping creates artificial probability mass at exact boundaries.
- Exact-zero reference parameters: configurable. Initially exclude them from
  CMA or seed them with a small positive physical value before normalization.
- Every saved candidate includes both normalized and physical values.

Wide bounds change both the feasible region and the physical size represented
by one normalized step. Global-search comparisons must therefore use one fixed
bound policy.

## Architecture

### 1. Extract a forward-only objective evaluator

Create `jaxley_refactored/fitting/evaluator.py` with a small interface:

```python
class ObjectiveEvaluator:
    def evaluate(self, normalized, *, predictions=False) -> Evaluation: ...
    def evaluate_population(self, population) -> tuple[Evaluation, ...]: ...
```

`Evaluation` contains:

- total training loss;
- component losses;
- bucket losses;
- common RMSE;
- finite/invalid status;
- optional simulated traces.

Refactor the current private `Trainer._evaluate()` logic so the trainer and
CMA-ES use the same compiled simulation kernels and loss reductions. CMA-ES
must call the forward path only; it must not construct reverse-mode gradients.

Acceptance checks:

- The evaluator and current trainer return identical loss values for the same
  normalized vector.
- Invalid simulations produce a configured finite penalty and a diagnostic,
  not NaN propagation into CMA state.
- Prediction arrays are not transferred to the host during ordinary CMA
  evaluations.

### 2. Add an optimizer-independent candidate record

Create an immutable record containing:

```text
candidate_id
generation
population_index
normalized_values
physical_values
training_loss
component_losses
status
parent/search provenance
```

Write one summary row per candidate to `candidates.jsonl`. Store large numeric
arrays in versioned `.npz` snapshots. Do not use pickle.

### 3. Add a CMA-ES ask/tell adapter

Create `jaxley_refactored/fitting/global_search/`:

```text
base.py          GlobalOptimizer protocol
cma_es.py        CMA-ES ask/tell implementation
checkpoints.py   versioned JSON/NPZ state
runner.py        generation orchestration
```

The adapter must expose:

```python
ask() -> population
tell(population, losses) -> generation_metrics
state_dict() -> JSON/NumPy-compatible state
load_state_dict(state)
```

Implementation decision:

1. First evaluate a maintained ask/tell CMA package behind the adapter.
2. Add it as an optional dependency, e.g. `global-search`, not a mandatory
   dependency for ordinary fitting.
3. Accept the package only if all state required for exact resume can be
   serialized without pickle.
4. If exact non-pickle state export is not supported, implement the canonical
   full-covariance CMA-ES state directly with NumPy. Keep it independent of JAX;
   JAX is responsible only for candidate evaluation.

Required CMA state includes generation, mean, global sigma, covariance matrix,
evolution paths, recombination weights, evaluation count, best candidate, and
random-generator state.

### 4. Configuration schema

Add a top-level `search` section without overloading the existing Adam schema:

```yaml
search:
  strategy: hybrid

  global:
    algorithm: cma_es
    seed: runtime
    parameter_names: [AXNa, gna, gnadend, scale_Na_conduct, ...]
    population_size: 16
    generations: 20
    sigma0: 0.15
    boundary_policy: resample
    invalid_loss: 1.0e12
    elites: 4
    checkpoint_every_generations: 1

  local_exploration:
    optimizer: adam
    epochs: 50
    learning_rate: 0.005
    gradient_clip_norm: 10.0
    line_search:
      enabled: false

  local_refinement:
    optimizer: adam
    epochs: 150
    learning_rate: 0.001
    gradient_clip_norm: 10.0
    line_search:
      enabled: true

  selection:
    rank_by: validation_loss
    keep_after_global: 4
    keep_after_exploration: 2
```

Validate population size, generation count, sigma, unique parameter names,
subset membership, stage counts, and checkpoint intervals before building the
model.

### 5. Candidate handoff to Adam

Extend initialization with:

```yaml
fit:
  initialization:
    mode: candidate_file
    path: ...
    coordinate_system: normalized
```

The handoff must:

- reproduce the CMA candidate exactly;
- reset Adam moments to zero;
- preserve the same parameter ordering and bounds;
- reject candidates produced by incompatible model/config/input hashes;
- write parent candidate ID into the local run manifest.

Do not perturb CMA elites during the first comparison. Add optional
elite-neighborhood perturbation only after exact handoff is verified.

### 6. Hybrid orchestration command

Add:

```bash
jaxley-refactored hybrid-fit --config configs/search/LSU_1_cma_adam.yaml
```

The orchestrator should:

1. Build the model and training evaluator once.
2. Run or resume CMA generations.
3. Save and rank elite candidates.
4. Start independent fixed-step Adam child runs from each elite.
5. Rank completed exploratory runs by training objective.
6. Start backtracking refinement from the retained candidates.
7. Evaluate all refined candidates on held-out traces.
8. Write a final comparison table and selected-model pointer.

Each stage gets its own directory:

```text
runs/<hybrid-id>/
  resolved_config.yaml
  run_manifest.json
  global/
    candidates.jsonl
    generations.jsonl
    checkpoints/
    elites/
  local_exploration/<candidate-id>/
  local_refinement/<candidate-id>/
  validation/
    metrics.csv
    plots/
  selected_model.json
```

### 7. Population evaluation strategy

Implement in this order:

1. **Serial population evaluation in one process.** This is the correctness
   baseline and reuses compiled kernels.
2. **CPU process parallelism.** Split candidates among independent Slurm array
   tasks, with one coordinator collecting completed candidate files before
   `tell()`. Use atomic result files and generation IDs.
3. **Nested JAX batching only after profiling.** Potential shape:

   ```text
   vmap(candidate) -> vmap(trace) -> integrate
   ```

   This may improve throughput but can multiply memory use during detailed
   multicompartment simulation. It is optional and must match serial losses.

Do not use MPI to split one trace. Parallelize independent candidate
simulations.

### 8. Slurm workflow

Provide two scripts:

- `slurm/hybrid_cma_coordinator.sbatch`: owns ask/tell state and generations;
- `slurm/hybrid_cma_candidates.sbatch`: evaluates one or more candidates from a
  generation manifest.

For the first implementation, use one CPU coordinator job with serial candidate
evaluation. Once correct, add a generation-at-a-time array workflow:

```text
coordinator writes generation_0003.tsv
        |
        v
Slurm array evaluates candidates 0..15
        |
        v
coordinator validates all atomic result files
        |
        v
CMA tell() and checkpoint
```

Never allow a partially completed generation to advance CMA state. Resubmission
must evaluate only missing or invalid candidates.

## Fidelity and compute budget

Start with full trace windows but consider `dt=0.2 ms` for CMA screening only.
Re-evaluate every elite at `dt=0.1 ms` before Adam handoff because spike counts
and rankings can change with timestep.

Initial experiment:

```text
CMA subset dimensions       12–20
population size             16
generations                 20
CMA evaluations             320
global elites               4
fixed-step Adam             4 × 50 epochs
backtracking finalists      2 × 150 epochs
```

Record wall time and number of forward/gradient evaluations. Compare methods
under both equal wall-clock and equal simulation-evaluation budgets.

## Failure handling

- Nonfinite voltages or losses: assign `invalid_loss` and record the reason.
- Simulation exception: record a failed candidate; do not crash the generation.
- Too many invalid candidates: shrink sigma and retry the generation once.
- Boundary crowding: report per-parameter boundary occupancy.
- CMA covariance ill-conditioning: regularize eigenvalues and checkpoint before
  recovery.
- Interrupted Adam stage: use existing durable checkpoints.
- Interrupted CMA stage: restore the last fully completed generation only.

## Testing

### Unit tests

- CMA ask/tell improves a sphere objective.
- Deterministic seeds reproduce populations and state.
- State save/load reproduces the next population exactly.
- Bounds are never violated.
- Invalid candidates receive the configured penalty.
- Parameter subsets embed into and extract from the full vector correctly.
- Candidate-file initialization is exact.
- Global and trainer evaluators return the same LSU_1 loss.

### Integration tests

- Two-generation synthetic hybrid search completes and resumes.
- A tiny Jaxley smoke model runs CMA -> Adam -> backtracking.
- Serial and population-batched evaluations agree in float64.
- Training and validation records never overlap.
- The final selected candidate is chosen from validation metrics.

### Scientific acceptance

Against randomized multistart Adam:

- hybrid search finds equal or lower best training loss under a matched budget;
- held-out firing-rate and waveform metrics do not degrade materially;
- results reproduce from config, seed, and input hashes;
- parameter solutions are inspected for boundary saturation and cross-seed
  variability;
- conclusions are based on validation performance, not one favorable seed.

## Implementation milestones

1. **M1 — Shared forward evaluator:** refactor without changing current fit
   results.
2. **M2 — Candidate files and imported initialization:** exact handoff into
   Adam.
3. **M3 — CMA ask/tell core:** serial population evaluation, checkpoints, and
   analytic-function tests.
4. **M4 — Hybrid CLI:** CMA elites -> fixed Adam -> backtracking Adam.
5. **M5 — Validation split and physiological report:** traces 2/4 and discrete
   metrics.
6. **M6 — Slurm population arrays:** atomic generation manifests and resume.
7. **M7 — Performance work:** profile nested `vmap`, candidate microbatches,
   and coarse-to-full fidelity.

M1–M5 are required for a scientifically usable first version. M6–M7 are
throughput improvements and should not block correctness experiments.

## Recommended first experiment

Before full CMA-ES, establish the baseline:

1. Run 16 seed-jittered `LSU_1_wide_bounds_adam` searches for 50 epochs.
2. Backtracking-refine the best four.
3. Evaluate all four on held-out traces.
4. Record compute cost and parameter diversity.

Then run CMA-ES with the same total evaluation budget on a 12–20 parameter
subset. This determines whether CMA-ES adds value beyond the simpler multistart
system before investing in distributed population evaluation.

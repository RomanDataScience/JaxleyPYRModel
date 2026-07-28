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

## Publication-grade research contract

This project must distinguish four claims that require different evidence:

1. **Software correctness:** the implementation computes the stated model,
   objective, gradients, optimizer updates, checkpoints, and metrics.
2. **Optimization efficacy:** under a matched compute budget, the hybrid method
   finds better training optima than prespecified baselines.
3. **Predictive validity:** fitted models predict held-out stimulation
   conditions and held-out cells.
4. **Biophysical inference:** fitted parameter values support claims about ion
   channels or mechanisms.

Passing an earlier claim does not establish a later one. In particular, a low
voltage-trace loss does not establish that individual conductances are uniquely
identified. Mechanistic claims require identifiability and uncertainty
analysis.

Before evaluating a locked test set, freeze and version:

- model equations, morphology processing, mechanisms, solver, and timestep;
- parameter subset, canonical/wide bounds, and parameter transforms;
- LSU_1 definition, windows, thresholds, scales, and component weights;
- optimizer baselines, CMA population/generations, Adam schedules, and stopping
  criteria;
- exclusion criteria for cells/traces and numerical failures;
- primary/secondary endpoints and statistical analysis;
- random seeds and compute budgets.

Any change made after test-set inspection creates a new analysis version and
requires a fresh test set or must be labeled exploratory. Optimizer seeds are
technical replicates, not biological replicates.

No plan can guarantee publication in a particular journal. The target should be
a defensible, reproducible result whose claims are proportional to the number of
independent cells, protocols, and external validation conditions.

## Scientific scope

### Data hierarchy and leakage control

- A trace is not an independent biological replicate; the independent unit is
  normally the recorded cell, animal, or preparation appropriate to the
  experimental design.
- Repeated sweeps from one cell quantify within-cell repeatability but must not
  inflate the biological sample size.
- The current first/third training and second/fourth validation split is useful
  for within-cell interpolation only.
- Because hyperpolarizing pulses may repeat the same amplitude across sweeps,
  verify whether they are technical replicates rather than distinct stimulation
  conditions.
- CMA-ES, Adam stages, loss-weight tuning, parameter-subset selection, and
  stopping-rule development must see development data only.
- Maintain three conceptual partitions:
  - **development/training:** optimizer updates;
  - **validation:** hyperparameter and model selection;
  - **locked test:** one-time final assessment.
- For a multi-cell dataset, prefer nested cell-level evaluation:
  - outer split holds out entire cells for final testing;
  - inner split tunes optimizer and loss settings using development cells;
  - within each cell, hold out current amplitudes or protocols to measure
    interpolation/extrapolation.
- If the available dataset is too small for a locked test set, use
  leave-one-cell-out evaluation and state clearly that independent external
  validation remains outstanding.
- A convincing generalization claim should include additional cells not used
  during LSU_1 or optimizer development, ideally from a separate acquisition
  batch and, if feasible, an external laboratory.

Candidate ranking during optimization uses training loss. Hyperparameter
selection uses validation performance. The locked test set is evaluated once
after every decision is frozen.

### Endpoints

- Prespecify one primary endpoint, such as held-out voltage RMSE or a composite
  of standardized held-out physiological errors.
- Prespecify key secondary endpoints: firing-rate error, spike-count error,
  spike peak/width, plateau voltage, AHP depth, recovery time, and
  hyperpolarizing RMSE.
- LSU_1 is the training objective, not automatically the primary scientific
  endpoint.
- Compute discrete spike metrics only for evaluation; retain smooth surrogates
  for differentiation.
- Define spike-detection thresholds, filtering, and failure handling before
  viewing locked test results.
- Report both the continuous LSU_1 objective and discrete post-hoc metrics:
  spike count, firing rate, spike peak, spike width, plateau voltage, AHP depth,
  recovery time, and hyperpolarizing RMSE.

### Synthetic ground-truth benchmark

Before interpreting experimental fits, create synthetic datasets from known
parameters:

- simulate the exact training/validation protocols;
- add realistic measurement noise, baseline drift, and stimulus uncertainty;
- test several ground-truth parameter vectors, including boundary-near cases;
- quantify parameter recovery, prediction recovery, confidence-interval
  coverage, optimizer failure rate, and basin discovery;
- repeat with model mismatch, e.g. an omitted current or perturbed morphology.

This separates optimizer failure from non-identifiability and from model
misspecification. A method that cannot recover identifiable synthetic
parameters under realistic noise is not ready for biological interpretation.

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

The first implementation should make this list configurable. Parameter-subset
selection must be based on prior biology or sensitivity analysis performed on
development data only. Parameters outside the CMA subset remain at the supplied
starting vector and become trainable during later Adam stages if selected by
the fit configuration.

Record and report every tested subset, not only the best-performing one.

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
    global_rank_by: training_loss
    local_rank_by: training_loss
    final_rank_by: validation_primary_endpoint
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

## Comparator study and fair compute accounting

The hybrid method must be compared against prespecified baselines:

1. reference initialization + backtracking Adam;
2. randomized multistart + fixed-step Adam;
3. randomized multistart + backtracking Adam;
4. CMA-ES only;
5. CMA-ES + fixed-step Adam;
6. full CMA-ES + fixed-step Adam + backtracking hybrid;
7. random or quasi-random search with the same forward-evaluation budget.

Required ablations:

- canonical versus wide bounds;
- with versus without seed jitter;
- full LSU_1 versus removal of firing-rate, spike-height, or recovery terms;
- all selected parameters versus the reduced CMA subset;
- fixed-step versus backtracking local stages;
- coarse-to-full fidelity versus full fidelity throughout.

Fairness rules:

- use identical development/test partitions and model code;
- use a prespecified set of optimizer seeds shared across methods where
  meaningful;
- count forward simulations, reverse-mode gradient evaluations, compilation
  time, wall time, peak memory, and CPU/GPU hours;
- report results both with compilation included and amortized;
- do not call one epoch from different methods an equal budget;
- cap all methods by a common wall-clock or forward-equivalent budget;
- include failed and divergent runs in success-rate statistics;
- tune each baseline on development data rather than intentionally using weak
  defaults.

The principal optimizer result should be a distribution across independent
cells and repeated algorithm seeds, not one best trajectory.

## Numerical verification

Before optimizer comparisons:

- reproduce reference voltage traces against the source NEURON implementation
  at frozen parameters;
- verify stimulus timing, units, initial conditions, recording site, and
  morphology discretization;
- perform timestep convergence at, for example, `0.2`, `0.1`, `0.05`, and
  `0.025 ms` on representative candidates;
- verify that conclusions and candidate rankings do not depend materially on
  the chosen timestep;
- compare final fitted candidates in both Jaxley and the source simulator where
  equivalent parameter transfer is possible;
- check autodiff gradients against central finite differences or directional
  derivatives on a reduced model and selected full-model parameters;
- test gradients near spike-count transitions and projected parameter bounds;
- report solver failures, nonfinite states, and bound projections.

Coarse-fidelity CMA evaluations are screening approximations. Every elite must
be reevaluated at the final solver/timestep before ranking or Adam handoff.

## Identifiability and uncertainty

Biophysical inference requires more than optimizer convergence:

- compute the Jacobian of selected voltage/features with respect to fitted
  parameters near each optimum;
- examine singular values, parameter correlations, and sloppy directions;
- run profile likelihoods for parameters central to biological claims;
- use parametric bootstrap or repeated-noise synthetic datasets to quantify
  uncertainty and optimizer variability;
- compare parameter distributions across near-optimal seeds and cells;
- distinguish parameter uncertainty from predictive uncertainty;
- report ensembles of near-optimal models when many parameter vectors predict
  similarly;
- avoid interpreting an individual Na/K conductance if only a correlated
  combination is identifiable;
- use additional protocols or measurements when required to break parameter
  degeneracies.

For each claimed mechanism, define in advance:

```text
claim -> parameter/combination -> observable constraint
      -> identifiability diagnostic -> uncertainty interval
      -> perturbation or held-out prediction
```

When likelihood assumptions are defensible, formulate the observation noise
model explicitly rather than treating an ad hoc weighted loss as a likelihood.
Otherwise, describe LSU_1 as an optimization criterion and do not attach
likelihood-based confidence intervals to it without additional justification.

## Statistical analysis

- Define the biological replicate and sample-size rationale before final data
  collection.
- Use paired comparisons because every optimizer should be evaluated on the
  same cells and partitions.
- Report effect sizes and confidence intervals, not only P values.
- Use hierarchical or mixed-effects analysis when seeds are nested within
  cells and cells are nested within animals/batches.
- Never treat optimizer seeds or repeated sweeps as independent biological
  observations.
- Bootstrap at the biological-unit level, not the trace level.
- Prespecify handling of missing traces, failed simulations, and excluded
  cells.
- Correct or hierarchically organize multiple secondary endpoint comparisons.
- Show all cells and all prespecified seeds in supplementary results.
- Report median, dispersion, failure rate, and tail behavior; the best run
  alone is insufficient.

## Failure handling

- Nonfinite voltages or losses: assign `invalid_loss` and record the reason.
- Simulation exception: record a failed candidate; do not crash the generation.
- Too many invalid candidates: shrink sigma and retry the generation once.
- Boundary crowding: report per-parameter boundary occupancy.
- CMA covariance ill-conditioning: regularize eigenvalues and checkpoint before
  recovery.
- Interrupted Adam stage: use existing durable checkpoints.
- Interrupted CMA stage: restore the last fully completed generation only.

## Reproducibility and audit trail

Every reported result must be reconstructible from:

- immutable raw-data identifiers and checksums;
- preprocessing and segmentation version;
- resolved configuration and compatibility hash;
- source commit, dirty-worktree status, environment lockfile/container, JAX,
  Jaxley, NEURON, compiler, and hardware versions;
- model/morphology/mechanism provenance;
- optimizer seed, candidate ancestry, and exact evaluation budget;
- initial, best, and final parameters with bounds and units;
- full metrics history, failure logs, and generated figures.

Prepare:

- a one-command reproduction workflow for every main figure/table;
- archived code and processed data with persistent identifiers;
- machine-readable metadata and configuration files;
- a minimal CPU smoke reproduction plus documented full-compute workflow;
- deterministic tests in continuous integration;
- a reporting checklist aligned with the experimental study design.

Data or code that cannot be released must have a documented access path and a
synthetic public fixture that exercises the complete pipeline.

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

- hybrid search improves the prespecified primary validation endpoint under a
  matched budget, with an effect size and uncertainty interval;
- improvement is observed across independent cells, not driven by one cell or
  one seed;
- locked-test firing-rate and waveform metrics meet prespecified success
  thresholds;
- optimizer success and failure rates are reported;
- synthetic identifiable parameters and predictions are recovered within
  prespecified tolerances;
- numerical conclusions survive timestep and simulator-parity checks;
- results reproduce from config, seed, and input hashes;
- parameter solutions pass prespecified boundary, sensitivity, and
  identifiability diagnostics;
- uncertainty is reported for biological predictions and claimed parameter
  combinations;
- conclusions are based on locked-test performance, not one favorable seed or
  post-hoc metric.

## Implementation milestones

1. **M0 — Research freeze and benchmark specification:** define claims,
   biological units, partitions, endpoints, baselines, budgets, exclusions, and
   statistical analysis before locked-test evaluation.
2. **M1 — Shared forward evaluator:** refactor without changing current fit
   results.
3. **M2 — Candidate files and imported initialization:** exact handoff into
   Adam.
4. **M3 — CMA ask/tell core:** serial population evaluation, checkpoints, and
   analytic-function tests.
5. **M4 — Hybrid CLI:** CMA elites -> fixed Adam -> backtracking Adam.
6. **M5 — Dataset partitions and physiological report:** within-cell held-out
   sweeps, cell-level validation/test partitions, and discrete metrics.
7. **M6 — Synthetic recovery and numerical verification:** known-parameter,
   noise, model-mismatch, timestep, gradient, and simulator-parity tests.
8. **M7 — Baseline/ablation study:** matched-budget optimizer comparisons on
   development data.
9. **M8 — Slurm population arrays:** atomic generation manifests and resume.
10. **M9 — Performance work:** profile nested `vmap`, candidate microbatches,
   and coarse-to-full fidelity.
11. **M10 — Identifiability and uncertainty:** Jacobian diagnostics, profiles,
    bootstraps, ensembles, and claim-specific evidence.
12. **M11 — Locked test and publication package:** one-time test evaluation,
    statistical report, archived artifacts, and figure/table reproduction.

M0–M7 are required before interpreting optimizer performance. M10 is required
before mechanistic parameter claims. M11 occurs only after methods and analysis
are frozen.

## Recommended first experiment

Before full CMA-ES, establish the baseline:

1. Run 16 seed-jittered `LSU_1_wide_bounds_adam` searches for 50 epochs.
2. Backtracking-refine the best four.
3. Evaluate all four on held-out traces.
4. Record compute cost and parameter diversity.

Then run CMA-ES with the same total evaluation budget on a 12–20 parameter
subset. This determines whether CMA-ES adds value beyond the simpler multistart
system before investing in distributed population evaluation.

This first experiment is a development benchmark, not the final biological
test. Do not use its cells or outcomes as an untouched confirmation set after
using them to revise LSU_1, bounds, parameter subsets, or optimizer settings.

## Methodological references

- Deistler M. et al. *Jaxley: differentiable simulation enables large-scale
  training of detailed biophysical models of neural dynamics*. Nature Methods
  22, 2649–2657 (2025).
  https://doi.org/10.1038/s41592-025-02895-w
- Hansen N. *The CMA Evolution Strategy: A Tutorial* (2016).
  https://arxiv.org/abs/1604.00772
- Raue A. et al. *Structural and practical identifiability analysis of
  partially observed dynamical models by exploiting the profile likelihood*.
  Bioinformatics 25, 1923–1929 (2009).
  https://doi.org/10.1093/bioinformatics/btp358
- Kreutz C. *Profile likelihood in systems biology*. FEBS Journal 280,
  2564–2571 (2013). https://doi.org/10.1111/febs.12276
- Percie du Sert N. et al. *The ARRIVE guidelines 2.0*. PLOS Biology 18,
  e3000410 (2020). https://doi.org/10.1371/journal.pbio.3000410

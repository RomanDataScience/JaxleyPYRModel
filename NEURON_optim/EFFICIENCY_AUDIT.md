# NEURON optimization efficiency audit

Date: 2026-09-22

Scope: `NEURON_optim/`, including candidate evaluation, NEURON/Jaxley
simulation setup, objective calculation, plotting, checkpoints, and stage
orchestration.

The audit combined local code inspection with an independent read-only review.
Impact and confidence below are qualitative: **High** means the issue is
directly visible in the code path and likely material for production runs.

## Implemented in this change

### Simulation results are reused by plotting

`neuron_optim/stages.py` now returns each candidate's simulation time and
voltage arrays from the evaluation call. Each generation writes:

- `population_generation_XXXX.npz`: normalized candidate parameters and losses;
- `simulations_generation_XXXX.npz`: candidate parameters, losses, validity
  flags, and per-candidate/per-trace simulation arrays.

Plotting consumes those in-memory results, so it does not simulate the top
candidates a second time. The persisted archive also makes the evaluated
parameters and simulator outputs available for later analysis.

### Multi-worker plotting no longer starts a simulator pool

One `ProcessPoolExecutor` is created per study and reused across generations.
Plotting happens in the parent process from cached outputs. The optimizer state,
population archive, and generation history are committed before plotting, and
plotting exceptions are logged and do not force the generation to be rerun.

### Serial simulator construction is reused

Serial studies construct one simulator wrapper per study instead of repeating
the wrapper and mechanism/source-cache checks for every candidate. Worker-local
simulators continue to be initialized once per worker.

### Completed and partial-study resume is safe

If a compatible checkpoint already reached the configured generation count and
`best_parameters.json` exists, `run_study()` returns the saved result rather
than entering an empty loop and raising `Study produced no generations`.
The best-so-far state is also written after improvements and restored when a
partial study resumes.

## Remaining findings

### 1. A fresh model is built for every candidate

- Location: `neuron_optim/simulator.py:212-214`, `neuron_optim/jaxley_simulator.py:63-77`
- Impact: High. Morphology construction and parameter application traverse the
  complete model for every parameter vector.
- Confidence: High.
- Recommendation: Keep a worker-local model and reset its state between
  candidates if the backend permits it. Cache static morphology, segment
  groups, and mechanism handles; apply only candidate-dependent values.
- Risk: NEURON global state and Jaxley functional parameter semantics need
  explicit reset tests before changing lifecycle behavior.

### 2. Objective preprocessing is repeated for every candidate

- Location: `neuron_optim/objective.py:113-158`, `196-248`
- Impact: Medium to high. Experimental baselines, region masks, experimental
  spike detections, terminal masks, and median time steps are invariant within a
  study but are recomputed for each objective call.
- Confidence: High.
- Recommendation: Build an immutable objective context once per study/worker
  containing masks, centered experimental voltages, experimental spikes, `dt`,
  and terminal indices.

### 3. Spike detection recomputes time-step statistics inside loops

- Location: `neuron_optim/objective.py:41-50`, `168-172`
- Impact: Medium. `np.median(np.diff(time))` is repeated for threshold
  crossings and matched-spike windows.
- Confidence: High.
- Recommendation: Precompute `dt` and reusable window offsets. Consider a
  single-pass detector for noisy long traces.

### 4. Trace loading repeats disk reads

- Location: `neuron_optim/data.py:55-60`, `90-120`; `neuron_optim/stages.py:349`
- Impact: Medium. Time vectors are used to determine windows and then loaded
  again. Stage-2 studies also load the same four traces for every basin/seed.
- Confidence: High.
- Recommendation: Read each vector once and construct the prepared `Trace`
  directly. Cache immutable prepared traces by data/config hash at the outer
  pipeline level.

### 5. Optuna checkpoints rewrite the full study every generation

- Location: `neuron_optim/cma.py:147-179`, called from
  `neuron_optim/stages.py`
- Impact: Medium to high as trial counts grow. The full `study.pkl` is
  serialized and replaced after every generation, and completed trials are
  rescanned to derive generation counts.
- Confidence: High.
- Recommendation: Benchmark a compact versioned optimizer-state checkpoint or
  configurable checkpoint interval. Preserve atomic replacement and explicit
  resume guarantees.

### 6. Default plot volume is potentially very large

- Location: `neuron_optim/plotting.py:34-67`; plotting defaults in the YAML
  configurations.
- Impact: High for the full basin workflow. Ten multi-panel PNGs per
  generation can dominate filesystem usage and wall time.
- Confidence: High.
- Recommendation: Add `plot_every`, a generation-best-only mode, or a deferred
  plotting command. Keep the compact simulation archives as the primary
  diagnostic artifact.

### 7. Independent studies are serialized

- Location: `neuron_optim/stages.py:484-516`
- Impact: High for many stage-1 seeds and stage-2 basin/seed combinations.
- Confidence: High.
- Recommendation: Add an explicit study-array mode for cluster/job-array use,
  or parallelize independent studies at the outer level while setting inner
  candidate workers to one. Avoid oversubscribing both levels.

### 8. Candidate diagnostics are transported for the whole population

- Location: `neuron_optim/stages.py:48-63`, `276-296`
- Impact: Medium. Detailed objective dictionaries for every candidate are
  pickled back to the parent even though only the current best is retained.
- Confidence: High.
- Recommendation: Return scalar losses for normal generations and request full
  diagnostics only for the best/top-k candidates, unless per-candidate details
  are intentionally needed for analysis.

## Recommended follow-up order

1. Cache/reuse the model structure inside each worker, with lifecycle tests.
2. Precompute objective and trace-derived features once per study.
3. Add a regression test for partial-resume best-state restoration.
4. Add configurable plot frequency and a deferred plot command.
5. Benchmark compact checkpoints against the current Optuna pickle workflow.
6. Add an outer study-array launcher for independent seeds and basins.

## Verification

The package compiles, and the test suite passes in the documented `Jaxley`
environment:

```text
12 passed in 12.27s
```

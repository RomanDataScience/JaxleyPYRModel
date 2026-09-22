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

### Models, objective features, and trace loading are reused

Both simulator backends now retain their morphology/model within a worker and
reset simulation state between candidates. Objective contexts precompute trace
masks, baselines, experimental spikes, terminal masks, and time steps once per
study. Protocol loading reads each time/voltage/current vector once before
windowing and constructing a `Trace`.

Candidate evaluation returns scalar losses and simulation arrays; detailed
objective dictionaries are reconstructed only for a newly selected best
candidate.

### Independent studies can run at an outer level

`runtime.study_workers` parallelizes independent stage-1 seeds and stage-2
basin/seed studies. When enabled, each child study uses one candidate worker so
the two parallelism levels do not oversubscribe the machine.

### Plot output is consolidated and throttled

Each plotted generation now writes one `candidates.png`: candidate rows and
trace columns form a single multi-panel figure. `plotting.every` controls the
generation interval, while the final generation is always plotted.

### Completed and partial-study resume is safe

If a compatible checkpoint already reached the configured generation count and
`best_parameters.json` exists, `run_study()` returns the saved result rather
than entering an empty loop and raising `Study produced no generations`.
The best-so-far state is also written after improvements and restored when a
partial study resumes.

## Remaining findings

The main findings are implemented. The one deliberate exception is the
checkpoint representation: the default still serializes Optuna's complete
study because that is the mechanism that preserves exact sampler state and
therefore exact resumed trajectories. Two lower-risk improvements are now in
place: generation counters no longer scan every completed trial after each
`tell()`, and `checkpoint_every` can reduce serialization frequency when the
user accepts replaying uncheckpointed generations after interruption.

The model-reuse change was numerically checked against the old fresh-model path
using stored candidates from the existing runs:

| Backend | Artifact | Samples | Maximum voltage delta | Maximum time delta |
| --- | --- | ---: | ---: | ---: |
| Jaxley | `runs/jaxley_seeded_calibration/.../population_generation_0001.npz` | 13,000 | 0.0 mV | 0.0 ms |
| NEURON | `runs/smoke_full/.../population_generation_0001.npz` | 13,000 | 0.0 mV | 0.0 ms |

The comparison used the same candidate and trace with `reuse_model=False` and
`reuse_model=True`.

## Recommended follow-up order

1. Benchmark compact optimizer checkpoints while preserving exact-resume mode.
2. Add a regression test for partial-resume best-state restoration.
3. Benchmark `runtime.study_workers` on the target machine and avoid native
   backend oversubscription.

## Verification

The package compiles, and the test suite passes in the documented `Jaxley`
environment:

```text
13 passed in 12.06s
```

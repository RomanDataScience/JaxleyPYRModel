# NEURON Combe two-stage CMA-ES optimization plan

## Status

This is a design-only plan for review. The implementation should not start until
the interpretation of the 100 basins and the parameter set is approved.

## Goal

Create a self-contained optimization pipeline under `NEURON_optim/` that:

1. Uses the original Combe2023 model through NEURON, including its HOC
   morphology and compiled MOD mechanisms.
2. Fits the four hyperpolarizing recordings (`v75ctrl`, `v76ctrl`, `v77ctrl`,
   and `v78ctrl`) with one scalar CMA-ES objective that makes simulated voltage
   (`v_sim`) match measured membrane voltage (`v_memb`), with extra emphasis on
   the hyperpolarizing step and recovery afterward.
3. Runs the hyperpolarizing search for 10 deterministic seeds, 200 generations,
   and 30 offspring per generation.
4. Retains 100 stage-1 parameter vectors as initialization basins.
5. Starts a separate depolarizing optimization from every basin. Each run
   evaluates all four matching depolarizing recordings, uses a 15% basin
   perturbation for its initial point, and runs 10 seeds, 200 generations, and
   30 offspring per generation.

The existing Jaxley implementation remains useful as a reference for parameter
names and bounds, but it must not be the production simulator for this folder.

The pipeline will run inside the repository's existing `Jaxley` environment.
NEURON, Jaxley, NumPy, YAML, and the compiled Combe MOD mechanisms will be
validated in that environment. No new environment will be created by
`NEURON_optim`; the production launcher will use `conda run --name Jaxley` or
an equivalent activated environment.

## Existing inputs to reuse

- NEURON model source: `Combe2023.zip`, extracted or referenced without
  changing the archive.
- NEURON model loader: `channels_converted/modelComparison/neuron_model.py`.
- Compiled mechanism source/build area: `channels_converted/mod/` and the
  existing `nrnivmodl` workflow.
- Combe parameter names, defaults, and bounds in
  `JaxleyModel/model/model_Combe.py`.
- Experimental segmented traces and epoch metadata in
  `JaxleyModel/Experimental_currentClamp_Analysis/Segmented_Traces/`.

The four signals are paired by trace name. Their metadata supplies the pulse
onset, offset, current waveform, voltage waveform, time grid, and trial end.
The implementation must resolve paths relative to the repository, not rely on
absolute paths embedded in `segment_metadata.csv`.

## Proposed package layout

```text
NEURON_optim/
├── PLAN.md                         # this design
├── README.md                       # environment, compilation, and commands
├── pyproject.toml                  # minimal Python dependencies
├── configs/
│   ├── hyperpolarizing.yaml       # stage 1 and basin extraction settings
│   └── depolarizing.yaml          # stage 2 settings and 15% initialization
├── neuron_optim/
│   ├── __init__.py
│   ├── config.py                   # strict YAML/config validation
│   ├── data.py                     # metadata and v/i/t loading
│   ├── mechanisms.py               # private cached MOD build with kinetic scales
│   ├── parameters.py               # Combe names, bounds, and HOC application
│   ├── simulator.py                # isolated NEURON build, replay, recording
│   ├── objective.py                # weighted voltage-shape objectives
│   ├── cma.py                      # bounded ask/tell CMA-ES and checkpoints
│   ├── basins.py                   # deterministic 100-basin extraction
│   ├── plotting.py                 # best-candidate trace figures per generation
│   ├── stages.py                   # stage-1/stage-2 orchestration
│   └── cli.py                      # validate, run, resume, report commands
├── tests/
│   ├── test_data.py
│   ├── test_parameters.py
│   ├── test_objective.py
│   ├── test_cma_checkpoint.py
│   ├── test_basin_selection.py
│   └── test_neuron_smoke.py
└── runs/                           # generated outputs; never source inputs
```

## Model and parameter handling

### NEURON-only simulation path

Each candidate evaluation will:

1. Build a fresh Combe NEURON cell, or reset a worker-local cell to a clean
   state before `finitialize`.
2. Apply the candidate's physical parameter vector to the corresponding HOC
   passive properties and channel mechanism variables.
3. Replay the experimental current waveform at the soma with a NEURON
   `IClamp` and `Vector.play`, after converting the recorded current from pA to
   nA.
4. Record time and soma voltage at the same `dt` as the prepared experimental
   trace.
5. Return finite NumPy arrays for `time`, `v_sim`, and the replayed current.

The current must be replayed from the data rather than replaced with a single
idealized square pulse. This preserves the recorded pulse amplitude and any
sample-level onset/offset behavior. The exact NEURON model setup must match the
Combe CCh-driven setup already used by
`build_combe_neuron_model(quiet=True, d_lambda=...)`.

### Parameter vector

The default fit set will be all 44 numeric Combe parameters already exposed by
the repository:

- conductance parameters;
- passive parameters;
- kinetic time-scale parameters.

The config will allow explicit inclusion/exclusion so a smaller smoke run can
be used before the production search. Every parameter will have a finite
physical lower and upper bound taken from the existing Combe catalog. CMA-ES
will operate in normalized `[0, 1]` coordinates and convert to physical values
only at the NEURON boundary. Candidates will be clipped or resampled so no
candidate is applied outside its configured bounds.

The NEURON parameter application layer is a required implementation item: the
existing `set_fitted_parameters` function is Jaxley-oriented and cannot be
assumed to update a NEURON cell. It will be tested by changing one parameter at
a time and verifying the expected HOC segment/mechanism values.

### Initial state and reset

The default initial voltage will be the measured voltage at the beginning of
each simulation window. This avoids using information after the stimulus while
allowing the model to settle for the requested 800 ms before the step. The
choice will be configurable between `observed_first_sample` and the Combe
resting value (`Epas`). A failed or non-finite NEURON run will receive a finite
large objective penalty and will not advance the optimizer with NaN/Inf.

## Trace windows and alignment

For each hyperpolarizing trace:

- detect the current-step onset from the metadata/current waveform;
- set `simulation_start = step_onset - 800 ms`;
- set `simulation_stop = trial_end`;
- crop both the measured voltage and current to that interval;
- use the trace's recorded time grid as the comparison grid, or linearly
  interpolate the simulation output onto it when necessary.

For each depolarizing trace, use the full available trial window by default so
the objective can include baseline, the depolarizing step, and post-step
behavior. The depolarizing window will be configurable independently, but its
four traces must use the same convention within a run.

The objective will never compare arrays with an implicit index shift. It will
validate monotonic time, uniform sampling (or explicitly resample once), equal
length after alignment, finite values, and the current-unit conversion.

## Stage 1: hyperpolarizing optimization

### Objective

For each of the four hyperpolarizing traces, calculate a weighted voltage-shape
similarity kernel between `v_sim(t)` and `v_exp(t)` (where `v_exp` is the
recorded membrane voltage, also referred to as `v_memb`):

```text
e(t) = v_sim(t) - v_exp(t)
similarity(t) = exp(-(e(t)^2) / sigma_hyper_mV^2)
L_region = 1 - weighted_mean(similarity(t))
L_trace = (w_pre * L_pre + w_step * L_step + w_recovery * L_recovery)
           / (w_pre + w_step + w_recovery)
L_hyper = mean(L_trace over the four traces)
```

The raw exponential is highest when the traces match, so `1 - similarity` is
the minimized loss supplied to CMA-ES. This keeps the requested exponential
form while preserving a conventional minimization direction. The weighted mean
will be normalized within each time region so the long pre-step period does
not overwhelm the response. The default regions are:

| Region | Interval | Relative emphasis |
| --- | --- | ---: |
| Pre-step | onset - 800 ms through onset | 1.0 |
| Hyperpolarizing step | onset through offset | 4.0 |
| Recovery | offset through trial end | 3.0 |

The exact weights, voltage scale, and optional robust loss will live in the
config. The implementation will save region-wise losses as well as the scalar
objective so it is clear whether a candidate improved the step, recovery, or
baseline. Each trace contributes equally after its regional averages are
computed.

The exponential width `sigma_hyper_mV` is a required fixed configuration value
and will be recorded in the run manifest. It will not be estimated from the
candidate population.

### Hyperpolarizing spike penalty

Stage 1 will include a hard spike constraint. A spike detector with a
configurable voltage threshold, refractory period, and optional minimum
prominence will be applied to each simulated hyperpolarizing trace. If any
spike is detected inside the hyperpolarizing stimulus interval
`[step_onset, step_offset]`, the candidate will receive the configured finite
`hyper_spike_penalty` instead of its ordinary objective value. The result will
also record the detected spike times and the penalty reason. Spikes outside the
step remain visible in diagnostics but do not trigger this specific constraint.

### CMA-ES budget and seeds

Stage 1 will run 10 independent CMA-ES studies with seeds `0` through `9` by
default. Each study has:

- 200 generations;
- population size 30 (offspring per generation);
- a documented initial mean, initial sigma, parent fraction, and boundary
  policy;
- a checkpoint after every fully evaluated generation.

The exact total is 6,000 candidate evaluations per seed and 60,000 candidates
for stage 1, with each candidate requiring four NEURON simulations.

Stage 1 will evaluate the 30 offspring in parallel using 6 worker processes by
default. Workers will use process isolation rather than shared threads because
NEURON's global HOC state and section registry must not be shared between
independent candidate evaluations. The worker count will be configurable in
the YAML and overridable from the CLI, with `parallel_workers: 6` as the
production default.

The final generation population, its losses, the best-so-far candidate, and the
final CMA state will all be retained. This is important because basin creation
uses multiple good endpoints rather than only one winner per seed.

## Creating the 100 initial basins

The default basin policy will retain the ten best candidates from the final
generation of each of the ten stage-1 seeds:

```text
10 seeds × 10 retained candidates per seed = 100 basins
```

Each basin record will contain:

- a stable basin ID;
- source seed and source generation;
- normalized and physical parameter vectors;
- total and region-wise hyperpolarizing losses for all four traces;
- rank within its source seed;
- model/config/data hashes.

The default policy treats a basin as a high-quality initialization point, even
when two points are close. Exact duplicate vectors will be removed and
backfilled from the next ranked final-generation candidate so the output still
contains exactly 100 records. A later optional diversity policy can be added,
but it should not silently change the requested count.

## Stage 2: depolarizing optimization from every basin

For every one of the 100 basin records and every one of 10 stage-2 seeds, start
an independent CMA-ES run whose initial mean is a perturbed version of that
basin. The default perturbation is uniform in normalized coordinates:

```text
start = clip(basin + U(-0.15, +0.15), 0, 1)
```

The perturbation RNG seed will be derived deterministically from
`(stage2_seed, basin_id)` and saved in the run manifest. This makes the 15%
variation reproducible and gives each seed a distinct start. In physical
coordinates this means 15% of each parameter's configured bound range, which
also gives zero-valued parameters a meaningful perturbation.

Each stage-2 optimization evaluates all four depolarizing traces with one
single scalar objective. The objective will be a configurable composite loss:

```text
L_depolarizing =
    w_trajectory * L_trajectory
  + w_spike_shape * L_spike_shape
  + w_plateau * L_plateau
  + w_return_baseline * L_return_baseline
```

All four components will be calculated per trace and averaged across the four
traces, so one recording cannot dominate solely because it has more samples.
The component values and their weighted contributions will be saved for every
candidate and for every final solution.

The stage-2 terms use the following notation. For each component-specific
width `sigma_*`, define the pointwise minimized discrepancy

```text
D_sigma(a, b) = 1 - exp(-((a - b)^2) / sigma_*^2)
```

and, for a sample mask `M` with optional nonnegative sample weights `w(t)`,

```text
mean_D(M, a, b) = sum(t in M) w(t) * D_sigma(a(t), b(t))
                  / sum(t in M) w(t)
```

Thus every `L_*` below is finite, nonnegative, and minimized when the relevant
simulated and experimental quantities agree. The widths
`sigma_trajectory_mV`, `sigma_spike_mV`, `sigma_plateau_mV`,
`sigma_recovery_mV`, and `sigma_terminal_mV` are fixed config values recorded
in the manifest.

### Depolarization trajectory term

`L_trajectory` will retain the region-normalized voltage-shape comparison over
the complete depolarizing trial, with configurable baseline, current-step, and
post-step weights. This term preserves time-locked agreement of the overall
trace, including spike timing and the voltage between detected events.

If `M_trajectory` is the full-trial mask and `w_region(t)` is the configured
region weight, then:

```text
L_trajectory = mean_D(M_trajectory, v_sim(t), v_exp(t))
```

using `sigma_* = sigma_trajectory_mV`. Sample weights are normalized within the
baseline, current-step, and recovery regions before they are combined.

### Spike-shape term during depolarization

`L_spike_shape` will compare the waveform of each experimentally detected AP
with the corresponding simulated AP during the depolarizing interval. For
each ordinally matched spike, the implementation will:

1. Extract a configurable window around the spike, using the detected peak or
   threshold crossing as the alignment anchor.
2. Interpolate both waveforms onto a common relative-time grid.
3. Compare the aligned voltage shape, including the rising phase, peak,
   repolarization, and afterhyperpolarizing portion within the configured
   window.

Unmatched experimental or simulated spikes will add a finite configured
`unmatched_spike_penalty` to this component. The global, time-locked trajectory
term remains active, so alignment for this local shape term does not hide
incorrect spike timing. The detector threshold, refractory period, matching
window, relative-time grid, and component weight will be configurable.

For matched spike pairs `p` and relative-time samples `tau` in the spike
window `W_p`, the term is:

```text
L_spike_shape = mean_D((p, tau) in matched_spikes,
                       v_sim,p(tau), v_exp,p(tau))
                 + unmatched_spike_penalty * N_unmatched
```

The mean uses `sigma_* = sigma_spike_mV`. If no spikes can be matched, the
finite unmatched-spike term is used rather than an undefined mean.

### Depolarization plateau term

`L_plateau` will measure voltage agreement during the applied depolarizing
current step, including the depolarized plateau between spikes. To prevent the
short AP waveform from dominating this term, samples within a configurable
exclusion window around each detected spike will be masked (default: a small
window around the detected peak and immediate repolarization). The remaining
step samples will be compared time-locked between `v_sim` and `v_exp`.

If `M_plateau` is the depolarization-step mask after removing the configured
spike exclusion windows from the union of experimental and simulated spike
times, then:

```text
L_plateau = mean_D(M_plateau, v_sim(t), v_exp(t))
```

using `sigma_* = sigma_plateau_mV`. A missing plateau mask uses the configured
`invalid_plateau_penalty`.

The plateau term will preserve subthreshold depolarization, inter-spike
voltage, and post-spike settling during the current step. If the trace has no
valid plateau samples after masking, it will receive a finite configured
invalid-plateau penalty rather than NaN/Inf.

### Return-to-baseline term after depolarization

`L_return_baseline` will explicitly quantify the comeback toward the
pre-step baseline after the current step ends. The baseline will be estimated
from the configured pre-step window. The term will compare the simulated and
measured voltage over the post-step recovery interval and will include:

- normalized post-step trajectory error;
- terminal return-to-baseline error at the end of the available trial;
- optionally, recovery latency or slope when the configured data window is
  long enough.

This explicit term prevents a good in-step fit from compensating for a model
that remains depolarized or returns too quickly after the current is removed.
The post-step interval, terminal window, baseline window, scale, and weight
will be configurable.

Let `M_recovery` be the post-step mask, `M_terminal` its final configured
window, and `b_exp` the measured pre-step baseline. The default return term is:

```text
L_recovery_trajectory = mean_D(M_recovery, v_sim(t), v_exp(t))
L_terminal = 1 - exp(-((mean(v_sim[M_terminal]) - b_exp)^2)
                      / sigma_terminal_mV^2)
L_return_baseline = alpha_return * L_recovery_trajectory
                   + (1 - alpha_return) * L_terminal
```

`L_recovery_trajectory` uses `sigma_* = sigma_recovery_mV`. The terminal term
explicitly rewards the simulated voltage coming back to the measured baseline,
while the trajectory term preserves the experimentally observed recovery
course. `alpha_return`, the recovery mask, and terminal window are fixed
configuration values.

The initial configuration will expose all four weights rather than hiding them
inside one regional weight. Suggested starting values are
`w_trajectory = 1`, `w_spike_shape = 3`, `w_plateau = 2`, and
`w_return_baseline = 3`; these are defaults for review and can be changed
before production runs.

### Depolarizing out-of-window spike penalty

Stage 2 will use the same deterministic spike detector. For each depolarizing
trace, define the allowed spike window as:

```text
[step_onset - 50 ms, step_offset + 50 ms]
```

Any detected spike outside that window will be counted as an out-of-window
spike. If a trace contains more than one such spike, the candidate will receive
the configured finite `depolarizing_extra_spike_penalty` instead of its
ordinary objective value. One or zero out-of-window spikes will not trigger
this particular penalty, although all spike times and counts will be retained
in the diagnostics. The penalty decision will be made independently per trace;
the candidate is penalized if any of the four depolarizing traces violates the
constraint.

Both spike penalties will be configurable, deterministic, and large enough to
dominate ordinary voltage-shape differences without using infinity or NaN.
The unpenalized loss, penalty name, offending trace, allowed interval, and
detected spike times will be saved for every penalized candidate. The penalty
rules will be applied before `CMAES.tell()` so every offspring still supplies
one finite scalar loss.

Every stage-2 run has:

- 10 seeds;
- 200 generations;
- 30 offspring per generation;
- 6,000 candidate evaluations;
- four NEURON simulations per candidate.

The literal requested workload is therefore 100 × 10 = 1,000 stage-2 studies,
or 6,000,000 candidate evaluations and 24,000,000 NEURON simulations. The
runner must support resumable batch execution and array-job scheduling rather
than requiring all studies in one process.

Within each stage-2 study, the 30 offspring will likewise be evaluated by the
default pool of 6 NEURON worker processes. The generation remains a strict
barrier: all 30 results must return, be validated, and be written before CMA-ES
updates its state. Across the 1,000 stage-2 studies, separate studies may also
be distributed as independent array tasks; the per-study worker count must be
configured to avoid oversubscribing a node. A `--workers` override and an
array-task mode will be included in the CLI.

## Checkpointing and execution

The implementation will support both local execution and one-study-per-task
HPC/SLURM execution. A study is allowed to advance only after all 30 offspring
in a generation have finite objective values and have been written to disk.
Interrupted generations are rerun from the last completed checkpoint.

Each study directory will include:

```text
manifest.json              # versions, hashes, seed, basin, config
parameters.json            # parameter names, bounds, initial mean
generations.jsonl          # generation summary and best candidate
population_generation_*.npz
cma_checkpoint_*.npz       # serializable CMA state and RNG state
best_parameters.json
best_traces.npz            # measured and simulated traces for the winner
loss_breakdown.json
```

When `plotting.enabled` is true (the default), each completed generation also
writes `plots/generation_XXXX/rank_XX.png` for the 10 lowest-loss candidates.
Each figure overlays measured and simulated voltage for all four traces and
marks the stimulus interval. Plot capture runs after ordinary offspring
evaluation and does not alter CMA-ES state. It adds four NEURON simulations
per plotted candidate and can be disabled for performance-focused diagnostic
runs.

The top-level run will additionally contain `basins.jsonl`, a stage-1 index,
and a stage-2 index mapping each `(basin_id, seed)` to its output directory.
Writes will be atomic where practical, and manifests will reject accidental
resume with a different model, data, parameter, objective, or budget hash.

## CLI and configuration

The planned commands are:

```bash
python -m neuron_optim.cli validate --config configs/hyperpolarizing.yaml
python -m neuron_optim.cli run-hyper --config configs/hyperpolarizing.yaml
python -m neuron_optim.cli make-basins --run-dir runs/<stage1-run>
python -m neuron_optim.cli run-depolarizing \
  --config configs/depolarizing.yaml \
  --basins runs/<stage1-run>/basins.jsonl
python -m neuron_optim.cli report --run-dir runs/<run>
```

Configuration validation will check the four expected traces, units, windows,
all parameter bounds, `generations: 200`, `population_size: 30`,
`parallel_workers: 6`, and the seed counts. Smoke overrides may reduce the
budget or worker count only when explicitly supplied on the command line;
production configs retain the requested values.

## Verification plan

Before a production run:

1. Compile/load the Combe MOD mechanisms in headless NEURON mode.
2. Run one unmodified candidate on all eight traces and verify finite output,
   correct current replay, expected time range, and no cross-trace state leak.
3. Compare the NEURON voltage from the new simulator with the existing
   `run_neuron_step` path for an equivalent square-pulse smoke protocol.
4. Perturb one applied parameter and verify that the resulting voltage changes
   and that the HOC mechanism value changed as expected.
5. Run a tiny 2-generation, 3-offspring CMA smoke study and verify that resume
   reproduces the next population from the checkpoint.
6. Run a parallel smoke study with six workers and verify that worker-local
   NEURON state does not leak across candidates and that results are identical
   to serial evaluation for the same candidate list and seeds.
7. Verify that stage 1 creates exactly 100 basin records with reproducible
   ordering.
8. Run one basin and one stage-2 seed on all four depolarizing traces and
   verify that the 15% initialization and study metadata are correct.
9. Inject a synthetic hyperpolarizing candidate with a spike during the step
   and verify that `hyper_spike_penalty` is applied.
10. Inject synthetic depolarizing traces with zero, one, and two spikes outside
   `[step_onset - 50 ms, step_offset + 50 ms]` and verify that only the two-spike
   case receives `depolarizing_extra_spike_penalty`.
11. Verify the depolarizing objective components independently using synthetic
   traces: altered AP waveform, altered inter-spike plateau, and altered
   post-step baseline recovery must increase only their corresponding terms.
12. Verify that unmatched simulated/experimental spikes produce the configured
   finite `unmatched_spike_penalty`.
13. Run the full budgets only after the smoke outputs and objective plots have
   been reviewed.

## Review decisions needed before implementation

1. Confirm that the four signals are the `v75ctrl`–`v78ctrl` traces for the
   currently selected cell (`m20240527cd` by default), rather than a different
   cell or a train/validation subset.
2. Confirm the basin interpretation used here: top 10 final-generation
   candidates from each of 10 stage-1 seeds. If “100 basins” means 100 separate
   stage-1 seeds instead, the stage-1 seed budget must change.
3. Confirm that all 44 Combe parameters should be optimized, or provide the
   intended subset.
4. Confirm whether `v_memb` means the experimental recorded voltage (the
   assumption in this plan) and whether the initial voltage should default to
   the first measured sample or `Epas`.
5. Confirm the proposed default regional weights (1, 4, 3) or provide desired
   weights for pre-step, stimulus, and recovery.
6. Confirm `sigma_hyper_mV` and the stage-2 component widths/weights. The
   suggested stage-2 weights are trajectory `1`, spike shape `3`, plateau `2`,
   and return-to-baseline `3`.

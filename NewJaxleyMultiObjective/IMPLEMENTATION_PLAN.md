# Jaxley multi-objective MOCMA pipeline

## Scope

Build a new, self-contained pipeline under `NewJaxleyMultiObjective/` for the
existing Combe2023/Jaxley model and the existing segmented current-clamp traces.
The only search algorithm will be OptunaHub's `MoCmaSampler` (s-MO-CMA-ES):

<https://hub.optuna.org/samplers/mocma/>

The pipeline will define and execute every candidate with Jaxley, measure the
requested experimental and simulated features automatically, return one
minimization value per requested feature, and save the complete Pareto front.

Explicitly out of scope:

- Adam, gradient descent, local refinement, or any other post-MOCMA optimizer;
- the existing single-objective CMA-ES implementation or the CMA-ES/Adam
  hybrid pipeline;
- NSGA-II, TPE, random, grid, or fallback samplers;
- a scalar weighted score, knee-point selection, or automatic choice of one
  “best” Pareto solution;
- hypervolume-history plots or other secondary optimization stages.

The Pareto-front annotations described below are reporting only. They do not
change the sampler or select a final model.

## Existing code to reuse

The implementation should reuse, rather than duplicate, these interfaces:

- `Jaxley_refactored/jaxley_refactored/models/builder.py` for building the
  static Jaxley cell, mechanisms, morphology, and fitted parameter catalog;
- `Jaxley_refactored/jaxley_refactored/data/segmented_traces.py` for loading
  voltage/current/time triplets and manifest epoch metadata;
- `Jaxley_refactored/jaxley_refactored/data/batching.py` for static-shape
  `TraceBucket` objects and protocol windows;
- `Jaxley_refactored/jaxley_refactored/simulation/kernel.py` for the Jaxley
  `SimulationKernel`, initial states, `jax.jit`, and batched execution;
- `Jaxley_refactored/jaxley_refactored/parameters/space.py` and the model's
  `ParameterSpec` values for physical bounds and parameter reporting;
- the existing run/provenance conventions in
  `Jaxley_refactored/jaxley_refactored/reporting/runs.py`.

The feature extractor itself should operate on NumPy arrays after a Jaxley
forward pass. This is appropriate for hard spike and event measurements and
keeps nondifferentiable reporting code out of Jaxley's simulation kernels.
The existing `extract_ipfx_fi_passive_neurons.py` CSV output is a regression
reference for experimental values, not a runtime dependency of the MOCMA
objective.

## Proposed layout

```text
NewJaxleyMultiObjective/
├── IMPLEMENTATION_PLAN.md       # this design
├── README.md                    # setup and run instructions
├── pyproject.toml               # package plus optuna/optunahub dependencies
├── configs/
│   └── mocma_features.yaml      # complete standalone run configuration
├── newjaxley_multiobjective/
│   ├── __init__.py
│   ├── config.py                # strict pipeline-specific config validation
│   ├── features.py              # shared automatic feature extraction
│   ├── objectives.py             # experimental targets and feature errors
│   ├── runner.py                # Jaxley build, simulation, and study loop
│   ├── pareto.py                # front extraction and annotations
│   ├── plotting.py              # full traces plus 100 ms spike zooms
│   └── cli.py                   # validate/run/plot-front commands
└── tests/
    ├── test_features.py
    ├── test_objectives.py
    ├── test_pareto_reporting.py
    ├── test_plotting.py
    └── test_mocma_smoke.py
```

The package should import the refactored project by a documented editable
install or `PYTHONPATH`; it should not copy the model or channel definitions
into the new folder.

## Configuration contract

Start with one complete YAML file, without relying on the existing hybrid
configuration's search section. It should retain the model, protocol, dataset,
runtime, and output settings needed by the Jaxley refactor and add a dedicated
multi-objective section similar to:

```yaml
multi_objective:
  sampler: mocma
  seed: 1234
  population_size: 40
  trials: 4000
  storage: auto             # SQLite study.db in the run directory
  resume: true
  heartbeat_interval_s: 60
  grace_period_s: 300
  study_name: mocma_features

  parameters:
    source: model.parameters.fit
    # Every selected parameter becomes a non-conditional FloatDistribution.

  feature_detection:
    threshold_mV: -20.0
    refractory_ms: 2.0
    smoothing_ms: 0.2
    baseline_window_ms: 30.0
    fahp_search_ms: 20.0
    post_step_analysis_ms: 700.0
    hyper_peak_window_ms: 20.0
    hyper_steady_state_fraction: 0.20

  objectives:
    scale_policy: explicit
    invalid_feature_penalty: 1.0e6
    overlap_sigma_mV: 3.0
    overlap_window_weights:
      baseline: 1.0
      stimulus: 1.0
      recovery: 1.0
      first_100ms_spiking: 2.0
    # Eleven entries, one for each requested feature/objective, all with
    # direction minimize.
    features: [resting_membrane_potential, average_fahp,
               time_mahp, amplitude_mahp, ap_count, last_stabilized_isi,
               last_ap_half_width, peak_input_resistance,
               steady_state_input_resistance, sag, trajectory_overlap]
```

Important validation rules:

1. `sampler` must be exactly `mocma`.
2. The selected model parameters must be numeric, unconditional, and have
   finite lower/upper bounds. Their names and physical bounds must come from
   the existing parameter catalog.
3. The number of objective directions must equal the number of feature labels,
   and every direction must be `minimize`.
4. Detection windows must be physically valid for every selected trace. A
   trace-specific missing event produces a finite configured penalty and a
   validity flag; it must not produce NaN/Inf and silently corrupt an Optuna
   trial.
5. Feature scales must be fixed before optimization and written to the run
   manifest. They must not be recomputed from the candidate population.

Use `optuna.distributions.FloatDistribution(low, high)` for each parameter and
load the sampler through OptunaHub:

```python
module = optunahub.load_module("samplers/mocma")
sampler = module.MoCmaSampler(
    search_space=search_space,
    popsize=config.multi_objective.population_size,
    seed=config.multi_objective.seed,
)
study = optuna.create_study(
    study_name=config.multi_objective.study_name,
    directions=["minimize"] * len(objective_labels),
    sampler=sampler,
    storage=config.multi_objective.storage,
    load_if_exists=config.multi_objective.storage is not None,
)
```

This follows the MOCMA API's documented limitation that non-conditional
numerical parameters are the parameters handled by MOCMA; no categorical or
conditional parameter should enter this study.

## Automatic feature extraction

Implement one detector and one set of window conventions for both experimental
and simulated voltages. That makes target and candidate values comparable and
avoids using the experimental trace to define candidate event locations.

For every trace, first derive:

- `baseline_voltage_mV`: median voltage in the final configured baseline
  window immediately before `epoch_start_ms`;
- `stimulus_mask`: `epoch_start_ms <= t <= epoch_stop_ms`;
- `recovery_mask`: `t > epoch_stop_ms` through the configured post-step
  analysis window;
- the centered current step, using the median current in the same baseline
  window and the median current during the protocol epoch.

### Depolarizing features

Use a threshold-crossing detector with configurable threshold, smoothing, and
refractory interval. Refine threshold and peak times by local interpolation so
the measurements are not limited to one sample. Apply the same detector to
experimental and simulated traces.

The ten requested feature objectives and definitions are:

| Objective label | Automatic measurement | Validity/aggregation |
| --- | --- | --- |
| `resting_membrane_potential` | Baseline-window median voltage | Mean normalized squared error over all selected traces; the same value is also reported separately for depolarizing and hyperpolarizing traces. |
| `average_fahp` | For each detected in-step AP, search from repolarization toward the next AP threshold (or a bounded post-peak window for the last AP) and record the minimum voltage; average valid trough voltages | Depolarizing traces only; preserve the per-AP trough list and valid-event count. Also report baseline-relative fAHP depth diagnostically. |
| `time_mahp` | Time from `epoch_stop_ms` to the minimum voltage in the configured post-step recovery window | Depolarizing traces only; report latency in ms, not absolute recording time. |
| `amplitude_mahp` | `baseline_voltage - post_step_minimum_voltage` | Depolarizing traces only; positive amplitude means a post-step hyperpolarization. |
| `ap_count` | Number of upward threshold crossings inside the stimulus epoch | Depolarizing traces only; zero is a valid value. Keep detected crossing/peak times. |
| `last_stabilized_isi` | Difference between the final two in-step AP peak times (or threshold times, selected once in config) | Depolarizing traces with at least two APs; otherwise mark invalid and apply the finite invalid-feature penalty. |
| `last_ap_half_width` | Time between the rising and falling crossings of the last AP's half-amplitude voltage level | Depolarizing traces with at least one AP; use interpolated crossings and report ms. |

The fAHP and mAHP windows must be recorded in the feature metadata so the
solution figure can show the actual measurement regions. Do not use the
existing differentiable soft-spike loss for these objective values.

### Hyperpolarizing features

For each hyperpolarizing trace, use the centered hyperpolarizing current step
and the baseline voltage above:

| Objective label | Automatic measurement | Validity/aggregation |
| --- | --- | --- |
| `peak_input_resistance` | `(ΔV_peak / ΔI_step)` in MOhm, where `ΔV_peak` is the most negative voltage deflection in the configured early-step peak window | Hyperpolarizing traces only; because mV/nA = MOhm, preserve the positive resistance sign by dividing the negative voltage deflection by the negative current deflection. |
| `steady_state_input_resistance` | `(ΔV_steady / ΔI_step)` using the mean voltage in the final configured fraction of the step | Hyperpolarizing traces only; retain the exact steady-state time window and current estimate. |
| `sag` | Default normalized sag ratio `(ΔV_steady - ΔV_peak) / (V_baseline - V_peak)`; also save raw sag amplitude in mV | Hyperpolarizing traces only; a zero/near-zero peak deflection is invalid and receives the configured finite penalty. |

The implementation should preserve raw values (`peak_voltage_mV`,
`steady_state_voltage_mV`, `raw_sag_mV`, and current amplitude) in addition to
the three objective features. This will make it possible to audit the sign
convention and distinguish a resistance mismatch from a current-window error.

### Time-course objective

The ten landmark features above do not constrain the voltage between their
measurement points. Add an eleventh objective, `trajectory_overlap`, to track
the voltage course sample by sample. For each aligned experimental/simulated
sample, calculate a soft overlap score in absolute voltage units:

```text
overlap(t) = exp(-0.5 * ((v_sim(t) - v_exp(t)) / overlap_sigma_mV)**2)
trajectory_loss = 1 - weighted_mean(overlap(t))
```

The weighted mean must:

- use the existing score mask and the same time grid for both traces;
- average per trace before averaging across traces;
- balance baseline, stimulus, recovery, and the first 100 ms after
  `epoch_start_ms` using the configured window weights;
- remain in absolute mV space, with no per-trace mean subtraction or
  variance normalization;
- use no dynamic time warping, interpolation to a shifted event, or other
  temporal alignment that could hide timing errors.

The first-100-ms window is intentionally weighted separately so the rapid
spiking onset is not overwhelmed by hundreds of milliseconds of baseline or
recovery. The objective should also retain per-window overlap scores, pointwise
MAE/RMSE, and fractions of samples within 1, 2, and 5 mV as diagnostics. These
diagnostics are not additional Optuna objectives.

This objective complements, rather than replaces, spike count, ISI, width, AHP,
and resistance objectives. It is deliberately a trajectory comparison, not a
global correlation: correlation alone can remain high when the simulated trace
has an offset or amplitude mismatch.

### Experimental targets

Before the study starts, run the same extractor on every selected experimental
record. Write the resulting target table, feature validity, detector settings,
window coordinates, and checksums to `experimental_features.json` and
`experimental_features.csv`.

The MOCMA pipeline uses an explicit train/test split within every protocol:
trace indices 2 and 4 are optimization traces, while indices 1 and 3 are
held-out test traces. The time-course fitness masks use 200 ms before and 600
ms after the depolarizing step, and 200 ms before and 200 ms after the
hyperpolarizing pulse. Masks are clipped to the available recording and the
requested and actual coordinates are stored in trace metadata.

Prefer the direct segmented trace measurement as the optimization target. Where
an existing IPFX metric has an equivalent definition, compare it in a validation
report and record the difference; do not mix IPFX and direct-detector values in
one objective without an explicit conversion.

## Objective calculation

For each candidate and each objective label:

1. Simulate all selected trace buckets with the same Jaxley model and parameter
   vector.
2. Extract the ten scalar features from simulated traces and calculate the
   time-course overlap from the full aligned voltage traces.
3. Pair simulated and experimental feature values by trace key.
4. For valid pairs, calculate a squared normalized error:

   `((simulated - experimental) / fixed_feature_scale) ** 2`.

5. Average over valid traces within that feature's applicable protocol set,
   without allowing a long trace or a high-sample-rate trace to dominate.
6. If a required feature is invalid for a candidate, add the configured finite
   invalid-feature penalty and record the reason. Never return NaN or Inf to
   Optuna.

The first ten objective values use the normalized feature-error calculation.
The eleventh value is the `trajectory_overlap` loss defined above. The objective
function returns exactly eleven values in a stable order. It should
also attach the per-trace feature table, validity reasons, Jaxley simulation
status, per-window overlap diagnostics, and wall-clock timing to
`trial.user_attrs` or the durable candidate record. Objective values remain the
only values passed to `study.optimize`.

No gradients are computed. A model is built once per process, its Jaxley
simulation kernels are compiled lazily by static bucket shape, and each trial
only supplies a new physical parameter dictionary from Optuna. The physical
parameter vector, normalized vector if used internally, and bounds are all
saved for reproducibility.

## Pareto-front reporting

After `study.optimize` completes, use Optuna's nondominated trials as the
Pareto front. This is reporting of the MOCMA study, not a second optimization
step.

Write at least:

```text
runs/<run-id>/
├── resolved_config.yaml
├── run_manifest.json
├── experimental_features.csv
├── experimental_features.json
├── objective_definitions.json
├── all_trials.jsonl
├── pareto_front.jsonl
├── pareto_front.csv
├── pareto_objectives.png       # optional visualization of objective values
└── solutions/
    └── trial_<number>/
        ├── parameters.csv
        ├── features_simulated.csv
        ├── objective_values.json
        ├── objective_ranks.json
        ├── predictions.npz
        └── solution.png
```

Each Pareto row must contain:

- Optuna trial number and sampler/study metadata;
- all physical parameter values and the exact parameter bounds;
- all eleven objective values with their labels and fixed scales/overlap
  settings;
- per-objective rank among Pareto trials, where rank 1 is the smallest value;
- normalized gap from the best Pareto value for that objective;
- percentile among Pareto trials;
- `best_for_objectives`, listing objective labels for which this solution is
  tied for the minimum within a documented tolerance;
- per-trace simulated feature values, experimental values, residuals, and
  validity reasons;
- a deterministic relative path to its solution figure and prediction file.

This directly answers which solution minimizes which physiological objective
most. If no solution is best for a particular objective, the front summary must
still include the objective winner and its trial number. Do not collapse these
annotations into one aggregate score.

Use a stable tie tolerance and write it to `objective_definitions.json`. Keep
all nondominated trials, including duplicate objective vectors with different
parameters, unless Optuna has explicitly marked the trial non-complete.

## Solution figures

For every Pareto solution, plot experimental and simulated voltages for every
selected trace. Each depolarizing trace gets:

1. the full segmented signal;
2. an additional zoom panel covering the first 100 ms after
   `epoch_start_ms`—the first 100 ms of the depolarizing/spiking epoch, not the
   first 100 ms of the recording, which is pre-step baseline in these files.

Hyperpolarizing traces get their full segment plus the marked early peak and
steady-state windows. The figure should show the current-epoch boundaries and,
where applicable, detected AP peaks, fAHP troughs, mAHP minimum, and
hyperpolarizing peak/steady-state levels. Titles or a compact side panel should
include the trial number, Pareto objective values, and the
`best_for_objectives` labels. This keeps the extra 100 ms plot inside each
solution figure rather than producing a separate unlinked diagnostic image.

## Execution flow

1. Validate the standalone YAML and confirm that the selected Jaxley runtime
   is available.
2. Load all selected segmented traces, apply existing resampling/window rules,
   and build static buckets.
3. Build the Jaxley model once and create one simulation kernel per bucket.
4. Extract and persist experimental feature targets before sampling.
5. Construct a physical-bound `FloatDistribution` search space and one
   `MoCmaSampler`.
6. Run exactly `n_trials` calls to the Jaxley-backed objective, optionally
   persisting the Optuna study to the configured RDB storage.
7. Extract nondominated trials, calculate reporting-only ranks/gaps/winners,
   re-simulate each Pareto trial if predictions were not retained, and write
   all durable records.
8. Generate the full-trace plus 100 ms-spiking solution figures.

Suggested commands, after implementation, are:

```bash
cd NewJaxleyMultiObjective
# Activate the existing Jaxley environment used by Jaxley_refactored.
python -m newjaxley_multiobjective.cli validate \
  --config configs/mocma_features.yaml
python -m newjaxley_multiobjective.cli run \
  --config configs/mocma_features.yaml \
  --cell-id m20240527cd \
  --seed 1234
```

The README should document the environment's Jaxley, JAX, Optuna, and
OptunaHub versions and should fail early with an actionable message if
`optunahub.load_module("samplers/mocma")` is unavailable. Do not create a
second environment as part of this pipeline.

## Verification plan

### Feature tests

- Synthetic no-spike traces verify resting voltage, zero AP count, and invalid
  fAHP/ISI/half-width handling.
- Synthetic traces with known spike times verify AP count, final ISI, half-width,
  fAHP trough averaging, and mAHP latency/amplitude.
- Synthetic hyperpolarizing traces with known baseline/current/peak/steady
  values verify both resistances, sag ratio, and sign conventions.
- Boundary tests verify windows clipped at trace ends and finite invalid-event
  penalties.
- Synthetic shifted and amplitude-offset traces verify that
  `trajectory_overlap` worsens for time-course mismatch and is not improved by
  subtracting the simulated mean.
- Window-balance tests verify that the first 100 ms of spiking contributes its
  configured weight instead of being diluted by long baseline/recovery traces.
- Experimental regression tests compare direct extracted values with the
  existing IPFX metrics CSV where definitions match, while documenting expected
  differences where they do not.

### Pipeline tests

- Config validation rejects any sampler other than MOCMA, categorical/conditional
  search parameters, missing scales, invalid windows, or non-finite bounds.
- A one- or two-trial smoke run in the existing Jaxley environment proves that
  each trial calls Jaxley, returns eleven finite objective values, and writes the
  manifest and trial records.
- A deterministic seed gives identical parameter suggestions and objective
  values on a fixed test fixture.
- Pareto extraction tests verify all returned trials are nondominated and that
  per-objective ranks, gaps, and `best_for_objectives` are correct.
- Plot tests verify a solution figure is created for every Pareto trial and
  contains the full trace and the additional first-100-ms post-step panel.

### Acceptance criteria

The work is complete when a documented command in the existing Jaxley
environment can run the configured study, all candidates are simulated by
Jaxley, all eleven objectives—including time-course overlap—are finite and auditable, no optimizer other
than `MoCmaSampler` is invoked, and the output contains the complete annotated
Pareto front plus solution figures with the requested 100 ms spiking plots.

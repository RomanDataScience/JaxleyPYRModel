# Jaxley–Currentscape wrapper plan

## Status and objective

This document specifies a post-fitting analysis wrapper that replays one Jaxley
simulation and produces a Currentscape for every membrane-current mechanism
present at a selected recording site.

The wrapper must:

- use the same model, morphology, fitted parameters, stimulus, initial state,
  time step, and solver as the source run;
- execute the neuron simulation once;
- retain separate traces for mechanisms that share Jaxley current names such as
  `i_Na`, `i_K`, and `i_Ca`;
- verify the recovered currents against Jaxley's directly recorded aggregate
  currents;
- produce machine-readable data and publication-ready Currentscape figures;
- remain a post-fit diagnostic and leave the fitting and hybrid pipelines
  unchanged.

## Executive design decision

Jaxley exposes channel currents through `current_name`, but the Combe mechanisms
intentionally share current names by ion family. For example, `kd`, `kap`,
`Kv2like`, `km`, `kca`, `mykca`, `kad`, and `kir` all contribute to `i_K`.
Likewise, sodium mechanisms contribute to `i_Na`, and calcium mechanisms
contribute to `i_Ca`.

The wrapper must not rename those current states. In particular, `cal4` consumes
the aggregate `i_Ca` state to update intracellular calcium, so renaming calcium
currents could change the simulated model.

The preferred extraction method is therefore:

1. build the original model and apply the selected physical parameter vector;
2. select one experimental trace and one or more anatomical recording sites;
3. discover every membrane channel structurally present at each site;
4. add recordings for voltage, direct aggregate currents, the union of required
   gating states, and relevant ion concentrations;
5. perform one Jaxley integration;
6. evaluate each channel object's own `compute_current` method over the recorded
   voltage/state trajectories using the exact resolved parameter state;
7. group the recovered mechanism currents by `current_name` and compare their
   sums with the directly recorded Jaxley aggregates;
8. write validated arrays and pass them to Currentscape.

This avoids duplicated hand-written current equations and keeps the diagnostic
coupled to the mechanism implementations that actually generated the voltage.

## Scientific scope

### Definition of “every current”

For the first implementation, “every current” means every structurally present
membrane-current mechanism at a selected compartment, including mechanisms
whose fitted conductance or resulting current is exactly zero.

It does not mean summing a mechanism over the entire cell. Ionic-current balance
is local and changes between soma, axon, basal dendrite, and apical dendrite.
Each selected site therefore gets its own Currentscape.

The default site is the configured somatic recording site. Additional sites can
be requested explicitly, for example an axon/AIS site and representative basal
or apical sites. All requested sites should be recorded in the same integration.

### Included observables

- membrane voltage;
- injected current for context;
- one current-density trace per membrane mechanism;
- direct Jaxley current-family aggregates;
- total inward and outward membrane current;
- intracellular calcium concentration when present;
- optional gating-state traces in the data artifact, disabled by default after
  current reconstruction to keep outputs compact.

### Excluded from the current stack

- `d3`, because it is a zero-current geometry/state placeholder;
- `cal4`, because it is a concentration pump/dynamics mechanism rather than a
  membrane ionic current;
- axial current, capacitive current, and injected current;
- synaptic currents in the initial version.

`cal4` state and calcium flux diagnostics may still be exported separately, and
`CaCon_i` can be supplied to Currentscape as an ion-concentration panel.

### Non-goals

- adding ionic-current terms to LSU_1 or differentiating through Currentscape;
- recording currents during CMA-ES or every Adam epoch;
- changing channel equations or current names;
- treating normalized current percentage as causal attribution;
- producing a whole-cell spatial current map;
- silently accepting an unclassified new mechanism.

## Current mechanism inventory

The runtime cell remains the source of truth. A static metadata registry will
classify mechanism type, ion family, display label, plot order, and whether it
belongs in the membrane-current stack. At runtime, the registry must be checked
against `cell.channels`, `cell.pumps`, and the enabled-mechanism set.

| Mechanism | Current family/name | Inserted regions | Currentscape |
| --- | --- | --- | --- |
| `Leak` | leak; `i_Leak` | all | include |
| `icand` | nonspecific cation; `i_icand` | soma, apical | include |
| `na16a` | Na; `i_Na` | soma, apical | include |
| `nax` | Na; `i_Na` | axon | include |
| `na3dend` | Na; `i_Na` | basal | include |
| `nap` | persistent Na; `i_Na` | soma, basal | include |
| `kd` | K; `i_K` | soma, apical, axon, basal | include |
| `Kv2like` | K; `i_K` | soma, apical, axon, basal | include |
| `kap` | proximal A-type K; `i_K` | soma, apical, axon, basal | include |
| `kad` | distal A-type K; `i_K` | apical | include |
| `km` | M-type K; `i_K` | soma, apical, axon | include |
| `kca` | calcium-activated K; `i_K` | soma, apical | include |
| `mykca` | fast calcium-activated K; `i_K` | soma, apical | include |
| `kir` | inward-rectifier K; `i_K` | apical, basal | include |
| `h` | hyperpolarization-activated mixed current; `i_H` | soma, apical, basal | include |
| `cal` | L-type Ca; `i_Ca` | soma | include |
| `cat` | T-type Ca; `i_Ca` | soma, apical | include |
| `car` | R-type Ca; `i_Ca` | soma, apical | include |
| `calH` | high-voltage Ca; `i_Ca` | apical | include |
| `d3` | zero placeholder; `i_d3` | all | exclude with reason |
| `cal4` | calcium concentration dynamics | all | exclude from stack; export state |

Structural presence must be determined from the runtime cell's mechanism mask,
not from whether the current happens to be nonzero. The output manifest must
list zero-current mechanisms so the phrase “every current” is auditable.

The expected membrane-current counts are 14 at a somatic site, 15 at an apical
site, 6 at an axonal site, and 8 at a basal site. These are regression
expectations, not a substitute for runtime discovery.

## Proposed architecture

```text
completed fit/hybrid run + parameter CSV + selected trace + site specification
                                  |
                                  v
                         ReplayInputLoader
                  (hash and compatibility checks)
                                  |
                                  v
                      CurrentRecordingPlanner
          (active channels, states, aggregate currents, ions)
                                  |
                                  v
                      one Jaxley integration
                                  |
                                  v
                    MechanismCurrentEvaluator
       (each channel.compute_current using resolved states/params)
                                  |
                                  v
                     CurrentValidationService
        (finite data, timing, units, family sums, voltage replay)
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              analysis arrays             Currentscape
            + provenance tables          PNG/PDF figures
```

### Package responsibilities

The exact names can change during implementation, but the responsibilities
should remain separated:

- `analysis/current_specs.py`: immutable current metadata and complete
  classification of supported mechanisms;
- `analysis/replay.py`: load a completed run, parameter vector, selected trace,
  and exact model configuration;
- `analysis/current_recording.py`: resolve sites, construct the recording plan,
  execute the single forward simulation, and return recorded trajectories;
- `analysis/current_evaluation.py`: evaluate each active channel's
  `compute_current` without duplicating channel equations;
- `analysis/current_validation.py`: perform all numerical and scientific
  consistency checks;
- `reporting/current_outputs.py`: write tables, compressed arrays, metadata, and
  summary statistics;
- `reporting/currentscape_plot.py`: optional dependency boundary around the
  external Currentscape package;
- `cli/commands.py`: expose a thin post-fit analysis command.

Currentscape and its plotting dependencies should be an optional project
dependency. Importing or running fitting must not require Currentscape.

## Input contract

### Preferred source: completed run directory

The safest interface starts from a completed fit or hybrid run directory:

- load `resolved_config.yaml`;
- require a complete `status.json`;
- use `parameters_best.csv` by default;
- allow `parameters_final.csv` for a local fit when explicitly requested;
- verify parameter names are unique and exactly match the built model's fitted
  parameter catalog;
- reject missing, unknown, duplicate, nonfinite, or out-of-bounds values;
- rebuild the model and verify its model signature against `run_manifest.json`;
- retain the source compatibility/config hash in every analysis artifact.

The hybrid selected model already writes `parameters_best.csv`, so local and
hybrid runs can share this interface.

### Trace selection

One invocation analyzes exactly one trace. Selection must be unambiguous:

- cell ID comes from the source run unless explicitly checked and overridden;
- require an exact trace ID plus protocol when needed;
- load the experimental current waveform through the existing
  `SegmentedTraceLoader`;
- use the same resampling, `dt`, sample count, alignment, and initial-voltage
  rule as fitting;
- never silently select the first matching trace.

The first scientific target should be one depolarizing training trace because
it exposes the spike-generating currents. Hyperpolarizing and validation traces
can be analyzed through identical later invocations.

### Site selection

Each site uses the existing group/branch/location semantics. Defaults:

- use the configured somatic recording site;
- support repeated explicit site specifications;
- reject invalid or ambiguous sites before JAX compilation;
- create one Currentscape per site from the same integration.

A full-cell “all compartments” mode should not be part of the first release.
If needed later, it should be a separate spatial-analysis product with explicit
area weighting, not an overloaded Currentscape.

## Parameter and state correctness

The fitted parameter vector must be converted through the existing
`Parameterizer`, producing the same `param_state` used during fitting. Current
reconstruction must read the resolved Jaxley parameter arrays after applying
that state, using the equivalent of `cell.get_all_parameters(param_state)`.

It is not sufficient to read values from `cell.nodes`: fitted parameters are
dynamic JAX inputs and may not be written back to the host-side node table.
Using stale node-table values would produce plausible but scientifically wrong
currents. Shared reversal potentials such as `eNa`, `eK`, and `eCa` must also
come from the resolved site parameters rather than channel-class defaults,
because exact-HOC builds can carry segment-specific imported values.

At each requested site, record:

- `v`;
- every distinct direct aggregate `current_name` needed for validation;
- the union of `channel_states` required by structurally present channels;
- shared states such as `CaCon_i`;
- optional pump/ion states requested for auxiliary plots.

The recording planner should deduplicate shared states and preserve a stable
ordering. It must store the exact mapping between output rows and state names.

## Individual-current reconstruction

For each active membrane channel at a site:

1. obtain the recorded voltage vector;
2. assemble the state dictionary expected by the channel from recorded
   trajectories and shared states;
3. extract the site's resolved parameters from the applied Jaxley parameter
   state;
4. call that channel instance's `compute_current` on the full time vector;
5. convert to the declared canonical current-density unit if necessary;
6. retain the mechanism name, ion family, original `current_name`, site, sign
   convention, and structural-presence flag.

No current equation should be reimplemented in the wrapper. This is especially
important for GHK calcium currents, the Markov `na16a` mechanism, and
calcium-dependent mechanisms.

### Timing convention

The implementation must establish whether Jaxley recordings expose pre-update
or post-update states at each returned voltage sample. A short feasibility test
must compare reconstructed currents against directly recorded aggregate
currents across:

- the initial sample;
- subthreshold samples;
- spike upstroke and peak;
- recovery samples;
- the final sample.

If standard recordings cannot reproduce aggregate currents at a single,
well-defined sample offset, the fallback is a diagnostic-only low-level Jaxley
scan that captures states and per-channel currents at the exact solver step.
The fallback must still reuse Jaxley's step functions and run the neuron only
once.

## Units and signs

All Currentscape inputs must use one current-density unit. The converted
electrical mechanisms return `mA/cm²` and declare their convention through
Jaxley channel metadata. The wrapper should use `mA/cm²` as its canonical
mechanism-current unit, normalize any future mixed conventions explicitly, and
record the unit in the artifact. M0 must also confirm that the directly recorded
Jaxley aggregates and native `Leak` current are expressed on the same scale
before they are used as validation targets.

The sign convention should remain Jaxley's native membrane-current convention:

- outward current is positive;
- inward current is negative.

This must be verified empirically with sodium current during the action
potential upstroke and potassium current during repolarization. Currentscape
must receive the unchanged signs.

Current density is the canonical representation for a local Currentscape.
Optional conversion to absolute current requires the exact compartment membrane
area and must be reported as a separate derived field. Different representations
must never be mixed in one stack.

## Validation gates

The wrapper must fail closed before plotting if any required gate fails.

### Structural validation

- every enabled runtime channel is either classified as a membrane current or
  explicitly excluded with a documented reason;
- every plotted channel is structurally present at the requested site;
- every required state and parameter is available;
- output labels are unique even when `current_name` values collide;
- zero-conductance channels remain represented in the data manifest.

### Numerical validation

- time, voltage, states, concentrations, and currents have identical lengths;
- time is strictly increasing and uniformly sampled;
- all plotted values are finite;
- the replayed voltage matches the ordinary simulation kernel for the same
  inputs within a strict numerical tolerance;
- for every shared current family, the sum of reconstructed mechanism currents
  matches the directly recorded Jaxley aggregate at every valid sample;
- direct and reconstructed totals use the same unit and sign;
- no unexplained sample shift is tolerated.

Absolute tolerance must be combined with a scale-aware relative tolerance so
near-zero currents do not generate misleading failures. The report must store
maximum absolute error, RMS error, relative error over active samples, and the
number of excluded edge samples, if any. Edge samples may only be excluded
after the timing convention is documented and tested.

### Scientific validation

- Na current is inward during spike initiation;
- delayed-rectifier/Kv currents become outward during repolarization;
- leak and H-current behavior is plausible during the hyperpolarizing pulse;
- calcium current families and `CaCon_i` remain consistent with the aggregate
  used by `cal4`;
- a short soma/axon regression agrees with the existing NEURON-versus-Jaxley
  current diagnostic for overlapping mechanisms.

At a compartment in a branched model, ionic current alone does not close the
full voltage balance because axial, capacitive, and injected currents also
contribute. The validation report must not mislabel the sum of ionic currents as
a full Kirchhoff residual unless those terms are included.

## Currentscape reporting

Currentscape accepts a voltage trace and an ordered collection of current
traces. The adapter should:

- preserve a deterministic mechanism order and color mapping across cells,
  traces, sites, and parameter sets;
- use publication-safe, colorblind-readable styling;
- display voltage, inward fractions, outward fractions, and absolute inward and
  outward totals;
- optionally display `CaCon_i` in the ion-concentration panel;
- include both full-trace and configured time-window views without rerunning the
  model;
- label the cell, trace, protocol, site, parameter source, and current unit;
- save raster PNG and vector PDF/SVG outputs;
- keep zero-current mechanisms in the manifest even if Currentscape suppresses
  them visually.

Normalized fractions can make a tiny total current look dominant. Every
Currentscape must therefore be accompanied by absolute-current panels or a
summary table containing at least peak inward current, peak outward current,
time-integrated absolute current, and total contribution.

## Output contract

Each invocation gets a collision-safe analysis directory under the source run:

```text
<run>/analysis/currentscape/<analysis-id>/
  analysis_config.yaml
  current_manifest.csv
  current_summary.csv
  traces.npz
  validation.json
  provenance.json
  sites/
    soma_branch0_loc0p5/
      currentscape_full.png
      currentscape_full.pdf
      currentscape_stimulus.png
      currentscape_recovery.png
```

`traces.npz` should include:

- `time_ms`;
- `voltage_mV`;
- `stimulus_nA`;
- one current-density array per site and mechanism;
- direct aggregate current arrays;
- requested concentration arrays.

`current_manifest.csv` should map safe array keys to display labels, mechanism
names, ion families, current names, sites, units, inclusion status, and
exclusion reasons.

`provenance.json` should include:

- source run and selected parameter filename;
- configuration and compatibility hashes;
- model signature;
- parameter-file checksum;
- experimental-input checksums;
- trace key and site definitions;
- Jaxley, JAX, NumPy, Currentscape, and wrapper versions;
- solver, voltage solver, precision, backend, `dt`, and sample count;
- git state and analysis timestamp;
- validation status and tolerances.

The analysis ID should hash the source model/config, parameter file, trace,
sites, extraction settings, and wrapper version. Repeating an identical analysis
must be reproducible and must not overwrite a different analysis.

## User-facing interface

The planned CLI should be a separate post-fit command conceptually equivalent
to:

```text
jaxley-refactored currentscape
  --run <completed-run-directory>
  --parameters best
  --trace <exact-trace-id>
  --protocol depolarizing_step
  --site soma:0:0.5
```

Useful options:

- `--parameters best|final|reference|<csv-path>`;
- repeated `--site group:branch:location`;
- `--window full|stimulus|recovery`;
- `--include-states` for debugging artifacts;
- `--validate-only` to exercise extraction without plotting;
- `--output-root` for an external analysis directory.

The command should print the output path, sites, mechanisms found, validation
errors, and figure paths. It should return a nonzero exit status on any failed
validation gate.

## Implementation milestones

### M0 — extraction feasibility spike

1. Build the Combe model from an existing config.
2. Select a short depolarizing trace and the somatic midpoint.
3. Discover active mechanisms and required states programmatically.
4. Record voltage, states, `i_Na`, `i_K`, `i_Ca`, and unique currents.
5. Evaluate each mechanism through its own `compute_current`.
6. Compare per-mechanism sums to direct aggregates and resolve sample timing
   and units.

Exit gate: all soma current families match direct Jaxley recordings within
documented tolerances, with no hand-coded mechanism formulas.

### M1 — typed analysis model and run replay

1. Add current metadata with complete mechanism classification.
2. Add immutable analysis, site, parameter-source, and output specifications.
3. Implement strict fitted-parameter CSV loading.
4. Rebuild and verify source-run compatibility and model signature.
5. Reuse existing trace loading, initial-state construction, and site
   resolution.

Exit gate: an analyzed run is provably tied to the same model, parameters, and
input trace as the source fit.

### M2 — generic single-pass recorder

1. Build a recording plan from runtime channel/pump metadata.
2. Record all requested sites in one integration.
3. Resolve dynamic fitted parameters at each site.
4. Reconstruct individual currents generically.
5. Support soma, axon, basal, and apical sites without site-specific current
   equations.

Exit gate: every structurally present membrane mechanism produces a finite
trace or a documented exact-zero trace.

### M3 — validation and durable outputs

1. Implement structural, numerical, family-sum, sign, timing, and voltage-replay
   validation.
2. Write atomic NPZ, CSV, JSON, and YAML artifacts.
3. Add deterministic analysis IDs and complete provenance.
4. Generate current summary statistics.

Exit gate: corrupted parameters, missing mechanisms, unit mismatches, sample
offsets, and aggregate-current mismatches all fail before plotting.

### M4 — Currentscape adapter

1. Add Currentscape as an optional dependency.
2. Implement deterministic labels, ordering, and colors.
3. Generate full, stimulus, and recovery figures.
4. Include absolute-current context and optional calcium concentration.
5. Verify headless CPU rendering.

Exit gate: a validated best-fit depolarizing replay produces nonempty
publication-resolution PNG and vector outputs for soma and axon.

### M5 — CLI, documentation, and regression coverage

1. Add the post-fit CLI command without changing `fit` or `hybrid-fit`.
2. Document local and Slurm post-processing workflows.
3. Add unit, integration, and regression tests.
4. Run the diagnostic on both available cells and archive manifests.

Exit gate: the legacy fit pipeline, hybrid pipeline, and full existing test
suite remain unchanged and passing.

## Test matrix

### Unit tests

- current metadata covers every mechanism in `combe2023_mechanisms()`;
- duplicate display names and unclassified mechanisms are rejected;
- structural presence is distinguished from numerical zero;
- shared state recordings are deduplicated;
- parameter CSV validation rejects all malformed cases;
- unit conversion and sign preservation are deterministic;
- output array keys and plot colors are stable.

### Small integration tests

- one-compartment reference channels: reconstructed current equals direct
  current recording;
- short Combe soma trace: Na/K/Ca and unique current sums match;
- short Combe axon trace: `nax`, `kd`, `Kv2like`, `kap`, `km`, and leak are
  recovered;
- multiple sites are recorded in one integration;
- fitted dynamic parameters, rather than stale node defaults, change recovered
  currents;
- standard simulation and current-aware replay produce matching voltage;
- Currentscape renders under the noninteractive `Agg` backend.

### Scientific regression tests

- compare overlapping soma and axon currents with
  `channels_converted/modelComparison/current_diagnostics.py`;
- check inward Na and outward K signs around a known spike;
- check exact-current-family sums across subthreshold, spike, and recovery
  windows;
- run one depolarizing and one hyperpolarizing trace for each cell;
- store numerical summaries as versioned regression fixtures with explicit
  tolerances rather than image-only comparisons.

The existing diagnostic is only an overlapping reference: its manually
constructed somatic total omits `nap`, `cal`, `cat`, `kca`, and `mykca`, so it
must not be used as the oracle for complete total membrane current.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Shared current names hide mechanism identity | Recompute each channel through its own `compute_current`, then validate against the shared aggregate |
| Renaming currents changes calcium dynamics | Never rename model currents; instrumentation is read-only |
| Host node values differ from fitted dynamic parameters | Resolve parameters from the applied `param_state` |
| Recorded state/current timing differs by one step | M0 timing tests across spikes; low-level step-function fallback |
| Mixed current-unit conventions | Explicit per-channel normalization plus aggregate parity tests |
| New mechanisms are silently omitted | Complete fail-closed metadata classification |
| Full-cell recording exhausts memory | Site-local MVP; estimate recording size before integration |
| Currentscape percentages overstate tiny currents | Always retain absolute totals and summary metrics |
| Plot settings alter fit checkpoint hashes | Separate post-fit analysis configuration and outputs |
| Version changes alter current observables | Store exact package versions and maintain versioned integration tests |

## Definition of done

The wrapper is complete when a user can point it at a completed LSU_1 local or
hybrid run, select one recorded trace and one or more sites, and receive:

1. one replayed Jaxley simulation using the selected fitted parameters;
2. a separate current trace for every structurally present membrane mechanism;
3. direct evidence that the individual traces sum to Jaxley's current-family
   aggregates;
4. reproducible machine-readable arrays and summaries;
5. publication-ready Currentscape figures;
6. a provenance and validation report sufficient to audit the figure without
   rerunning optimization.

## Primary references

- Jaxley channel-current recording:
  <https://jaxley.readthedocs.io/en/v0.11.5/tutorials/05_channel_and_synapse_models.html>
- Jaxley current observables and current-state changes:
  <https://jaxley.readthedocs.io/en/stable/changelog.html>
- Currentscape inputs and reporting:
  <https://currentscape.readthedocs.io/en/latest/>
- Existing project current diagnostic:
  `channels_converted/modelComparison/current_diagnostics.py`

# Jaxley MOCMA pipeline

This folder contains the feature-based multi-objective pipeline. It reuses the
model and segmented trace loader from `Jaxley_refactored` and uses only
OptunaHub's `MoCmaSampler` for parameter sampling.

Run it with the existing Jaxley environment:

```bash
cd NewJaxleyMultiObjective
python -m pip install -e .
bash run_mocma.sh
```

For a new Conda environment on a workstation or HPC login node, run the
installer once:

```bash
bash install_conda_env.sh
conda run --name jaxley-mocma bash NewJaxleyMultiObjective/run_mocma.sh validate
```

The installer creates or reuses `jaxley-mocma`, installs Jax/Jaxley and the
Optuna dependencies, and installs both local packages in editable mode. Set
`ENV_NAME` to use another environment name. Set `INSTALL_TESTS=1` to include
pytest.

The default configuration stores the Optuna study in `runs/<run-id>/study.db`
and resumes automatically. Completed trials are checkpoints, and MOCMA's
sampler state is persisted in the same SQLite study. If a run is interrupted,
rerun the same command; it counts existing trials and evaluates only the
remaining budget. A stale running trial is marked failed after the configured
heartbeat grace period, then the run continues.

Set `multi_objective.resume: false` only when starting a deliberately new
study with a new output/study name. Changing model, objective, seed, or
population settings for an existing study is rejected by the run hash check.

For a quick check, copy the config and reduce `multi_objective.trials` to 1 or
2. The supplied configs set `protocols: [depolarizing_step]`, so each trial
simulates only depolarizing traces 2 and 4. Test traces 1 and 3 remain
configured but are not simulated because `evaluate_test: false`. Set
`evaluate_test: true` only for a separate validation run. Set
`PYTHON_EXECUTABLE` when the Jaxley environment is not the shell's default
Python.

For the current depolarizing-only mode, the fixed-time fitness window is 200
ms before through 600 ms after each depolarizing step. If a recording starts
before the requested pre-window, the available portion is used and recorded in
the trace metadata.

The supplied configs use `objectives.mode: depolarizing_fidelity`. The main
configuration evaluates five objectives: one voltage objective for each
training trace, spike count/timing, AP waveform, and recovery. The quick
configuration uses six parallel workers; the main configuration uses eight.
Reduce `parallel_workers` if memory or CPU capacity is limited.

## AP-count-only MOCMA search

`configs/mocma_ap_count.yaml` runs MOCMA with exactly one objective,
`ap_count`. It minimizes the mean squared difference between simulated and
experimental upward threshold-crossing counts across the selected training
traces. Count mismatches remain graded, so zero, partial, and correct spiking
solutions are distinguishable; spikes outside the stimulus interval are still
rejected by the hard guard.

Run it with:

```bash
cd NewJaxleyMultiObjective
CONFIG_PATH=configs/mocma_ap_count.yaml bash run_mocma.sh validate
CONFIG_PATH=configs/mocma_ap_count.yaml bash run_mocma.sh run
```

The current four-objective configuration remains available below for later
waveform and recovery refinement.

## Four requested depolarizing objectives

`configs/mocma_four_objectives.yaml` configures exactly four minimization
objectives for the MOCMA study:

1. `firing_rate`: AP count divided by stimulus duration, in Hz.
2. `spike_shape`: peak-aligned voltage waveform error over ±8 ms around each
   corresponding AP, including amplitude and width.
3. `post_stimulus_recovery`: post-step voltage trajectory error plus explicit
   terminal return-to-baseline error.
4. `depolarized_plateau`: fixed-time inter-spike voltage error during the step;
   only the narrow AP core is excluded, so post-spike hyperpolarizing dips are
   penalized.

Run that study yourself with:

```bash
cd NewJaxleyMultiObjective
CONFIG_PATH=configs/mocma_four_objectives.yaml bash run_mocma.sh validate
CONFIG_PATH=configs/mocma_four_objectives.yaml bash run_mocma.sh run
```

For a smoke run, use `--trials 1` on the last command. Set `SEED`,
`PYTHON_EXECUTABLE`, `D_LAMBDA`, or `parallel_workers` in the config as needed.

The configured morphology resolution is `morphology_d_lambda: 0.1`. To run
the same pipeline at `d_lambda=0.3`, use `D_LAMBDA=0.3`; this automatically
changes the model signature and therefore creates a separate study/run:

```bash
D_LAMBDA=0.3 bash NewJaxleyMultiObjective/run_mocma.sh run
```

To compare `0.3` and `0.1` using the same model-reference parameter vector,
run the discretization benchmark in the Jaxley environment:

```bash
NEURON_MODULE_OPTIONS=-nogui \
  PYTHONPATH=NewJaxleyMultiObjective:Jaxley_refactored \
  conda run --no-capture-output --name jaxley-mocma \
  python NewJaxleyMultiObjective/compare_discretization.py \
  --config NewJaxleyMultiObjective/configs/mocma_features.yaml
```

The benchmark writes voltage RMSE, maximum voltage difference, and feature
differences to `runs/discretization_comparison-<cell>/comparison.json`. For a
specific fitted solution, add its `parameters.csv` with `--parameters` so the
comparison uses exactly the same fitted parameter vector at both resolutions.

For a quick MOCMA comparison using exactly one trial at each resolution, use
the same seed for both runs:

```bash
SEED=1234 bash NewJaxleyMultiObjective/compare_one_trial.sh
```

This writes separate `d_lambda=0.3` and `d_lambda=0.1` study directories. The
CLI also accepts `--trials 1` directly when launching an individual run.

The hard spike guard returns a loss of `100000` for every objective when a
depolarizing simulation spikes outside its stimulus interval or a
hyperpolarizing simulation spikes anywhere in the recorded segment.

## Features and loss calculation

The depolarizing training configuration uses traces 2 and 4. Each feature is
extracted from the same stimulus epoch metadata used by the Jaxley trace
loader. The feature extractor still contains the hyperpolarizing measurements
documented below for later protocol-specific runs, but they are not simulated
or optimized by the supplied depolarizing configs.

### Automatically measured features

For every trace, the pipeline measures resting membrane potential from the
30 ms before the stimulus. Depolarizing traces additionally measure:

| Feature | Measurement |
| --- | --- |
| `average_fahp` | Mean trough voltage after each detected AP, searching up to 20 ms or the next AP. |
| `time_mahp` | Time from stimulus offset to the minimum voltage in the post-step recovery window. |
| `amplitude_mahp` | Baseline voltage minus that mAHP minimum. |
| `ap_count` | Number of upward crossings of `-20 mV` during the depolarizing stimulus. |
| `last_stabilized_isi` | Peak-to-peak interval between the final two APs. |
| `last_ap_half_width` | Width of the final AP at half its threshold-to-peak amplitude. |

Hyperpolarizing traces measure peak input resistance from the first 20 ms of
the pulse, steady-state input resistance from the final 20% of the pulse, and
sag as `(steady voltage - peak voltage) / (baseline voltage - peak voltage)`.
Input resistance is calculated as voltage change divided by stimulus-current
change, so its sign follows the recorded current convention.

### Depolarizing fidelity objectives

All five objectives are minimized. The first objective is separate for each
training trace and compares the voltage at fixed time points. It uses a
normalized Huber loss:

```text
u(t) = abs(V_sim(t) - V_exp(t)) / trajectory_huber_delta_mV
Huber(t) = 0.5 * u(t)²,                         u(t) <= 1
           u(t) - 0.5,                          u(t) > 1
trajectory_loss = weighted_average(Huber(t))
```

This is an absolute, time-locked comparison: the voltage is not baseline
centered, mean-subtracted, shifted, or dynamically time-warped. Samples within
±3 ms of an experimentally detected AP peak receive weight 4. Consequently,
an AP that occurs at the wrong time remains a large trajectory error.

The `spike_timing_count` objective explicitly compares AP count and the timing
of corresponding threshold crossings and peaks:

```text
count_loss = ((N_sim - N_exp) / spike_count_scale)²
timing_loss = mean(0.5 * threshold_time_error²
                   + 0.5 * peak_time_error²)
```

The `ap_waveform` objective compares average fAHP, last stabilized ISI, and
last AP half-width. The `recovery` objective compares resting membrane
potential, mAHP timing, and mAHP amplitude. Scalar feature errors use

```text
((simulated_feature - experimental_feature) / objective_scale)²
```

Missing or invalid scalar features add `invalid_feature_penalty` (`1e6`).

Two hard constraints prevent clearly unusable solutions. If a depolarizing
trace spikes outside its stimulus interval, or a hyperpolarizing trace spikes
anywhere, every objective is set to `spike_violation_loss` (`1e5`). Also, if
the simulated AP count differs from the experimental count by more than
`ap_count_tolerance` (0 APs in the supplied depolarizing configs) on any
depolarizing training trace, every objective is set to
`ap_count_violation_loss` (`1e5`). The details of
these decisions are saved in each solution's `features_simulated.json` and in
the trial metadata.

These constraints are deliberately separate from the ordinary AP-count loss:
small count differences remain visible to MOCMA as a smooth objective, while
large count differences cannot survive merely because another voltage or
feature objective improved.

With `plot_completed_trials: true`, each trial writes a plot as soon as its
simulation finishes under `runs/<run-id>/trial_plots/trial_XXXXXX/`. These
plots use the predictions already calculated for the objective and do not
rerun the simulation. Final Pareto plots are still written under `solutions/`
when the study completes.

Outputs are written under `NewJaxleyMultiObjective/runs/`. Each Pareto solution
contains its parameters, optimization-trace features/predictions, objective
ranking, and a figure with full traces plus the first 100 ms after
depolarizing-step onset. Set `evaluate_test: true` to additionally evaluate
the configured held-out traces and write the `test_*` files.

To run another random seed, set `SEED`; the seed is included in the output
directory and study name, so separate seeds do not share SQLite checkpoints:

```bash
SEED=202 CONFIG_PATH=NewJaxleyMultiObjective/configs/mocma_features.yaml \
  conda run --name jaxley-mocma bash NewJaxleyMultiObjective/run_mocma.sh run
```

For Slurm, edit the seed list and resource requests in
`run_mocma_seeds.sbatch`, then submit an array such as:

```bash
sbatch --array=0-4 NewJaxleyMultiObjective/run_mocma_seeds.sbatch
```

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
2. The default configuration optimizes trace indices 2 and 4 independently
within each protocol. The test indices 1 and 3 remain configured for optional
held-out evaluation, but it is disabled in the supplied training configs. Set
`PYTHON_EXECUTABLE` when the Jaxley environment is not the shell's default
Python.

The time-course fitness window is protocol-specific: 200 ms before through
600 ms after the depolarizing step, and 200 ms before through 200 ms after the
hyperpolarizing pulse. If a recording starts before the requested pre-window,
the available portion is used and recorded in the trace metadata.

The supplied configs use `objectives.scales.ap_count: 0.25`, making AP-count
errors four times stronger in MOCMA's hypervolume-based elite selection. AP
count remains an independent Pareto objective; this does not create a strict
lexicographic priority. `parallel_workers: 6` evaluates six MOCMA trials at a
time on CPU. Reduce it if memory or CPU capacity is limited.

The hard spike guard returns a loss of `100000` for every objective when a
depolarizing simulation spikes outside its stimulus interval or a
hyperpolarizing simulation spikes anywhere in the recorded segment.

## Features and loss calculation

The supplied training configuration uses traces 2 and 4 from both protocols:
two depolarizing and two hyperpolarizing traces. Traces 1 and 3 are held out
and are only evaluated when `evaluate_test: true`. Each feature is extracted
from the same stimulus epoch metadata used by the Jaxley trace loader.

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

### Eleven objectives

All eleven objectives are minimized. For each scalar feature, the loss is the
mean over applicable training traces of

```text
((simulated_feature - experimental_feature) / objective_scale)²
```

The configured scales are in `objectives.scales`; for example,
`ap_count: 0.25` makes a one-AP error contribute `((1 / 0.25)²) = 16`
before trace averaging, making AP count more influential in MOCMA's
hypervolume calculations. It remains a Pareto objective rather than a strict
lexicographic priority. Missing or invalid scalar features add
`invalid_feature_penalty` (`1e6`).

The eleventh objective is `trajectory_overlap`. It does not subtract the
trace mean and therefore cannot solve a voltage mismatch by drifting toward
the mean. At each time sample it computes the local agreement

```text
overlap(t) = exp(-0.5 * ((V_sim(t) - V_exp(t)) / overlap_sigma_mV)²)
trajectory_loss = 1 - weighted_average(overlap(t))
```

The average is calculated separately over baseline, stimulus, recovery, and
the first 100 ms after depolarizing-step onset, then combined using
`overlap_window_weights`. In the supplied configs, samples within ±3 ms of an
experimentally detected depolarizing AP peak receive weight 4. This makes AP
timing and waveform overlap matter more without allowing the voltage objective
to replace the explicit AP-count objective.

Two hard constraints prevent clearly unusable solutions. If a depolarizing
trace spikes outside its stimulus interval, or a hyperpolarizing trace spikes
anywhere, every objective is set to `spike_violation_loss` (`1e5`). Also, if
the simulated AP count differs from the experimental count by more than
`ap_count_tolerance` (3 APs by default) on any depolarizing training trace,
every objective is set to `ap_count_violation_loss` (`1e5`). The details of
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

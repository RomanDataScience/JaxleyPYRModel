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

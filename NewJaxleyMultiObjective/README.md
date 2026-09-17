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

Outputs are written under `NewJaxleyMultiObjective/runs/`. Each Pareto solution
contains its parameters, optimization-trace features/predictions, objective
ranking, and a figure with full traces plus the first 100 ms after
depolarizing-step onset. Set `evaluate_test: true` to additionally evaluate
the configured held-out traces and write the `test_*` files.

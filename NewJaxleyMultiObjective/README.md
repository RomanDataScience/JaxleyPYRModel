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

For a quick check, copy the config and reduce `multi_objective.trials` to 1 or
2. The normal configuration uses all traces for the selected cell and 4,000
trials. Set `PYTHON_EXECUTABLE` when the Jaxley environment is not the shell's
default Python.

Outputs are written under `NewJaxleyMultiObjective/runs/`. Each Pareto solution
contains its parameters, features, predictions, objective ranking, and a
figure with full traces plus the first 100 ms after depolarizing-step onset.

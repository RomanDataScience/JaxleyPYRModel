# Calibrated-parameter importance analyses

`analyze_importance.py` reconstructs an in-memory Optuna study from the
evaluated CMA-ES populations saved by the optimizer. It analyzes the model
parameters for all three stages:

- Stage 0: passive-only calibration;
- Stage 1: full-model hyperpolarizing calibration;
- Stage 2: full-model depolarizing calibration across basins and seeds.

The main importance score is Optuna fANOVA, following the approach in the
[Optuna visualization tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html).
Parameters are passed to the estimator in normalized `[0, 1]` coordinates so
their physical units do not dominate the calculation. Reports and slice plots
also contain the physical calibrated values. PED-ANOVA and absolute rank
correlation are included as secondary diagnostics. If `scikit-learn` is not
available at runtime, the report automatically uses Optuna PED-ANOVA as the
primary score and records that fallback in `summary.json`.

Run it from `NEURON_optim`:

```bash
/Users/romanbaravalle/miniconda3/envs/Jaxley/bin/python \
  -m hyperparameter_analyses.analyze_importance \
  --config configs/.smoke-config.DDZfoG \
  --run-dir runs/smoke_full_workers16 \
  --output-dir hyperparameter_analyses/smoke_full_workers16
```

Each available stage writes:

- `<stage>_importance.json`: complete metadata, fANOVA/PED-ANOVA scores,
  physical ranges, and the best-candidate value;
- `<stage>_importance.csv`: sortable table of the same parameter results;
- `<stage>_importance.png`: importance bar chart;
- `<stage>_top_slices.png`: loss versus physical value for the most important
  parameters;
- `<stage>_contour.png`: Optuna-style pairwise loss contours for the four most
  important varying parameters;
- `<stage>_history.png`: best, median, and 10th–90th percentile loss by saved
  generation.

`summary.json` combines the stage reports. Stage 2 may contain many more
evaluations than the smoke run; the default `--max-records 50000` retains up to
the best three candidates from every saved population and samples the remainder
deterministically while respecting the total cap. Use `--max-records 0` to
analyze every finite candidate when memory and runtime permit.

Use `--contour-top-k N` to change the number of parameters in the pairwise
contour grid. The contour axes use physical calibrated units, while color shows
log-scaled loss relative to the best candidate. Since all pairwise combinations
would be unwieldy for the 43-parameter full model, the default focuses on the
most important varying parameters.

The Stage 2 result is a pooled analysis across basin/seed studies. Basin and
seed identifiers are retained as metadata but are not treated as calibrated
parameters. Consequently, Stage 2 importance should be interpreted as the
importance of parameters across the starting-solution ensemble, not as a
causal estimate from a single basin.

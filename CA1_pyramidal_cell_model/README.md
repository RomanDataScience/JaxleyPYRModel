# CA1 pyramidal cell model: fitting and parameter degeneracy

A detailed NEURON model of a CA1 pyramidal cell (reconstructed morphology,
spines folded in with an F factor), fitted with CMA-ES to whole-cell
current-clamp recordings. Analysis scripts then ask which parameter sets fit
the recordings, and which channels the recordings actually constrain.

Everything needed is in this folder: model (HOC + MOD files), recordings,
fitting code, analyses and SLURM scripts.

## Setup

```bash
conda create -n ca1 python=3.12 && conda activate ca1
pip install -r requirements.txt
```

The MOD files are compiled automatically with `nrnivmodl` into `.cache/mod/<arch>/`
the first time the model is built.

## Quick start

All commands run from this folder.

```bash
# Build the cell, score the default parameters on one recorded cell, exit
python run_cma.py --config configs/cma_m20240527cd.yaml --validate-only

# One generation of two candidates, written to runs/smoke_test
python run_cma.py --config configs/cma_m20240527cd.yaml --smoke-test

# A full fit (resumes from runs/<name>/cma_state.npz if interrupted)
python run_cma.py --config configs/cma_m20240527cd.yaml --seed 1 --generations 15 \
    --population-size 30 --workers 10
```

## Layout

| Path | Contents |
|---|---|
| `ca1_model.py` | Builds the cell from the HOC loader, applies a parameter set, replays a recorded current |
| `ca1_parameters.py` | The 21 fitted parameters: defaults and bounds (plus optional `cm`, `Ra`) |
| `*.hoc`, `mod_files/` | Morphology, biophysics, spines and channel mechanisms |
| `fitting/` | Trace loader, joint objective, bounded CMA-ES, fit plots |
| `configs/` | One fit configuration per recorded cell |
| `Experimental_currentClamp_Analysis/` | Recordings, trace segmentation and feature extraction |
| `slurm/` | Cluster scripts for the multi-seed study |
| `runs/` | Fit outputs (one folder per run) |
| `plots/` | Analysis figures |

## Fitting

`run_cma.py` fits all 21 parameters to four depolarizing steps and one
hyperpolarizing pulse of one cell (`fitting/objective.py`: voltage trajectory,
baseline, spike count, rate, timing and height). Each run writes to
`runs/<config output_dir>[_seed<N>]/`:

- `best_parameters.json`, `best_fit.png`, `depolarizing_steps_fit.png`
- `evaluations.jsonl`: every evaluated candidate with its parameters, loss and
  per-trace loss terms (the input of the landscape analysis)
- `plots/generation_NNNN/candidates.png` and `candidates_onset.png`: top
  candidates per generation, full traces and the first 150 ms of each step

Parameters can be fixed, excluded or given other bounds in the config's
`parameters` section.

## Analyses

| Script | Question | Output |
|---|---|---|
| `plot_mechanisms.py [--parameters runs/<run>/best_parameters.json]` | Where is each channel, and how dense? | `plots/ca1_mechanism_distribution*.png` |
| `analyze_landscape.py runs/<cell>_seed*` | Which parameter sets fit, and how do good fits differ? Loss vs parameter, UMAP/PCA by loss, good-fit profiles and correlations, fANOVA importance | `plots/landscape_<cell>/` |
| `sensitivity.py runs/<run>` | Which parameters change the dynamics in the same way, and which ones do these recordings constrain? Local sensitivity, parameter clustering, stiff/sloppy directions, core / compensable / irrelevant classification | `plots/sensitivity_<cell>/` |

`sensitivity.py` costs 43 simulations; `--replot` redraws from the saved
results, e.g. after changing `--threshold` or `--max-ratio`.

## Multi-seed study on a cluster

```bash
CONDA_ENV=ca1 SBATCH_EXTRA="--partition=... --account=..." bash slurm/submit_all.sh
```

This compiles the mechanisms once, runs 10 seeds per cell (15 generations x 30
candidates, 16 cores each) and then `analyze_landscape.py` over all seeds of
each cell. `GENERATIONS` and `POPULATION_SIZE` override the defaults; a seed
that hits the time limit resumes when resubmitted.

## Recordings

`Experimental_currentClamp_Analysis/Segmented_Traces/<cell>/` holds the
segmented traces used by the fits: `depolarizing_step/` (200 ms before, 300 ms
step, 700 ms after) and `hyperpolarizing_pulse/` (500 ms before, 50 ms pulse,
500 ms after), each as `<trace>_<protocol>_{t_ms,v,i}.txt`. A new cell needs
its segmented traces there and a config in `configs/`.

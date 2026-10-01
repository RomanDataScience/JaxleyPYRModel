#!/usr/bin/env bash
# Shared environment for the CA1 SLURM jobs; sourced, not run.
#
# Set CONDA_ENV to activate a conda env (it must have NEURON, numpy, scipy,
# matplotlib, pyyaml, scikit-learn and umap-learn), or point PYTHON_EXECUTABLE
# at the right interpreter.

if [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV is set but conda is not on PATH; load the cluster's conda module first." >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
CA1_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-$$}"}
# One NEURON process per core: keep numerical libraries single-threaded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "${MPLCONFIGDIR}"
cd "${CA1_DIR}"

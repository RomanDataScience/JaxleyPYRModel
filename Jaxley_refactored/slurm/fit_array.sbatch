#!/usr/bin/env bash
#SBATCH --job-name=lsu1-m20260331b
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --signal=B:USR1@120

# Fit cell m20260331b with the local Adam settings in the standalone LSU_1
# configuration. Adjust the SBATCH resource directives for the local cluster.
set -euo pipefail

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${SLURM_TMPDIR:-/tmp}/matplotlib"}
export RUN_SCRATCH=${RUN_SCRATCH:-"${SLURM_TMPDIR:-/tmp}/jaxley-${SLURM_JOB_ID}"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p logs "${MPLCONFIGDIR}" "${RUN_SCRATCH}"

cd "${PROJECT_DIR}"

export PYTHON_EXECUTABLE
srun bash scripts/run_full_fitting.sh \
  --config configs/LSU_1_cma_adam.yaml \
  --cells m20260331b

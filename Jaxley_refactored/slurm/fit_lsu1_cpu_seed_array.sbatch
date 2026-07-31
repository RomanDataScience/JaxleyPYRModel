#!/usr/bin/env bash
#SBATCH --job-name=lsu1-cpu-seeds
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --signal=B:USR1@120

# Run one independent LSU_1 fit for each seed in SEEDS. The array range above
# must contain exactly one task per SEEDS entry.
set -euo pipefail

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${SLURM_TMPDIR:-/tmp}/matplotlib"}
export RUN_SCRATCH=${RUN_SCRATCH:-"${SLURM_TMPDIR:-/tmp}/jaxley-${SLURM_JOB_ID}"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
export PYTHON_EXECUTABLE

mkdir -p "${MPLCONFIGDIR}" "${RUN_SCRATCH}"
cd "${PROJECT_DIR}"

srun bash scripts/run_full_fitting.sh \
  --config configs/LSU_1_cma_adam.yaml \
  --cells m20260331b \
  --seed "${SEED}" \
  --max-parallel-cells 1

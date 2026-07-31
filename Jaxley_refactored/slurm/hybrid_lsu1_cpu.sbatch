#!/usr/bin/env bash
#SBATCH --job-name=lsu1-hybrid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --signal=B:USR1@120

# Serial population baseline. CMA candidates are evaluated sequentially while
# JAX/XLA may use the allocated CPU cores within each simulation.
set -euo pipefail

SEED=${SEED:-0}
CELL_ID=${CELL_ID:-m20260331b}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export PYTHON_EXECUTABLE
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${SLURM_TMPDIR:-/tmp}/matplotlib"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT_DIR}"

srun bash scripts/run_hybrid_fitting.sh \
  --config configs/LSU_1_cma_adam.yaml \
  --cell "${CELL_ID}" \
  --seed "${SEED}"

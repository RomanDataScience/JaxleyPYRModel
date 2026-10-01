#!/usr/bin/env bash
#SBATCH --job-name=ca1-cma-seeds
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

# One independent CMA-ES fit per seed of one cell. Every evaluation is logged
# to <run>/evaluations.jsonl. The array range must have one task per SEEDS entry.
#
#   sbatch --export=ALL,CONFIG=configs/cma_m20240527cd.yaml slurm/cma_seeds.sbatch
#
# A task that hits the time limit can be resubmitted: the run resumes from
# its cma_state.npz checkpoint after the last completed generation.
set -euo pipefail
source "${CA1_SLURM_DIR:-$(dirname -- "$0")}/common.sh"

SEEDS=(1 2 3 4 5 6 7 8 9 10)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
CONFIG=${CONFIG:?set CONFIG to a config path relative to CA1_pyramidal_cell_model}
GENERATIONS=${GENERATIONS:-15}
POPULATION_SIZE=${POPULATION_SIZE:-30}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}

echo "config=${CONFIG} seed=${SEED} generations=${GENERATIONS} population=${POPULATION_SIZE} workers=${WORKERS}"
srun "${PYTHON_EXECUTABLE}" run_cma.py \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --generations "${GENERATIONS}" \
  --population-size "${POPULATION_SIZE}" \
  --workers "${WORKERS}"

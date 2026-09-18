#!/usr/bin/env bash
# Example Slurm array launcher. Adjust the resource requests for your cluster.
# Submit with:
#   sbatch --array=0-4 NewJaxleyMultiObjective/run_mocma_seeds.sbatch

#SBATCH --job-name=jaxley-mocma
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jaxley-mocma}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/mocma_features.yaml}"

# Replace these with the seeds you want. The array index selects one seed.
SEEDS=(101 202 303 404 505)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#SEEDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_INDEX} is outside the configured seed list." >&2
  exit 1
fi
SEED="${SEEDS[${TASK_INDEX}]}"

# On clusters where Conda is provided by a module, load it before this block.
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not on PATH. Load the cluster's Conda module first." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"
CONFIG_PATH="${CONFIG_PATH}" \
SEED="${SEED}" \
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}" \
bash "${SCRIPT_DIR}/run_mocma.sh" run

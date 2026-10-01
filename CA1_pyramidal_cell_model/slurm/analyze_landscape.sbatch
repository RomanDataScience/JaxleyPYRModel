#!/usr/bin/env bash
#SBATCH --job-name=ca1-landscape
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Loss-landscape / UMAP analysis over every seed run of one cell.
#
#   sbatch --export=ALL,RUN_PREFIX=runs/cma_m20240527cd slurm/analyze_landscape.sbatch
set -euo pipefail
source "${CA1_SLURM_DIR:-$(dirname -- "$0")}/common.sh"

RUN_PREFIX=${RUN_PREFIX:?set RUN_PREFIX, e.g. runs/cma_m20240527cd}
shopt -s nullglob
RUNS=("${RUN_PREFIX}"_seed*/)
if (( ${#RUNS[@]} == 0 )); then
  echo "No runs match ${RUN_PREFIX}_seed*" >&2
  exit 1
fi
# Skip seeds that never finished a generation.
READY=()
for run in "${RUNS[@]}"; do
  [[ -f "${run}/evaluations.jsonl" ]] && READY+=("${run}")
done
echo "Analysing ${#READY[@]} of ${#RUNS[@]} runs"
"${PYTHON_EXECUTABLE}" analyze_landscape.py "${READY[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=jaxley-fit
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --signal=B:USR1@120

# Launch one independent fit per SLURM array task. The manifest is a TSV with
# columns: config, seed, run_name. Blank lines and lines beginning with # are
# ignored. Adjust the SBATCH resource directives for the local cluster.
set -euo pipefail

MANIFEST=${1:?"usage: sbatch fit_array.sbatch MANIFEST.tsv"}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MANIFEST_DIR=$(cd -- "$(dirname -- "${MANIFEST}")" && pwd)
TASK_LINE=$(awk -v row="$((SLURM_ARRAY_TASK_ID + 1))" \
  'BEGIN {FS=\"\\t\"} !/^[[:space:]]*(#|$)/ {n++; if (n == row) {print; exit}}' \
  "${MANIFEST}")

if [[ -z "${TASK_LINE}" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r CONFIG SEED RUN_NAME <<< "${TASK_LINE}"
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${MANIFEST_DIR}/${CONFIG}"
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${SLURM_TMPDIR:-/tmp}/matplotlib"}
export RUN_SCRATCH=${RUN_SCRATCH:-"${SLURM_TMPDIR:-/tmp}/jaxley-${SLURM_JOB_ID}"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p logs "${MPLCONFIGDIR}" "${RUN_SCRATCH}"

srun "${PYTHON_EXECUTABLE}" -m jaxley_refactored.cli.bootstrap fit \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --run-name "${RUN_NAME}"

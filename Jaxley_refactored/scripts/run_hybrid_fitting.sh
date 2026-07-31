#!/usr/bin/env bash
#
# Run the serial CMA-ES -> Adam -> backtracking pipeline for one cell.
#
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
CONFIG="${PROJECT_DIR}/configs/LSU_1_cma_adam.yaml"
CELL_ID=m20260331b
SEED=0
RUN_NAME=

usage() {
  printf 'Usage: %s [--config PATH] [--cell ID] [--seed N] [--run-name NAME]\n' "$0"
}

while (($#)); do
  case "$1" in
    --config) CONFIG=${2:?"--config requires a path"}; shift 2 ;;
    --cell) CELL_ID=${2:?"--cell requires an ID"}; shift 2 ;;
    --seed) SEED=${2:?"--seed requires an integer"}; shift 2 ;;
    --run-name) RUN_NAME=${2:?"--run-name requires a value"}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/jaxley-matplotlib"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT_DIR}"

command=(
  "${PYTHON_EXECUTABLE}" -m jaxley_refactored.cli.bootstrap hybrid-fit
  --config "${CONFIG}"
  --cell-id "${CELL_ID}"
  --seed "${SEED}"
)
if [[ -n "${RUN_NAME}" ]]; then
  command+=(--run-name "${RUN_NAME}")
fi
"${command[@]}"

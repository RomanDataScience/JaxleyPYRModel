#!/usr/bin/env bash
#
# Run the same cell/seed with the shipped loss configurations.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
CELL_ID=m20240527cd
SEED=0
EPOCHS=
MAX_STEPS=

usage() {
  printf 'Usage: %s [--cell ID] [--seed N] [--epochs N] [--max-steps N]\n' "$0"
}

while (($#)); do
  case "$1" in
    --cell)
      CELL_ID=${2:?"--cell requires an ID"}
      shift 2
      ;;
    --seed)
      SEED=${2:?"--seed requires an integer"}
      shift 2
      ;;
    --epochs)
      EPOCHS=${2:?"--epochs requires an integer"}
      shift 2
      ;;
    --max-steps)
      MAX_STEPS=${2:?"--max-steps requires an integer"}
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/jaxley-matplotlib"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p "${MPLCONFIGDIR}"

COMPARISON_ID=$(date '+%Y%m%d-%H%M%S')
LOG_DIR="${PROJECT_DIR}/runs/loss_comparisons/${COMPARISON_ID}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "${PROJECT_DIR}/configs/losses/voltage_mse.yaml"
  "${PROJECT_DIR}/configs/losses/pseudo_huber.yaml"
  "${PROJECT_DIR}/configs/losses/huber_derivative_passive.yaml"
)

for config in "${CONFIGS[@]}"; do
  loss_name=$(basename "${config}" .yaml)
  run_name="loss-${CELL_ID}-${loss_name}-seed${SEED}-${COMPARISON_ID}"
  command=(
    "${PYTHON_EXECUTABLE}" -m jaxley_refactored.cli.bootstrap fit
    --config "${config}"
    --cell-id "${CELL_ID}"
    --seed "${SEED}"
    --run-name "${run_name}"
  )
  [[ -n "${EPOCHS}" ]] && command+=(--epochs "${EPOCHS}")
  [[ -n "${MAX_STEPS}" ]] && command+=(--max-steps "${MAX_STEPS}")

  echo "Starting ${loss_name}; run=${run_name}"
  "${command[@]}" 2>&1 | tee "${LOG_DIR}/${loss_name}.log"
done

echo "Loss comparison complete. Logs: ${LOG_DIR}"

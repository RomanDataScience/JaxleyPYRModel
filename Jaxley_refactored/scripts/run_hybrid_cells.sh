#!/usr/bin/env bash
#
# Launch the hybrid CMA-ES -> Adam -> backtracking pipeline for selected cells.
#
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPOSITORY_DIR=$(cd -- "${PROJECT_DIR}/.." && pwd)

CONFIG="${PROJECT_DIR}/configs/search/LSU_1_cma_adam.yaml"
MANIFEST="${REPOSITORY_DIR}/JaxleyModel/Experimental_currentClamp_Analysis/Segmented_Traces/segment_metadata.csv"
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
CELL_SELECTION=all
SEED=0
MAX_PARALLEL_CELLS=1
LIST_CELLS=false

usage() {
  printf 'Run the hybrid pipeline for selected recorded cells.\n\n'
  printf 'Usage: %s [options]\n\n' "$0"
  printf '%s\n' \
    '  --config PATH              Hybrid YAML configuration' \
    '  --manifest PATH            Segmented-trace CSV manifest' \
    '  --cells all|ID[,ID...]     Cell selector (default: all)' \
    '  --seed N                   Optimizer/CMA seed (default: 0)' \
    '  --max-parallel-cells N     Concurrent cell pipelines (default: 1)' \
    '  --list-cells               Print available cell IDs and exit' \
    '  --help                     Show this help'
}

while (($#)); do
  case "$1" in
    --config)
      CONFIG=${2:?"--config requires a path"}
      shift 2
      ;;
    --manifest)
      MANIFEST=${2:?"--manifest requires a path"}
      shift 2
      ;;
    --cells)
      CELL_SELECTION=${2:?"--cells requires all or a comma-separated list"}
      shift 2
      ;;
    --seed)
      SEED=${2:?"--seed requires a non-negative integer"}
      shift 2
      ;;
    --max-parallel-cells)
      MAX_PARALLEL_CELLS=${2:?"--max-parallel-cells requires an integer"}
      shift 2
      ;;
    --list-cells)
      LIST_CELLS=true
      shift
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

if [[ ! -f "${CONFIG}" ]]; then
  echo "Configuration does not exist: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest does not exist: ${MANIFEST}" >&2
  exit 2
fi
# The per-cell launcher changes into the project directory. Resolve user-supplied
# relative paths now so they remain valid after that directory change.
CONFIG=$(cd -- "$(dirname -- "${CONFIG}")" && pwd)/$(basename -- "${CONFIG}")
MANIFEST=$(cd -- "$(dirname -- "${MANIFEST}")" && pwd)/$(basename -- "${MANIFEST}")
case "${SEED}" in
  ''|*[!0-9]*)
    echo "--seed must be a non-negative integer" >&2
    exit 2
    ;;
esac
case "${MAX_PARALLEL_CELLS}" in
  ''|*[!0-9]*|0)
    echo "--max-parallel-cells must be a positive integer" >&2
    exit 2
    ;;
esac

AVAILABLE_CELLS=$(awk -F, 'NR > 1 && $1 != "" {print $1}' "${MANIFEST}" | sort -u)
if [[ -z "${AVAILABLE_CELLS}" ]]; then
  echo "No cells found in ${MANIFEST}" >&2
  exit 2
fi
if [[ "${LIST_CELLS}" == true ]]; then
  printf '%s\n' "${AVAILABLE_CELLS}"
  exit 0
fi

if [[ "${CELL_SELECTION}" == all ]]; then
  CELL_LIST=${AVAILABLE_CELLS}
else
  CELL_LIST=$(printf '%s\n' "${CELL_SELECTION}" | tr ',' '\n')
fi

while IFS= read -r cell; do
  [[ -z "${cell}" ]] && continue
  if [[ ! "${cell}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Unsafe cell identifier: ${cell}" >&2
    exit 2
  fi
  if ! printf '%s\n' "${AVAILABLE_CELLS}" | awk -v requested="${cell}" '$0 == requested {found=1} END {exit !found}'; then
    echo "Cell ${cell} is not present in ${MANIFEST}" >&2
    exit 2
  fi
done <<< "${CELL_LIST}"

export PYTHON_EXECUTABLE
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/jaxley-matplotlib"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p "${MPLCONFIGDIR}"

PIDS=()
PID_CELLS=()
FAILED=0

wait_for_oldest() {
  local pid=${PIDS[0]}
  local cell=${PID_CELLS[0]}
  if wait "${pid}"; then
    echo "Completed hybrid pipeline for ${cell}"
  else
    echo "Hybrid pipeline failed for ${cell}" >&2
    FAILED=1
  fi
  PIDS=("${PIDS[@]:1}")
  PID_CELLS=("${PID_CELLS[@]:1}")
}

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

trap 'terminate_children; exit 130' INT TERM HUP

while IFS= read -r cell; do
  [[ -z "${cell}" ]] && continue
  while ((${#PIDS[@]} >= MAX_PARALLEL_CELLS)); do
    wait_for_oldest
  done
  echo "Launching hybrid pipeline for ${cell} with seed ${SEED}"
  bash "${SCRIPT_DIR}/run_hybrid_fitting.sh" \
    --config "${CONFIG}" \
    --cell "${cell}" \
    --seed "${SEED}" &
  PIDS+=("$!")
  PID_CELLS+=("${cell}")
done <<< "${CELL_LIST}"

while ((${#PIDS[@]})); do
  wait_for_oldest
done

if ((FAILED)); then
  exit 1
fi

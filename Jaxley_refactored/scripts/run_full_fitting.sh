#!/usr/bin/env bash
#
# Fit every recorded cell in the segmented-trace manifest.
#
# Parallelism has two levels:
#   1. Within each cell, Jaxley's jitted vmap kernel simulates every trace in a
#      same-shape (dt, n_steps) bucket in parallel.
#   2. Optionally, independent cells can run as separate local processes with
#      --max-parallel-cells. Keep this at 1 on a single GPU or a memory-limited
#      workstation; vmap trace parallelism is still active.
#
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPOSITORY_DIR=$(cd -- "${PROJECT_DIR}/.." && pwd)

CONFIG="${PROJECT_DIR}/configs/runtimes/cpu_x64.yaml"
MANIFEST="${REPOSITORY_DIR}/JaxleyModel/Experimental_currentClamp_Analysis/Segmented_Traces/segment_metadata.csv"
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
CELL_SELECTION=all
MAX_PARALLEL_CELLS=1
SEED=0
EPOCHS=
MAX_STEPS=
DRY_RUN=false

usage() {
  printf 'Fit all recorded cells with jitted vmap trace parallelism.\n\n'
  printf 'Usage: %s [options]\n\n' "${BASH_SOURCE[0]}"
  printf '%s\n' \
    '  --config PATH              Fit/runtime YAML (default: CPU x64 config)' \
    '  --manifest PATH            Segmented-trace CSV manifest' \
    '  --cells all|ID[,ID...]     Cells to fit (default: all manifest cells)' \
    '  --max-parallel-cells N     Concurrent cell fits (default: 1)' \
    '  --seed N                   Optimizer/run seed (default: 0)' \
    '  --epochs N                 Override configured epoch count' \
    '  --max-steps N              Fit only the first N samples of every trace' \
    '  --dry-run                  Validate/build each fit without optimizing' \
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
    --max-parallel-cells)
      MAX_PARALLEL_CELLS=${2:?"--max-parallel-cells requires an integer"}
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
    --dry-run)
      DRY_RUN=true
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
case "${MAX_PARALLEL_CELLS}" in
  ''|*[!0-9]*|0)
    echo "--max-parallel-cells must be a positive integer" >&2
    exit 2
    ;;
esac
case "${SEED}" in
  ''|*[!0-9]*)
    echo "--seed must be a non-negative integer" >&2
    exit 2
    ;;
esac
if [[ -n "${EPOCHS}" ]]; then
  case "${EPOCHS}" in
    *[!0-9]*|0)
      echo "--epochs must be a positive integer" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${MAX_STEPS}" ]]; then
  case "${MAX_STEPS}" in
    *[!0-9]*|0|1)
      echo "--max-steps must be an integer greater than one" >&2
      exit 2
      ;;
  esac
fi

if [[ "${CELL_SELECTION}" == all ]]; then
  CELL_LIST=$(awk -F, 'NR > 1 && $1 != "" {print $1}' "${MANIFEST}" | sort -u)
else
  CELL_LIST=$(printf '%s\n' "${CELL_SELECTION}" | tr ',' '\n')
fi
if [[ -z "${CELL_LIST}" ]]; then
  echo "No cells were selected from ${MANIFEST}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/jaxley-matplotlib"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p "${MPLCONFIGDIR}"

LAUNCH_ID=$(date '+%Y%m%d-%H%M%S')
LOG_DIR="${PROJECT_DIR}/runs/launcher_logs/${LAUNCH_ID}"
mkdir -p "${LOG_DIR}"
echo "Live per-cell logs: ${LOG_DIR}"

PIDS=()
PID_CELLS=()
FAILED=0

wait_for_oldest() {
  local pid=${PIDS[0]}
  local cell=${PID_CELLS[0]}
  if wait "${pid}"; then
    echo "Completed ${cell}; log: ${LOG_DIR}/${cell}.log"
  else
    echo "Fit failed for ${cell}; log: ${LOG_DIR}/${cell}.log" >&2
    FAILED=1
  fi
  PIDS=("${PIDS[@]:1}")
  PID_CELLS=("${PID_CELLS[@]:1}")
}

launch_cell() {
  local cell=$1
  local run_name="full-${cell}-seed${SEED}-${LAUNCH_ID}"
  local command=(
    "${PYTHON_EXECUTABLE}"
    -m jaxley_refactored.cli.bootstrap
    fit
    --config "${CONFIG}"
    --cell-id "${cell}"
    --seed "${SEED}"
    --run-name "${run_name}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    command+=(--epochs "${EPOCHS}")
  fi
  if [[ -n "${MAX_STEPS}" ]]; then
    command+=(--max-steps "${MAX_STEPS}")
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    command+=(--dry-run)
  fi

  echo "Launching ${cell} as ${run_name}"
  (
    set -o pipefail
    {
      echo "Command: ${command[*]}"
      echo "Trace batching: natural-shape jit(vmap) buckets sharing one parameter vector"
      "${command[@]}"
    } 2>&1 | tee "${LOG_DIR}/${cell}.log"
  ) &
  PIDS+=("$!")
  PID_CELLS+=("${cell}")
}

while IFS= read -r cell; do
  [[ -z "${cell}" ]] && continue
  if [[ ! "${cell}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Unsafe cell identifier in manifest: ${cell}" >&2
    exit 2
  fi
  while ((${#PIDS[@]} >= MAX_PARALLEL_CELLS)); do
    wait_for_oldest
  done
  launch_cell "${cell}"
done <<< "${CELL_LIST}"

while ((${#PIDS[@]})); do
  wait_for_oldest
done

if ((FAILED)); then
  echo "One or more fits failed. Inspect ${LOG_DIR}" >&2
  exit 1
fi

echo "All selected cell fits completed. Logs: ${LOG_DIR}"

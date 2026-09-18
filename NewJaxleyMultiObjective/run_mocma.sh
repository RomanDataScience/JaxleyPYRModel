#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/mocma_features.yaml}"

export PYTHONPATH="${SCRIPT_DIR}:${REPO_DIR}/Jaxley_refactored${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/jaxley_mocma_mpl}"
mkdir -p "${MPLCONFIGDIR}"

COMMAND="run"
if [[ "${1:-}" == "validate" || "${1:-}" == "run" ]]; then
  COMMAND="$1"
  shift
fi

if [[ -n "${SEED:-}" ]]; then
  if (( $# > 0 )); then
    exec "${PYTHON_EXECUTABLE}" -m newjaxley_multiobjective.cli "${COMMAND}" --config "${CONFIG_PATH}" "$@" --seed "${SEED}"
  fi
  exec "${PYTHON_EXECUTABLE}" -m newjaxley_multiobjective.cli "${COMMAND}" --config "${CONFIG_PATH}" --seed "${SEED}"
fi

exec "${PYTHON_EXECUTABLE}" -m newjaxley_multiobjective.cli "${COMMAND}" --config "${CONFIG_PATH}" "$@"

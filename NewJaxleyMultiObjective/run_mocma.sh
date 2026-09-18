#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/mocma_features.yaml}"

export PYTHONPATH="${SCRIPT_DIR}:${REPO_DIR}/Jaxley_refactored${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="Agg"
export NEURON_MODULE_OPTIONS="${NEURON_MODULE_OPTIONS:--nogui}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/jaxley_mocma_mpl}"
mkdir -p "${MPLCONFIGDIR}"

COMMAND="run"
if [[ "${1:-}" == "validate" || "${1:-}" == "run" ]]; then
  COMMAND="$1"
  shift
fi

CLI_OPTIONS=(--config "${CONFIG_PATH}")
if [[ -n "${SEED:-}" ]]; then
  CLI_OPTIONS+=(--seed "${SEED}")
fi

if [[ -n "${D_LAMBDA:-}" ]]; then
  CLI_OPTIONS+=(--d-lambda "${D_LAMBDA}")
fi

exec "${PYTHON_EXECUTABLE}" -m newjaxley_multiobjective.cli "${COMMAND}" "${CLI_OPTIONS[@]}" "$@"

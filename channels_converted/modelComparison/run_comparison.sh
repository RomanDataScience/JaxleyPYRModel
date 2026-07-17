#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export NEURON_MODULE_OPTIONS="${NEURON_MODULE_OPTIONS:--nogui}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/jaxley_mpl}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m channels_converted.modelComparison.compare_models "$@"

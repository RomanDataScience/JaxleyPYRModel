#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/romanbaravalle/miniconda3/envs/Jaxley/bin/python}"

cd "$REPO_ROOT"
mkdir -p /private/tmp/jaxley_mpl
NEURON_MODULE_OPTIONS="${NEURON_MODULE_OPTIONS:--nogui}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/jaxley_mpl}" \
  "$PYTHON_BIN" channels_converted/validation/compare_channel.py "$@"

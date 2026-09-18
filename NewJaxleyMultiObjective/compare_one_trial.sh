#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-1234}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/mocma_features.yaml}"

echo "Running one trial at d_lambda=0.3"
SEED="${SEED}" D_LAMBDA=0.3 CONFIG_PATH="${CONFIG_PATH}" \
  bash "${SCRIPT_DIR}/run_mocma.sh" run --trials 1

echo "Running one trial at d_lambda=0.1"
SEED="${SEED}" D_LAMBDA=0.1 CONFIG_PATH="${CONFIG_PATH}" \
  bash "${SCRIPT_DIR}/run_mocma.sh" run --trials 1

echo "Completed one-trial comparisons for seed ${SEED}."

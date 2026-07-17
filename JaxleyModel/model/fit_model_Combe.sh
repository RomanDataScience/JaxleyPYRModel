#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CELL_NAME="${CELL_NAME:-m20240527cd}"
TRACE_NAME="${TRACE_NAME:-v75ctrl}"
SEGMENT_NAME="${SEGMENT_NAME:-hyperpolarizing_step}"
EPOCHS="${EPOCHS:-50}"
DELTA_T="${DELTA_T:-0.05}"
D_LAMBDA="${D_LAMBDA:-0.3}"
PLOT_EVERY="${PLOT_EVERY:-1}"
LR_SCALE="${LR_SCALE:-0.03}"
LOSS_PRE_MS="${LOSS_PRE_MS:-100}"
LOSS_POST_MS="${LOSS_POST_MS:-400}"

cd "${SCRIPT_DIR}"

if [[ -n "${STIM_THRESHOLD_NA:-}" ]]; then
  "${PYTHON_BIN}" -u fit_model_Combe.py \
    --cell-name "${CELL_NAME}" \
    --trace-name "${TRACE_NAME}" \
    --segment-name "${SEGMENT_NAME}" \
    --epochs "${EPOCHS}" \
    --delta-t "${DELTA_T}" \
    --d-lambda "${D_LAMBDA}" \
    --plot-every "${PLOT_EVERY}" \
    --lr-scale "${LR_SCALE}" \
    --loss-pre-ms "${LOSS_PRE_MS}" \
    --loss-post-ms "${LOSS_POST_MS}" \
    --stim-threshold-nA "${STIM_THRESHOLD_NA}" \
    "$@"
else
  "${PYTHON_BIN}" -u fit_model_Combe.py \
    --cell-name "${CELL_NAME}" \
    --trace-name "${TRACE_NAME}" \
    --segment-name "${SEGMENT_NAME}" \
    --epochs "${EPOCHS}" \
    --delta-t "${DELTA_T}" \
    --d-lambda "${D_LAMBDA}" \
    --plot-every "${PLOT_EVERY}" \
    --lr-scale "${LR_SCALE}" \
    --loss-pre-ms "${LOSS_PRE_MS}" \
    --loss-post-ms "${LOSS_POST_MS}" \
    "$@"
fi

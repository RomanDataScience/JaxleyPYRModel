#!/usr/bin/env bash
set -euo pipefail

CELL_NAME="${1:-m20240527cd}"
TRACE_NAME="${2:-v75ctrl}"
SEGMENT_NAME="${3:-hyperpolarizing_step}"
EPOCHS="${4:-500}"
D_LAMBDA="${5:-0.4}"
PLOT_EVERY="${6:-1}"

python -u fit_model_with_gH_fit.py \
  --cell-name "$CELL_NAME" \
  --trace-name "$TRACE_NAME" \
  --segment-name "$SEGMENT_NAME" \
  --epochs "$EPOCHS" \
  --d-lambda "$D_LAMBDA" \
  --plot-every "$PLOT_EVERY"

LABEL="${CELL_NAME}_${TRACE_NAME}_${SEGMENT_NAME}"
if [[ "$PLOT_EVERY" -gt 0 ]]; then
  echo "Current epoch plots: Fit_Results/${LABEL}_current_by_epoch"
  echo "Best-so-far plots:   Fit_Results/${LABEL}_best_by_epoch"
else
  echo "Epoch plots disabled."
fi

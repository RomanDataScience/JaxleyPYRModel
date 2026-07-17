#!/usr/bin/env bash
set -euo pipefail

CELL_NAME="${1:-m20240527cd}"
TRACE_NAME="${2:-v75ctrl}"
SEGMENT_NAME="${3:-depolarizing_step}"
EPOCHS="${4:-10}"

python -u fit_model.py \
  --cell-name "$CELL_NAME" \
  --trace-name "$TRACE_NAME" \
  --segment-name "$SEGMENT_NAME" \
  --epochs "$EPOCHS"

LABEL="${CELL_NAME}_${TRACE_NAME}_${SEGMENT_NAME}"
echo "Current epoch plots: Fit_Results/${LABEL}_current_by_epoch"
echo "Best-so-far plots:   Fit_Results/${LABEL}_best_by_epoch"

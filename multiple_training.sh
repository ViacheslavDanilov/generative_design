#!/usr/bin/env bash

# :::::::::::::::::: Options ::::::::::::::::::
MODE="Explain"
TARGET="Lumen"
INPUT_SCALERS="Raw MinMax Standard Robust Power"
OUTPUT_SCALERS="Raw MinMax Standard Robust Power"
# :::::::::::::::::::::::::::::::::::::::::::::

for INPUT_SCALER in ${INPUT_SCALERS}; do
  for OUTPUT_SCALER in ${OUTPUT_SCALERS}; do
    python train.py \
    --mode ${MODE} \
    --target ${TARGET} \
    --feature_scale ${INPUT_SCALER} \
    --target_scale ${OUTPUT_SCALER}
  done
done
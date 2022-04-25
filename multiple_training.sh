#!/usr/bin/env bash

# :::::::::::::::::: Options ::::::::::::::::::
MODE="Explain"
TARGET="LMN"
INPUT_SCALERS="Raw MinMax Standard Robust Power"
OUTPUT_SCALERS="Raw MinMax Standard Robust Power"
METRICS="rmse mse mae mape"
# :::::::::::::::::::::::::::::::::::::::::::::

for INPUT_SCALER in ${INPUT_SCALERS}; do
  for OUTPUT_SCALER in ${OUTPUT_SCALERS}; do
    for METRIC in ${METRICS}; do
      python train.py \
      --mode ${MODE} \
      --target ${TARGET} \
      --feature_scale ${INPUT_SCALER} \
      --target_scale ${OUTPUT_SCALER} \
      --metric ${METRIC}
    done
  done
done
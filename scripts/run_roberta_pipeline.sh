#!/bin/bash
set -e

# Default to python3 in active virtualenv, or override via PYTHON=/path/to/python
PYTHON="${PYTHON:-python3}"
MODEL_NAME="roberta-large"
MODEL_DIR="./trained_absa_model_${MODEL_NAME}"
PREDICTIONS_FILE="outputs/predictions_${MODEL_NAME}.csv"

echo "=========================================================="
echo "Starting RoBERTa-large ABSA Pipeline"
echo "Using Python: $(${PYTHON} --version 2>&1)"
echo "=========================================================="
echo ""

echo "=== Step 1: Training Model ==="
${PYTHON} train.py \
    --model_name "${MODEL_NAME}" \
    --epochs 3 \
    --batch_size 8

echo ""
echo "=== Step 2: Generating Test Predictions ==="
${PYTHON} predict.py \
    --model_dir "${MODEL_DIR}" \
    --output_file "${PREDICTIONS_FILE}"

echo ""
echo "=== Step 3: Validating Prediction IDs ==="
${PYTHON} check_id.py data/contest2_test.csv "${PREDICTIONS_FILE}"

echo ""
echo "Pipeline completed successfully for ${MODEL_NAME}!"
echo "Predictions saved to: ${PREDICTIONS_FILE}"

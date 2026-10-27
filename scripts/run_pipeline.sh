#!/bin/bash
set -e

# Default to python3 in active virtualenv, or override via PYTHON=/path/to/python
PYTHON="${PYTHON:-python3}"
MODEL_NAME="${1:-microsoft/deberta-v3-large}"
SANITIZED_NAME="${MODEL_NAME//\//_}"
MODEL_DIR="./trained_absa_model_${SANITIZED_NAME}"
PREDICTIONS_FILE="outputs/predictions_${SANITIZED_NAME}.csv"

echo "=========================================================="
echo "Starting ABSA Pipeline for: ${MODEL_NAME}"
echo "Using Python: $(${PYTHON} --version 2>&1)"
echo "=========================================================="
echo ""

echo "=== Step 1: Training Model ==="
${PYTHON} train.py \
    --model_name "${MODEL_NAME}" \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4

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

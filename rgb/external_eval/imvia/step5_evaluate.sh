#!/bin/bash
# Step 5: Evaluate all models using cluster-based fall event F1.
#
# Uses inference/calculate_fall_accuracy_other_datasets.py on each model's
# predictions/ directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

EVAL_DIR="${SCRIPT_DIR}/eval_results"
EVAL_SCRIPT="${PROJ_ROOT}/inference/calculate_fall_accuracy_other_datasets.py"

echo "=========================================="
echo "Step 5: Evaluate all models on ImViA"
echo "=========================================="

MODELS=(
    "imvia_clip_meanpool"
    "imvia_dinov3_meanpool"
    "imvia_videomae_linear"
    "imvia_clip_cnn1d_fusion"
    "imvia_dinov3_cnn1d_fusion"
    "imvia_videomae_cnn1d_fusion"
)

for model in "${MODELS[@]}"; do
    pred_dir="${EVAL_DIR}/${model}/predictions"
    if [ ! -d "${pred_dir}" ]; then
        echo "  SKIP: ${model} (no predictions directory)"
        continue
    fi

    echo ""
    echo "=== ${model} ==="
    python3 "${EVAL_SCRIPT}" "${pred_dir}"
done

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="

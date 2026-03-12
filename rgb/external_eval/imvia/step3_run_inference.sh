#!/bin/bash
# Step 3: Run inference on ImViA with RGB-only and fusion models.
#
# Evaluates the 6 feature-based models reported in the paper's ImViA table:
#   RGB-only:  CLIP MeanPool, DINOv3 MeanPool, VideoMAE Linear
#   Fusion:    CLIP Concat, DINOv3 Concat, VideoMAE Concat
#
# For skeleton-only models (CNN1D, PoseC3D), use inference/infer_imvia.py
# and inference/infer_imvia_pyskl.py instead.
#
# Usage:
#   bash step3_run_inference.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

RUNS_DIR="/home/work/rgb/training/runs"
FEATURES_ROOT="${SCRIPT_DIR}/features"
EVAL_DIR="${SCRIPT_DIR}/eval_results"

PKL="${SCRIPT_DIR}/imvia_dataset.pkl"

cd "${PROJ_ROOT}/rgb/training"

run_model() {
    local run_name="$1"
    local pkl="$2"
    local feature_dir="$3"
    local out_name="$4"

    echo ""
    echo "--- ${out_name} ---"
    local out_dir="${EVAL_DIR}/${out_name}"
    mkdir -p "${out_dir}"

    python3 infer_pkl.py \
        --run_dir "${RUNS_DIR}/${run_name}" \
        --pkl_path "${pkl}" \
        --out_dict_dir "${out_dir}" \
        --out_dict_name result.pkl \
        --feature_dir "${feature_dir}"
}

echo "=========================================="
echo "Step 3: Run inference on ImViA"
echo "  6 models (RGB-only + Concat fusion)"
echo "=========================================="

# --- RGB-only models ---
run_model "normal_clip_meanpool_seq" "${PKL}" \
    "${FEATURES_ROOT}/imvia_clip" "imvia_clip_meanpool"

run_model "normal_dinov3_meanpool_seq" "${PKL}" \
    "${FEATURES_ROOT}/imvia_dinov3" "imvia_dinov3_meanpool"

run_model "normal_videomae_linear_seq" "${PKL}" \
    "${FEATURES_ROOT}/imvia_videomae" "imvia_videomae_linear"

# --- Concat fusion models ---
run_model "normal_clip_cnn1d_fusion_seq" "${PKL}" \
    "${FEATURES_ROOT}/imvia_clip" "imvia_clip_cnn1d_fusion"

run_model "normal_dinov3_cnn1d_fusion_seq" "${PKL}" \
    "${FEATURES_ROOT}/imvia_dinov3" "imvia_dinov3_cnn1d_fusion"

run_model "normal_videomae_cnn1d_fusion_seq" "${PKL}" \
    "${FEATURES_ROOT}/imvia_videomae" "imvia_videomae_cnn1d_fusion"

echo ""
echo "=========================================="
echo "All 6 models done!"
echo "  Results: ${EVAL_DIR}/imvia_*/result.pkl"
echo "=========================================="

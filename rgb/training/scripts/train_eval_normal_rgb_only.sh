#!/bin/bash
# =============================================================================
# Example: Train & evaluate an RGB-only model on the normal (in-lab) dataset
#
# Model: VideoMAE linear head (RGB-only, no skeleton fusion)
# Evaluates on: in-lab test set + non-lab (OOD) test set
#
# Usage:
#   bash scripts/train_eval_normal_rgb_only.sh
#   bash scripts/train_eval_normal_rgb_only.sh --eval-only
# =============================================================================

set -e
cd "$(dirname "$0")/.."

EVAL_ONLY=false
[[ "$1" == "--eval-only" ]] && EVAL_ONLY=true

# ---------------------------------------------------------------------------
# Paths — adjust these to match your setup
# ---------------------------------------------------------------------------
CALC_ACC="../../inference/calculate_accuracy.py"
TOTAL_SPLITS=3

# Normal (in-lab) dataset
NORMAL_PKL="/home/work/pyskl/Pkl/aic_normal_dataset_with_3d_480p.pkl"
NORMAL_FEAT="/home/work/rgb/feature_extraction/features/normal_videomae"

# Non-lab (OOD) test set
NONLAB_PKL="/home/work/pyskl/Pkl/aic_normal_test_set_with_split_3d_normalized.pkl"
NONLAB_FEAT="/home/work/rgb/feature_extraction/features/nonlab_test_videomae"

# Output directories
EVAL_INLAB="./eval_results"
EVAL_NONLAB="./eval_results_nonlab_test"

# ---------------------------------------------------------------------------
# Helper: inference + accuracy calculation
# ---------------------------------------------------------------------------
run_eval() {
    local name="$1" run_dir="$2" pkl_path="$3" eval_dir="$4" feature_dir="$5"
    local out_dir="$eval_dir/$name"

    if [ ! -f "$run_dir/best.pth" ]; then
        echo "  *** SKIPPED: $name — no best.pth in $run_dir"
        return
    fi

    echo ""
    echo ">>> Evaluating: $name → $eval_dir"
    python spawn_multiple_infer_pkl.py \
        --run_dir "$run_dir" \
        --pkl_path "$pkl_path" \
        --out_dict_dir "$out_dir" \
        --total_splits "$TOTAL_SPLITS" \
        --feature_dir "$feature_dir"

    python "$CALC_ACC" \
        --pkl_path "$out_dir/result.pkl" \
        --out_accuracy_results "$out_dir"
}

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if [ "$EVAL_ONLY" = false ]; then
    echo "========== Training: VideoMAE RGB-only (normal) =========="
    python train.py --config configs/normal/videomae_linear.py
fi

# ---------------------------------------------------------------------------
# Evaluate: in-lab + non-lab
# ---------------------------------------------------------------------------
echo ""
echo "========== In-Lab Evaluation =========="
mkdir -p "$EVAL_INLAB"
run_eval "videomae_linear" \
    "runs/normal_videomae_linear_seq" \
    "$NORMAL_PKL" "$EVAL_INLAB" "$NORMAL_FEAT"

echo ""
echo "========== Non-Lab (OOD) Evaluation =========="
mkdir -p "$EVAL_NONLAB"
run_eval "videomae_linear" \
    "runs/normal_videomae_linear_seq" \
    "$NONLAB_PKL" "$EVAL_NONLAB" "$NONLAB_FEAT"

echo ""
echo "Done! Reports in: $EVAL_INLAB/videomae_linear/report/"
echo "                   $EVAL_NONLAB/videomae_linear/report/"

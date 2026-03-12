#!/bin/bash
# =============================================================================
# Example: Train & evaluate an RGB-only model on the wheelchair dataset
#
# Model: VideoMAE linear head (RGB-only, no skeleton fusion)
# Evaluates on: wheelchair test set
#
# Usage:
#   bash scripts/train_eval_wheelchair_rgb_only.sh
#   bash scripts/train_eval_wheelchair_rgb_only.sh --eval-only
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

# Wheelchair dataset
WHEELCHAIR_PKL="/home/work/pyskl/Pkl/aic_wheelchair_dataset_with_3d.pkl"
WHEELCHAIR_FEAT="/home/work/rgb/feature_extraction/features/wheelchair_videomae"

# Output directory
EVAL_WHEELCHAIR="./eval_results_wheelchair"

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
    echo "========== Training: VideoMAE RGB-only (wheelchair) =========="
    python train.py --config configs/wheelchair/wheelchair_videomae_linear.py
fi

# ---------------------------------------------------------------------------
# Evaluate: wheelchair test set
# ---------------------------------------------------------------------------
echo ""
echo "========== Wheelchair Evaluation =========="
mkdir -p "$EVAL_WHEELCHAIR"
run_eval "wheelchair_videomae_linear" \
    "runs/wheelchair_videomae_linear_seq" \
    "$WHEELCHAIR_PKL" "$EVAL_WHEELCHAIR" "$WHEELCHAIR_FEAT"

echo ""
echo "Done! Reports in: $EVAL_WHEELCHAIR/wheelchair_videomae_linear/report/"

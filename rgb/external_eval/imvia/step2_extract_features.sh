#!/bin/bash
# Step 2: Extract DINOv3, VideoMAE, and CLIP features from ImViA videos.
#
# Uses extract_features_from_videos.py which reads original video files
# and applies person-centered bbox cropping on-the-fly (matching the main
# pipeline's crop logic: upscale -> bbox crop -> resize to 640x480).
#
# Usage:
#   bash step2_extract_features.sh --imvia_root /path/to/ImViA
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Parse arguments
IMVIA_ROOT=""
FEATURES_ROOT="${SCRIPT_DIR}/features"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --imvia_root) IMVIA_ROOT="$2"; shift 2 ;;
        --features_root) FEATURES_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$IMVIA_ROOT" ]; then
    echo "ERROR: --imvia_root is required"
    echo "Usage: bash step2_extract_features.sh --imvia_root /path/to/ImViA"
    exit 1
fi

PKL="${SCRIPT_DIR}/imvia_dataset.pkl"
if [ ! -f "$PKL" ]; then
    echo "ERROR: ${PKL} not found. Run step1_build_pkl.py first."
    exit 1
fi

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Step 2a: Extract DINOv3 features"
echo "  model=dinov3 (vitb16, 768-dim)"
echo "=========================================="
python3 extract_features_from_videos.py \
    --pkl_path "${PKL}" \
    --video_root "${IMVIA_ROOT}" \
    --model dinov3 \
    --output_dir "${FEATURES_ROOT}/imvia_dinov3" \
    --batch_size 16

echo ""
echo "=========================================="
echo "Step 2b: Extract VideoMAE features"
echo "  model=videomae (base, 768-dim, 48-frame window)"
echo "=========================================="
python3 extract_features_from_videos.py \
    --pkl_path "${PKL}" \
    --video_root "${IMVIA_ROOT}" \
    --model videomae \
    --output_dir "${FEATURES_ROOT}/imvia_videomae" \
    --batch_size 16

echo ""
echo "=========================================="
echo "Step 2c: Extract CLIP features"
echo "  model=clip (ViT-B/16, 512-dim)"
echo "=========================================="
python3 extract_features_from_videos.py \
    --pkl_path "${PKL}" \
    --video_root "${IMVIA_ROOT}" \
    --model clip \
    --output_dir "${FEATURES_ROOT}/imvia_clip" \
    --batch_size 16

echo ""
echo "=========================================="
echo "All features extracted!"
echo "  DINOv3:   ${FEATURES_ROOT}/imvia_dinov3/"
echo "  VideoMAE: ${FEATURES_ROOT}/imvia_videomae/"
echo "  CLIP:     ${FEATURES_ROOT}/imvia_clip/"
echo "=========================================="

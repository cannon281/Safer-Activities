"""
Extract frozen VideoMAE features from pre-cropped bbox clips.

For each frame position t in a video, extracts the feature from a 48-frame
window [t, t+48). The 48 raw frames are subsampled to 16, preprocessed,
and passed through the frozen VideoMAE backbone.

Efficient clip-batch:
  1. Read entire clip (300 frames) with OpenCV
  2. Batch-preprocess all 300 frames on GPU (resize, crop, normalize) once
  3. Build 48-frame sliding windows
  4. Select 16 frames per window (UniformTemporalSubsample indices)
  5. Batch through frozen backbone on GPU
  6. Across clips: only process CLIP_STRIDE (156) new positions per clip
     to avoid re-extracting features for overlapping regions

Output: one .npy per video, shape (total_frames, 768), dtype float32.
  - feature[t] = mean-pooled patch tokens for window [t, t+48), subsampled to 16 frames
  - Last 47 frames are zero-padded (no valid 48-frame window possible) 
  - Syncs with CLIP/DINOv3 features, keypoints, and labels by frame index

Model: MCG-NJU/videomae-base-finetuned-kinetics (768-dim, ImageNet normalization)

Usage:
    python extract_video_features.py \
        --pkl_path /path/to/dataset_480p.pkl \
        --clips_root /path/to/BBoxClips/normal \
        --output_dir ./features/normal_videomae \
        --batch_size 16
"""

import argparse
import gc
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ── Constants (must match extract_bbox_clips.py) ─────────────────────────
CLIP_CHUNK_SIZE = 300
CLIP_STRIDE = 156
CLIP_LENGTH = 48       # temporal window size
MODEL_FRAMES = 16      # frames after temporal subsampling

# Precomputed: the 16 frame indices that UniformTemporalSubsample(16) picks
# from a 48-frame window. Equivalent to torch.linspace(0, 47, 16).long()
SUBSAMPLE_INDICES = torch.linspace(0, CLIP_LENGTH - 1, MODEL_FRAMES).long()

# VideoMAE config
FEAT_DIM = 768
SIDE_SIZE = 224
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ── Model loading ────────────────────────────────────────────────────────

def load_videomae(device='cuda'):
    """Load frozen VideoMAE backbone. Returns (model, forward_fn)."""
    from transformers import VideoMAEModel

    model = VideoMAEModel.from_pretrained(
        'MCG-NJU/videomae-base-finetuned-kinetics')
    model = model.to(device).eval().half()
    for p in model.parameters():
        p.requires_grad = False

    def forward_fn(m, x):
        # x: (B, C, T, H, W) on GPU — rearrange to (B, T, C, H, W) for HF
        x = x.half()
        x = x.permute(0, 2, 1, 3, 4)  # → (B, T, C, H, W)
        out = m(x)
        # Mean pool over patch tokens → (B, 768)
        return out.last_hidden_state.mean(dim=1).float()

    return model, forward_fn


# ── Clip reading ─────────────────────────────────────────────────────────

def read_clip_frames(clip_path):
    """Read all frames from a clip file as numpy array.
    Returns: (N, H, W, 3) uint8, or None on failure.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0:
        return None
    return np.stack(frames)


# ── Clip discovery ───────────────────────────────────────────────────────

def get_clip_files(clips_root, video_stem):
    """Return sorted list of (start_frame, clip_path) tuples."""
    clip_dir = os.path.join(clips_root, video_stem)
    if not os.path.isdir(clip_dir):
        return []
    clips = []
    for f in os.listdir(clip_dir):
        m = re.match(r'clip_(\d+)\.mp4', f)
        if m:
            clips.append((int(m.group(1)), os.path.join(clip_dir, f)))
    clips.sort(key=lambda x: x[0])
    return clips


# ── GPU preprocessing ────────────────────────────────────────────────────

def preprocess_clip_gpu(clip_np, device='cuda'):
    """Batch-preprocess an entire clip on GPU.

    Args:
        clip_np: (N, H, W, 3) uint8 numpy array
    Returns:
        (3, N, 224, 224) float32 tensor on GPU, normalized
    """
    # (N, H, W, 3) → (N, 3, H, W) float [0, 1]
    clip_t = torch.from_numpy(clip_np).to(device).permute(0, 3, 1, 2).float()
    clip_t.div_(255.0)

    # Batch spatial: resize short side → center crop
    h, w = clip_t.shape[2], clip_t.shape[3]
    if h <= w:
        new_h = SIDE_SIZE
        new_w = int(round(w * SIDE_SIZE / h))
    else:
        new_w = SIDE_SIZE
        new_h = int(round(h * SIDE_SIZE / w))

    clip_t = TF.resize(clip_t, [new_h, new_w], antialias=True)
    clip_t = TF.center_crop(clip_t, [CROP_SIZE, CROP_SIZE])
    clip_t = TF.normalize(clip_t, mean=MEAN, std=STD)

    # (N, 3, crop, crop) → (3, N, crop, crop) for temporal indexing
    return clip_t.permute(1, 0, 2, 3)


# ── Per-video processing ─────────────────────────────────────────────────

def process_video(clips, total_frames, model, forward_fn,
                  device='cuda', batch_size=16):
    """Extract features for all valid positions in one video."""
    features = np.zeros((total_frames, FEAT_DIM), dtype=np.float32)
    subsample_idx = SUBSAMPLE_INDICES.to(device)

    # Prefetch next clip in background
    prefetcher = ThreadPoolExecutor(max_workers=1)
    next_clip_future = None

    for clip_idx, (clip_start, clip_path) in enumerate(clips):
        is_last = (clip_idx == len(clips) - 1)

        # Get frames
        if next_clip_future is not None:
            clip_np = next_clip_future.result()
        else:
            clip_np = read_clip_frames(clip_path)

        # Prefetch next
        if not is_last:
            _, next_path = clips[clip_idx + 1]
            next_clip_future = prefetcher.submit(read_clip_frames, next_path)
        else:
            next_clip_future = None

        if clip_np is None:
            continue

        n_frames = clip_np.shape[0]

        # How many feature positions from this clip
        if is_last:
            n_positions = min(n_frames - CLIP_LENGTH + 1,
                              total_frames - clip_start - CLIP_LENGTH + 1)
        else:
            n_positions = min(CLIP_STRIDE, n_frames - CLIP_LENGTH + 1,
                              total_frames - clip_start - CLIP_LENGTH + 1)

        if n_positions <= 0:
            continue

        # GPU batch preprocess: (3, N, 224, 224)
        clip_gpu = preprocess_clip_gpu(clip_np, device)
        del clip_np

        # Extract features by sliding window with batched GPU inference
        batch_windows = []
        batch_positions = []

        for t in range(n_positions):
            abs_pos = clip_start + t
            if abs_pos >= total_frames:
                break

            # Select 16 frames from 48-frame window
            frame_indices = subsample_idx + t
            window_16 = clip_gpu[:, frame_indices, :, :]  # (3, 16, 224, 224)
            batch_windows.append(window_16)
            batch_positions.append(abs_pos)

            if len(batch_windows) >= batch_size:
                batch_tensor = torch.stack(batch_windows)  # (B, 3, 16, H, W)
                with torch.no_grad():
                    feats = forward_fn(model, batch_tensor)
                feats_np = feats.cpu().numpy()
                for i, pos in enumerate(batch_positions):
                    features[pos] = feats_np[i]
                batch_windows.clear()
                batch_positions.clear()

        # Flush remaining
        if batch_windows:
            batch_tensor = torch.stack(batch_windows)
            with torch.no_grad():
                feats = forward_fn(model, batch_tensor)
            feats_np = feats.cpu().numpy()
            for i, pos in enumerate(batch_positions):
                features[pos] = feats_np[i]

        del clip_gpu
        torch.cuda.empty_cache()

    prefetcher.shutdown(wait=True)
    return features


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract frozen VideoMAE features from bbox clips')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to remapped 480p pkl file')
    parser.add_argument('--clips_root', type=str, required=True,
                        help='Root dir with {video_stem}/clip_*.mp4 structure')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npy feature files')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='GPU batch size for backbone inference')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model: VideoMAE (MCG-NJU/videomae-base-finetuned-kinetics)")
    print(f"  feat_dim={FEAT_DIM}, spatial={SIDE_SIZE}x{CROP_SIZE}")
    print(f"  Temporal: {CLIP_LENGTH}-frame window -> subsample {MODEL_FRAMES}")
    print(f"  Subsample indices: {SUBSAMPLE_INDICES.tolist()}")
    print(f"  Batch size: {args.batch_size}")

    # Load model
    print("Loading model...")
    model, forward_fn = load_videomae(args.device)
    print("Model loaded.")

    # Load pkl
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    video_map = {}
    for ann in data['annotations']:
        name = ann['frame_dir']
        if name not in video_map:
            video_map[name] = ann['total_frames']
    del data
    gc.collect()

    video_list = sorted(video_map.items())
    print(f"Found {len(video_list)} unique videos")

    total_all_frames = sum(v[1] for v in video_list)
    storage_gb = total_all_frames * FEAT_DIM * 4 / (1024**3)
    print(f"Total frames: {total_all_frames:,}")
    print(f"Estimated storage: {storage_gb:.1f} GB")

    # Process videos
    skipped = 0
    processed = 0
    pbar = tqdm(video_list, desc="Videos", unit="vid")

    for video_name, total_frames in pbar:
        output_path = os.path.join(args.output_dir, f"{video_name}.npy")

        if os.path.exists(output_path):
            existing = np.load(output_path, mmap_mode='r')
            if existing.shape == (total_frames, FEAT_DIM):
                skipped += 1
                pbar.set_postfix_str(f"skip={skipped}")
                continue

        video_stem = Path(video_name).stem
        clips = get_clip_files(args.clips_root, video_stem)

        if len(clips) == 0:
            tqdm.write(f"  WARNING: No clips for {video_stem}")
            continue

        pbar.set_postfix_str(
            f"{video_stem} ({total_frames} fr, {len(clips)} clips)")

        features = process_video(
            clips, total_frames, model, forward_fn,
            args.device, batch_size=args.batch_size,
        )

        valid = int((np.abs(features).sum(axis=1) > 0).sum())
        expected_valid = max(0, total_frames - CLIP_LENGTH + 1)

        np.save(output_path, features)
        processed += 1

        pbar.set_postfix_str(
            f"done: {video_stem} ({valid}/{expected_valid} valid)")

        del features
        gc.collect()

    print(f"\nDone! Processed {processed}, skipped {skipped} "
          f"-> {args.output_dir}/")


if __name__ == '__main__':
    main()

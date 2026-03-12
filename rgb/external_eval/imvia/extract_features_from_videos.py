"""
Extract frozen DINOv3 / CLIP / VideoMAE features directly from video files,
applying person-centered bbox cropping on-the-fly.

This avoids the intermediate step of saving bbox-cropped clip files to disk.
The cropping logic matches rgb/feature_extraction/extract_bbox_clips.py exactly:
  1. Upscale frame to 1920x1080 (to match PKL bbox coordinates)
  2. Interpolate bboxes to fill gaps
  3. Compute person-centered crop region (10% padding, 4:3 aspect, min 640x480)
  4. Crop and resize to 640x480
  5. Feed through frozen backbone

Output: one .npy per video, shape (total_frames, feat_dim), dtype float32.
  - Filename = ann['frame_dir'] + '.npy'
  - Syncs with pkl annotation fields by frame index

Usage:
    python extract_features_from_videos.py \\
        --pkl_path imvia_dataset.pkl \\
        --video_root /path/to/ImViA \\
        --model dinov3 \\
        --output_dir ./features/imvia_dinov3
"""

import argparse
import gc
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


# ── Crop constants (matching rgb/feature_extraction/extract_bbox_clips.py) ──
OUT_W, OUT_H = 640, 480
TARGET_ASPECT = OUT_W / OUT_H  # 4:3
FRAME_W, FRAME_H = 1920, 1080
PADDING_RATIO = 0.1

# ── VideoMAE constants (matching rgb/feature_extraction/extract_video_features.py)
CLIP_LENGTH = 48
MODEL_FRAMES = 16
SUBSAMPLE_INDICES = torch.linspace(0, CLIP_LENGTH - 1, MODEL_FRAMES).long()

VIDEOMAE_CFG = {
    'feat_dim': 768,
    'side_size': 224,
    'crop_size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

IMVIA_ROOMS = ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02']
IMVIA_ROOMS.sort(key=len, reverse=True)

# ── Bbox interpolation (from rgb/feature_extraction/extract_bbox_clips.py) ──

def interpolate_bboxes(bboxes_raw):
    """Fill missing bboxes via interpolation and forward/backward fill."""
    total = len(bboxes_raw)
    valid_mask = np.abs(bboxes_raw[:, 2]) > 1
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.tile(
            [FRAME_W / 2, FRAME_H / 2, 100, 200],
            (total, 1)).astype(np.float32)

    bboxes = bboxes_raw.copy()
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    bboxes[:first_valid] = bboxes[first_valid]
    bboxes[last_valid + 1:] = bboxes[last_valid]

    still_invalid = np.where(
        (~valid_mask)
        & (np.arange(total) >= first_valid)
        & (np.arange(total) <= last_valid)
    )[0]
    if len(still_invalid) > 0:
        valid_x = np.where(valid_mask)[0]
        for col in range(4):
            valid_y = bboxes_raw[valid_mask, col]
            bboxes[still_invalid, col] = np.interp(
                still_invalid, valid_x, valid_y)

    return bboxes


def get_crop_region(bx, by, bw, bh):
    """Compute crop region centered on bbox with padding, 4:3 aspect, shift-to-fit."""
    padded_w = bw * (1 + 2 * PADDING_RATIO)
    padded_h = bh * (1 + 2 * PADDING_RATIO)

    crop_h = max(OUT_H, padded_h)
    crop_w = crop_h * TARGET_ASPECT

    if padded_w > crop_w:
        crop_w = padded_w
        crop_h = crop_w / TARGET_ASPECT

    if crop_w > FRAME_W:
        crop_w = FRAME_W
        crop_h = crop_w / TARGET_ASPECT
    if crop_h > FRAME_H:
        crop_h = FRAME_H
        crop_w = crop_h * TARGET_ASPECT

    cx = bx + bw / 2
    cy = by + bh / 2

    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = int(cx + crop_w / 2)
    y2 = int(cy + crop_h / 2)

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > FRAME_W:
        x1 -= (x2 - FRAME_W)
        x2 = FRAME_W
    if y2 > FRAME_H:
        y1 -= (y2 - FRAME_H)
        y2 = FRAME_H

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(FRAME_W, x2)
    y2 = min(FRAME_H, y2)

    return x1, y1, x2, y2


# ── Video path resolution ────────────────────────────────────────────────

def resolve_video_path(frame_dir, video_root):
    """Map PKL frame_dir to original video file path.

    ImViA frame_dir format: '{room}_{video_stem}'
    e.g. 'Coffee_room_01_video (1)' -> '{video_root}/Coffee_room_01/Videos/video (1).avi'
    """
    for room in IMVIA_ROOMS:
        prefix = room + '_'
        if frame_dir.startswith(prefix):
            video_stem = frame_dir[len(prefix):]
            # Try common extensions
            for ext in ('.avi', '.mp4', '.mkv'):
                path = os.path.join(video_root, room, 'Videos', video_stem + ext)
                if os.path.exists(path):
                    return path
            break

    # Fallback: try as direct filename under video_root
    for ext in ('.avi', '.mp4', '.mkv', ''):
        path = os.path.join(video_root, frame_dir + ext)
        if os.path.exists(path):
            return path

    return None


# ── Read and crop video frames ───────────────────────────────────────────

def read_and_crop_frames(video_path, bboxes_interp, total_frames):
    """Read video, upscale to 1080p, apply bbox crop, resize to 640x480.

    Returns: list of (H, W, 3) uint8 numpy arrays (640x480 BGR crops)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    crops = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            # Pad remaining with black if video is shorter than expected
            crops.append(np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8))
            continue

        # Upscale to 1080p to match PKL bbox coordinates
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # Apply person-centered crop
        if i < len(bboxes_interp):
            x1, y1, x2, y2 = get_crop_region(*bboxes_interp[i])
        else:
            x1, y1, x2, y2 = get_crop_region(*bboxes_interp[-1])

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (OUT_W, OUT_H))
        crops.append(crop)

    cap.release()
    return crops


# ── Model loaders ────────────────────────────────────────────────────────

def load_dinov3(device='cuda'):
    from transformers import AutoImageProcessor, AutoModel
    model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def preprocess(pil_img):
        inputs = processor(images=pil_img, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)

    return model, preprocess, 768


def load_clip(device='cuda'):
    import clip
    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval()
    return model, preprocess, 512


def load_videomae(device='cuda'):
    from transformers import VideoMAEModel
    model = VideoMAEModel.from_pretrained(
        'MCG-NJU/videomae-base-finetuned-kinetics')
    model = model.to(device).eval().half()
    for p in model.parameters():
        p.requires_grad = False
    return model, 768


# ── Feature extraction: per-frame (DINOv3 / CLIP) ───────────────────────

@torch.no_grad()
def extract_perframe_features(crops, model, preprocess, feat_dim,
                              model_type, device, batch_size=16):
    """Extract per-frame features from 640x480 BGR crops."""
    total = len(crops)
    features = np.zeros((total, feat_dim), dtype=np.float32)

    batch_tensors = []
    batch_indices = []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        if model_type == 'dinov3':
            out = model(pixel_values=x)
            feat = out.pooler_output
        elif model_type == 'clip':
            feat = model.encode_image(x).float()
        feats_np = feat.cpu().numpy()
        for i, idx in enumerate(batch_indices):
            features[idx] = feats_np[i]
        batch_tensors.clear()
        batch_indices.clear()

    for i, crop_bgr in enumerate(crops):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        batch_tensors.append(preprocess(pil_img))
        batch_indices.append(i)

        if len(batch_tensors) >= batch_size:
            flush()

    flush()
    return features


# ── Feature extraction: sliding window (VideoMAE) ────────────────────────

@torch.no_grad()
def extract_videomae_features(crops, model, feat_dim, device, batch_size=16):
    """Extract VideoMAE features using 48-frame sliding windows."""
    total = len(crops)
    features = np.zeros((total, feat_dim), dtype=np.float32)
    subsample_idx = SUBSAMPLE_INDICES.to(device)

    cfg = VIDEOMAE_CFG
    crop_size = cfg['crop_size']
    mean = torch.tensor(cfg['mean']).view(3, 1, 1)
    std = torch.tensor(cfg['std']).view(3, 1, 1)

    # Preprocess on CPU in chunks to avoid GPU OOM on long videos
    PREPROC_CHUNK = 256
    preprocessed = torch.zeros(total, 3, crop_size, crop_size)

    # Compute resize dimensions (same for all frames since all crops are 640x480)
    h, w = OUT_H, OUT_W
    side_size = cfg['side_size']
    if h <= w:
        new_h = side_size
        new_w = int(round(w * side_size / h))
    else:
        new_w = side_size
        new_h = int(round(h * side_size / w))

    for start in range(0, total, PREPROC_CHUNK):
        end = min(start + PREPROC_CHUNK, total)
        chunk_rgb = []
        for i in range(start, end):
            chunk_rgb.append(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB))
        chunk_np = np.stack(chunk_rgb)
        del chunk_rgb

        chunk_t = torch.from_numpy(chunk_np).permute(0, 3, 1, 2).float()
        chunk_t.div_(255.0)
        del chunk_np

        chunk_t = TF.resize(chunk_t, [new_h, new_w], antialias=True)
        chunk_t = TF.center_crop(chunk_t, [crop_size, crop_size])
        chunk_t = TF.normalize(chunk_t, mean=cfg['mean'], std=cfg['std'])
        preprocessed[start:end] = chunk_t
        del chunk_t

    # (N, 3, crop, crop) -> (3, N, crop, crop) on CPU for temporal indexing
    preprocessed = preprocessed.permute(1, 0, 2, 3)

    # Sliding window extraction
    n_positions = max(0, total - CLIP_LENGTH + 1)

    batch_windows = []
    batch_positions = []

    for t in range(n_positions):
        frame_indices = SUBSAMPLE_INDICES + t
        window_16 = preprocessed[:, frame_indices, :, :]  # (3, 16, crop, crop)
        batch_windows.append(window_16)
        batch_positions.append(t)

        if len(batch_windows) >= batch_size:
            batch_tensor = torch.stack(batch_windows).to(device)  # (B, 3, 16, H, W)
            x = batch_tensor.half().permute(0, 2, 1, 3, 4)  # -> (B, 16, 3, H, W)
            out = model(x)
            feats = out.last_hidden_state.mean(dim=1).float()
            feats_np = feats.cpu().numpy()
            for i, pos in enumerate(batch_positions):
                features[pos] = feats_np[i]
            batch_windows.clear()
            batch_positions.clear()

    # Flush remaining
    if batch_windows:
        batch_tensor = torch.stack(batch_windows).to(device)
        x = batch_tensor.half().permute(0, 2, 1, 3, 4)
        out = model(x)
        feats = out.last_hidden_state.mean(dim=1).float()
        feats_np = feats.cpu().numpy()
        for i, pos in enumerate(batch_positions):
            features[pos] = feats_np[i]

    del preprocessed
    return features


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract features from videos with on-the-fly bbox cropping')
    parser.add_argument('--pkl_path', type=str, required=True)
    parser.add_argument('--video_root', type=str, required=True,
                        help='Root dir with video files (e.g., ImViA dataset root)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['dinov3', 'clip', 'videomae'])
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load PKL
    print(f'Loading {args.pkl_path}...')
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Build video info: {frame_dir: (total_frames, bboxes)}
    video_info = {}
    for ann in data['annotations']:
        name = ann['frame_dir']
        if name not in video_info:
            raw_bboxes = ann.get('bboxes', ann.get('bbox', None))
            video_info[name] = {
                'total_frames': ann['total_frames'],
                'bboxes': raw_bboxes,
            }
    del data
    gc.collect()

    print(f'Found {len(video_info)} unique videos')

    # Load model
    print(f'Loading {args.model} model...')
    if args.model == 'dinov3':
        model, preprocess, feat_dim = load_dinov3(args.device)
    elif args.model == 'clip':
        model, preprocess, feat_dim = load_clip(args.device)
    elif args.model == 'videomae':
        model, feat_dim = load_videomae(args.device)
        preprocess = None
    print(f'Model loaded. feat_dim={feat_dim}')

    # Process videos
    processed = 0
    skipped = 0
    failed = 0
    pbar = tqdm(sorted(video_info.items()), desc='Videos')

    for frame_dir, info in pbar:
        output_path = os.path.join(args.output_dir, f'{frame_dir}.npy')
        total_frames = info['total_frames']

        # Skip if already done
        if os.path.exists(output_path):
            existing = np.load(output_path, mmap_mode='r')
            if existing.shape == (total_frames, feat_dim):
                skipped += 1
                continue

        # Resolve video path
        video_path = resolve_video_path(frame_dir, args.video_root)
        if video_path is None:
            tqdm.write(f'  WARNING: No video found for {frame_dir}')
            failed += 1
            continue

        # Interpolate bboxes
        raw_bboxes = info['bboxes']
        if raw_bboxes is None:
            tqdm.write(f'  WARNING: No bboxes for {frame_dir}, using center crop')
            bboxes_interp = np.tile(
                [FRAME_W / 4, FRAME_H / 4, FRAME_W / 2, FRAME_H / 2],
                (total_frames, 1)).astype(np.float32)
        else:
            bboxes_interp = interpolate_bboxes(raw_bboxes)

        pbar.set_postfix_str(f'{frame_dir} ({total_frames} fr)')

        # Read and crop frames
        crops = read_and_crop_frames(video_path, bboxes_interp, total_frames)
        if crops is None:
            tqdm.write(f'  WARNING: Could not open {video_path}')
            failed += 1
            continue

        # Extract features
        if args.model in ('dinov3', 'clip'):
            features = extract_perframe_features(
                crops, model, preprocess, feat_dim,
                args.model, args.device, args.batch_size)
        else:
            features = extract_videomae_features(
                crops, model, feat_dim, args.device, args.batch_size)

        # Save
        np.save(output_path, features)
        processed += 1

        valid = int((np.abs(features).sum(axis=1) > 0).sum())
        pbar.set_postfix_str(f'done: {frame_dir} ({valid}/{total_frames} valid)')

        del crops, features
        gc.collect()

    print(f'\nDone! Processed {processed}, skipped {skipped}, '
          f'failed {failed} -> {args.output_dir}/')


if __name__ == '__main__':
    main()

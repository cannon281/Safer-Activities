"""Extract person-centered 480p bbox-cropped video clips from raw videos.

This is done to prevent data loading bottlenecks during training.
The overlaps exist to allow sampling 48-144 frames (same way as with keypoint training)

Given raw 1080p videos and a PKL annotation file containing per-frame 
bounding boxes, this script produces overlapping 640x480 person-centered clips
and a PKL file with 480p keypoints added alongside the original 1080p coordinates.

Cropping process:
1. Bbox interpolation (once per video, for cropping only):
   - Bboxes in [x1, y1, w, h] corner-based format from pkl (1080p coords)
   - Leading gap: copy first valid bbox backward
   - Trailing gap: copy last valid bbox forward
   - Interior gaps: linear interpolation between nearest valid neighbors

2. Person-centered crop (per frame):
   - Add 10% padding around bbox on each side
   - crop_h = max(480, padded_bbox_height)
   - crop_w = crop_h * 4/3 (maintain 4:3 aspect ratio, min 640x480)
   - If padded bbox wider than crop_w, expand width and recalculate height
   - Center crop on bbox center
   - If crop goes outside frame, shift back in (to avoid black padding beyond frame area)
   - Resize to exactly 640x480

3. Pkl remapping (added alongside the original coordinates):
   - Keypoints and bboxes transformed to 640x480 crop coordinates
   - Only originally valid data is remapped; frames that had zero/invalid
     bboxes or keypoints in the original pkl stay zero in the new pkl

Naming convention:
    {clips_root}/{video_stem}/clip_{start_frame:06d}.mp4

Usage:
    python extract_bbox_clips.py \
        --video-root /path/to/Videos \
        --clips-root /path/to/output/BBoxClips \
        --pkl-file /path/to/dataset.pkl \
        --out-pkl /path/to/dataset_480p.pkl \
        --chunk-size 300 --stride 156 --workers 8
"""
import argparse
import os
import pickle
import subprocess
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

# Output resolution
OUT_W, OUT_H = 640, 480
TARGET_ASPECT = OUT_W / OUT_H  # 4:3
# Frame resolution (keypoint coordinate space)
FRAME_W, FRAME_H = 1920, 1080
PADDING_RATIO = 0.1  # 10% padding on each side


def interpolate_bboxes(bboxes_raw):
    """Fill missing bboxes via interpolation and forward/backward fill.
    Returns interpolated bboxes (for cropping) — does NOT modify the original."""
    total = len(bboxes_raw)
    valid_mask = np.abs(bboxes_raw[:, 2]) > 1  # width > 1
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        bboxes = np.tile([FRAME_W / 2, FRAME_H / 2, 100, 200], (total, 1)).astype(np.float32)
        return bboxes

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
            bboxes[still_invalid, col] = np.interp(still_invalid, valid_x, valid_y)

    return bboxes


def get_crop_region(bx, by, bw, bh):
    """Compute crop region centered on bbox with padding, aspect ratio, and shift-to-fit.

    Args:
        bx, by: top-left corner of bbox (corner-based xywh)
        bw, bh: width and height of bbox

    Returns:
        x1, y1, x2, y2: crop region clamped to frame bounds (no black padding)
    """
    padded_w = bw * (1 + 2 * PADDING_RATIO)
    padded_h = bh * (1 + 2 * PADDING_RATIO)

    crop_h = max(OUT_H, padded_h)
    crop_w = crop_h * TARGET_ASPECT

    if padded_w > crop_w:
        crop_w = padded_w
        crop_h = crop_w / TARGET_ASPECT

    # Cap crop to frame dimensions while maintaining 4:3 aspect ratio
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

    # Shift back into frame (no black padding)
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


def compute_all_crop_regions(bboxes_interp):
    """Compute crop regions for all frames. Returns (N, 4) array of [x1, y1, x2, y2]."""
    regions = np.zeros((len(bboxes_interp), 4), dtype=np.int32)
    for i in range(len(bboxes_interp)):
        x1, y1, x2, y2 = get_crop_region(*bboxes_interp[i])
        regions[i] = [x1, y1, x2, y2]
    return regions


def remap_annotation(ann, crop_regions):
    """Add 480p bbox-crop coordinates alongside the original 1080p keypoints.

    Original 1080p keypoints are preserved as 'keypoint'. The remapped 480p
    coordinates are stored as 'keypoint_480p'. Bboxes are remapped in place.
    Only originally valid data is remapped; frames that had zero/invalid
    bboxes or keypoints in the original pkl stay zero in the output.
    """
    new_ann = {}

    # Copy all fields as-is first
    for key in ann:
        if key not in ('keypoint', 'keypoint_score', 'keypoint_3d', 'keypoint_3d_score',
                        'bboxes', 'bbox', 'img_shape', 'width', 'height'):
            new_ann[key] = ann[key]

    total_frames = ann['total_frames']

    # Remap 2D keypoints: keep original 1080p as 'keypoint', add 480p as 'keypoint_480p'
    if 'keypoint' in ann:
        old_kp = ann['keypoint'].copy()  # (P, N, 17, 2) — original 1080p
        new_kp = np.zeros_like(old_kp)
        for i in range(total_frames):
            x1, y1, x2, y2 = crop_regions[i]
            crop_w = x2 - x1
            crop_h = y2 - y1
            scale_x = OUT_W / max(crop_w, 1)
            scale_y = OUT_H / max(crop_h, 1)

            # Only remap frames where keypoints are non-zero
            frame_kp = old_kp[:, i, :, :]  # (P, 17, 2)
            is_valid = np.any(frame_kp != 0, axis=-1)  # (P, 17)

            new_frame_kp = np.zeros_like(frame_kp)
            new_frame_kp[..., 0] = (frame_kp[..., 0] - x1) * scale_x
            new_frame_kp[..., 1] = (frame_kp[..., 1] - y1) * scale_y

            # Zero out keypoints that were originally zero
            new_frame_kp[~is_valid] = 0
            new_kp[:, i, :, :] = new_frame_kp

        new_ann['keypoint'] = old_kp       # preserve original 1080p coordinates
        new_ann['keypoint_480p'] = new_kp   # 480p bbox-crop coordinates
    new_ann['img_shape'] = (FRAME_H, FRAME_W)
    new_ann['width'] = FRAME_W
    new_ann['height'] = FRAME_H

    # Copy keypoint_score unchanged (confidence scores don't depend on coords)
    if 'keypoint_score' in ann:
        new_ann['keypoint_score'] = ann['keypoint_score'].copy()

    # Remap 3D keypoints: (num_people, num_frames, 17, 3)
    if 'keypoint_3d' in ann:
        # 3D keypoints are in normalized space, not pixel space — copy as-is
        new_ann['keypoint_3d'] = ann['keypoint_3d'].copy()

    if 'keypoint_3d_score' in ann:
        new_ann['keypoint_3d_score'] = ann['keypoint_3d_score'].copy()

    # Remap bboxes: (num_frames, 4) in xywh format
    raw_bboxes = ann.get('bboxes', ann.get('bbox', None))
    bbox_key = 'bboxes' if 'bboxes' in ann else 'bbox'
    if raw_bboxes is not None:
        old_bb = raw_bboxes.copy()
        new_bb = np.zeros_like(old_bb)
        for i in range(total_frames):
            bx, by, bw, bh = old_bb[i]
            # Only remap originally valid bboxes
            if abs(bw) < 1 or abs(bh) < 1:
                continue  # stays zero

            x1, y1, x2, y2 = crop_regions[i]
            crop_w = x2 - x1
            crop_h = y2 - y1
            scale_x = OUT_W / max(crop_w, 1)
            scale_y = OUT_H / max(crop_h, 1)

            new_bb[i, 0] = (bx - x1) * scale_x
            new_bb[i, 1] = (by - y1) * scale_y
            new_bb[i, 2] = bw * scale_x
            new_bb[i, 3] = bh * scale_y

        new_ann[bbox_key] = new_bb

    return new_ann


def extract_bbox_clips_for_video(args):
    """Extract bbox-cropped sub-clips for one video."""
    video_path, clips_dir, chunk_size, stride, bboxes_interp = args

    video_name = Path(video_path).stem
    out_dir = os.path.join(clips_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    clips_written = 0
    for clip_start in range(0, total, stride):
        clip_end = min(clip_start + chunk_size, total)
        n_frames = clip_end - clip_start
        clip_path = os.path.join(out_dir, f'clip_{clip_start:06d}.mp4')

        if os.path.exists(clip_path):
            clips_written += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{OUT_W}x{OUT_H}', '-pix_fmt', 'bgr24',
            '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '28', '-pix_fmt', 'yuv420p', '-an',
            clip_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frames_written = 0
        try:
            for j in range(n_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                idx = clip_start + j
                if idx < len(bboxes_interp):
                    x1, y1, x2, y2 = get_crop_region(*bboxes_interp[idx])
                else:
                    x1, y1, x2, y2 = get_crop_region(*bboxes_interp[-1])
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (OUT_W, OUT_H))
                proc.stdin.write(crop.tobytes())
                frames_written += 1
        except BrokenPipeError:
            pass

        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        proc.wait()

        if proc.returncode != 0 or frames_written == 0:
            if os.path.exists(clip_path):
                os.remove(clip_path)
            continue

        clips_written += 1

    cap.release()
    return video_name, total, clips_written


def main():
    parser = argparse.ArgumentParser(description='Extract bbox-cropped 480p sub-clips')
    parser.add_argument('--video-root', required=True,
                        help='Directory containing raw .mp4 videos')
    parser.add_argument('--clips-root', required=True,
                        help='Output directory for cropped clips')
    parser.add_argument('--pkl-file', required=True,
                        help='Path to the annotation pkl file with bboxes')
    parser.add_argument('--out-pkl', required=True,
                        help='Output path for remapped pkl file')
    parser.add_argument('--chunk-size', type=int, default=300,
                        help='Frames per clip (default: 300)')
    parser.add_argument('--stride', type=int, default=156,
                        help='Stride between clip start frames (default: 156)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    overlap = args.chunk_size - args.stride
    print(f'Chunk size: {args.chunk_size}, stride: {args.stride}, overlap: {overlap}')
    print(f'Output resolution: {OUT_W}x{OUT_H}')
    print(f'Bbox padding: {PADDING_RATIO*100:.0f}% on each side')

    # Load pkl
    print(f'Loading {args.pkl_file}...')
    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)

    # ---- Step 1: Generate remapped pkl ----
    print('Remapping pkl annotations to 480p crop coordinates...')
    new_annotations = []
    bbox_lookup = {}  # video_name -> interpolated bboxes (for clip extraction)

    for ann in tqdm(data['annotations'], desc='Remapping pkl'):
        name = ann['frame_dir']
        raw_bboxes = ann.get('bboxes', ann.get('bbox', None))

        if raw_bboxes is None:
            print(f'WARNING: No bboxes for {name}, copying as-is')
            new_annotations.append(ann)
            continue

        # Interpolate bboxes (for cropping only)
        bboxes_interp = interpolate_bboxes(raw_bboxes)
        bbox_lookup[name] = bboxes_interp

        # Compute crop regions for all frames
        crop_regions = compute_all_crop_regions(bboxes_interp)

        # Remap annotation (uses original raw data, not interpolated)
        new_ann = remap_annotation(ann, crop_regions)
        new_annotations.append(new_ann)

    # Save remapped pkl
    new_data = {
        'split': data.get('split', {}),
        'annotations': new_annotations,
    }
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(new_data, f)
    print(f'Saved remapped pkl to {args.out_pkl}')

    # ---- Step 2: Extract bbox-cropped clips ----
    videos = sorted(Path(args.video_root).glob('*.mp4'))
    print(f'\nFound {len(videos)} videos')
    os.makedirs(args.clips_root, exist_ok=True)

    # Build tasks
    tasks = []
    skipped = 0
    for v in videos:
        vname = v.name
        if vname in bbox_lookup:
            bboxes = bbox_lookup[vname]
        else:
            matched = [k for k in bbox_lookup if os.path.basename(k) == vname]
            if matched:
                bboxes = bbox_lookup[matched[0]]
            else:
                print(f'WARNING: No bbox data for {vname}, skipping')
                skipped += 1
                continue
        tasks.append((str(v), args.clips_root, args.chunk_size, args.stride, bboxes))

    if skipped:
        print(f'Skipped {skipped} videos without bbox data')

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, (name, total, n_clips) in enumerate(
                    tqdm(pool.imap(extract_bbox_clips_for_video, tasks),
                         total=len(tasks), desc='Extracting clips')):
                print(f'[{i + 1}/{len(tasks)}] {name}: {total} frames -> {n_clips} clips')
    else:
        for i, task in enumerate(tqdm(tasks, desc='Extracting clips')):
            name, total, n_clips = extract_bbox_clips_for_video(task)
            print(f'[{i + 1}/{len(tasks)}] {name}: {total} frames -> {n_clips} clips')

    print('Done!')


if __name__ == '__main__':
    main()

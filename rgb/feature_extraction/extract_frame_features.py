"""
Extract frozen CLIP / DINOv3 per-frame features from pre-cropped bbox clips.

Clips are already person-centered 640x480 crops (from extract_bbox_clips.py).
This script streams each frame through the model, extracting the CLS token
feature vector per frame.

Uses multiprocessing: each worker is a separate process with its own model
instance on the GPU. Videos are split evenly across workers.

Output: one .npy per video (not the clips, but the original untrimmed video):
     shape (total_frames, feat_dim), dtype float32.
  - Filename = ann['frame_dir'] + '.npy'  (e.g. "video_01.mp4.npy")
  - features[i] = feature vector from frame i of the original video
  - Syncs directly with pkl annotation fields:
      labels[i], keypoint[:, i, :, :], bboxes[i], etc.

Models:
  - CLIP ViT-B/16:  512-dim CLS token per frame
  - DINOv3 ViT-B/16: 768-dim CLS token per frame

Usage:
    # CLIP features
    python extract_frame_features.py \
        --pkl_path /path/to/dataset_480p.pkl \
        --clips_root /path/to/BBoxClips/normal \
        --model clip \
        --output_dir ./features/normal_clip \
        --stride 156 --workers 2

    # DINOv3 features
    python extract_frame_features.py \
        --pkl_path /path/to/dataset_480p.pkl \
        --clips_root /path/to/BBoxClips/normal \
        --model dinov3 \
        --output_dir ./features/normal_dinov3 \
        --stride 156 --workers 2
"""

import argparse
import gc
import os
import pickle
import re
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ── Model loaders ─────────────────────────────────────────────────────────

def load_dinov3(model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', device='cuda'):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    feat_dim = {
        'facebook/dinov3-vits16-pretrain-lvd1689m': 384,
        'facebook/dinov3-vitb16-pretrain-lvd1689m': 768,
        'facebook/dinov3-vitl16-pretrain-lvd1689m': 1024,
    }.get(model_name, 768)

    def preprocess(pil_img):
        inputs = processor(images=pil_img, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
    return model, preprocess, feat_dim


def load_clip(model_name='ViT-B/16', device='cuda'):
    import clip
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    feat_dim = {
        'ViT-B/16': 512, 'ViT-B/32': 512,
        'ViT-L/14': 768, 'ViT-L/14@336px': 768,
    }[model_name]
    return model, preprocess, feat_dim


def load_model(model_type, model_name, device):
    if model_type == 'dinov3':
        return load_dinov3(model_name, device)
    elif model_type == 'clip':
        return load_clip(model_name, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── GPU inference (mini-batch) ────────────────────────────────────────────

@torch.no_grad()
def extract_features_batch(model, tensors, model_type, device):
    """Extract features for a batch of frames. Returns (N, D) numpy array."""
    x = torch.stack(tensors).to(device)
    if model_type == 'dinov3':
        outputs = model(pixel_values=x)
        feat = outputs.pooler_output
    elif model_type == 'clip':
        feat = model.encode_image(x).float()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return feat.cpu().numpy()


# ── Clip discovery ────────────────────────────────────────────────────────

def get_clip_files(clips_dir, video_stem):
    clip_dir = os.path.join(clips_dir, video_stem)
    if not os.path.isdir(clip_dir):
        return []
    clips = []
    for f in os.listdir(clip_dir):
        m = re.match(r'clip_(\d+)\.mp4', f)
        if m:
            clips.append((int(m.group(1)), os.path.join(clip_dir, f)))
    clips.sort(key=lambda x: x[0])
    return clips


# ── Per-video processing ─────────────────────────────────────────────────

def process_video_from_clips(clips, total_frames, model, preprocess,
                             feat_dim, model_type, stride, device,
                             batch_size=16, worker_id=0, result_queue=None):
    """Stream frames, flush through GPU in mini-batches."""
    features = np.zeros((total_frames, feat_dim), dtype=np.float32)
    frames_done = 0

    # Mini-batch buffer
    batch_tensors = []
    batch_indices = []

    def flush():
        nonlocal frames_done
        if not batch_tensors:
            return
        feats = extract_features_batch(model, batch_tensors, model_type, device)
        for i, idx in enumerate(batch_indices):
            features[idx] = feats[i]
        frames_done += len(batch_tensors)
        batch_tensors.clear()
        batch_indices.clear()
        if result_queue is not None and frames_done % 100 < batch_size:
            result_queue.put(('frame_progress', worker_id, '',
                              frames_done, total_frames))

    for clip_idx, (start_frame, clip_path) in enumerate(clips):
        is_last_clip = (clip_idx == len(clips) - 1)

        cap = cv2.VideoCapture(clip_path)
        clip_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Since there is overlap, only need to process stride frames per clip
        frames_to_process = clip_frame_count if is_last_clip else min(stride, clip_frame_count)

        for local_idx in range(clip_frame_count):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if local_idx >= frames_to_process:
                break

            orig_frame_idx = start_frame + local_idx
            if orig_frame_idx >= total_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            batch_tensors.append(preprocess(pil_img))
            batch_indices.append(orig_frame_idx)

            del frame_bgr, frame_rgb, pil_img

            if len(batch_tensors) >= batch_size:
                flush()

        cap.release()

    flush()

    return features


# ── Worker process ────────────────────────────────────────────────────────

def worker_fn(worker_id, video_list, clips_root, output_dir, model_type,
              model_name, feat_dim, stride, batch_size, device, result_queue):
    """Each worker loads its own model and processes its share of videos."""
    print(f"[Worker {worker_id}] Loading model {model_type} ({model_name}) on {device}...")
    model, preprocess, _ = load_model(model_type, model_name, device)
    print(f"[Worker {worker_id}] Model loaded. Processing {len(video_list)} videos.")

    processed = 0
    skipped = 0
    for video_name, total_frames in video_list:
        output_path = os.path.join(output_dir, f"{video_name}.npy")
        if os.path.exists(output_path):
            skipped += 1
            result_queue.put(('skip', worker_id, video_name, 0, 0))
            continue

        video_stem = Path(video_name).stem
        clips = get_clip_files(clips_root, video_stem)

        if len(clips) == 0:
            result_queue.put(('no_clips', worker_id, video_name, 0, 0))
            continue

        result_queue.put(('video_start', worker_id, video_name,
                          total_frames, len(clips)))

        features = process_video_from_clips(
            clips, total_frames,
            model, preprocess, feat_dim, model_type,
            stride, device, batch_size=batch_size,
            worker_id=worker_id, result_queue=result_queue,
        )

        valid = int((np.abs(features).sum(axis=1) > 0).sum())
        np.save(output_path, features)
        del features
        gc.collect()

        processed += 1
        result_queue.put(('done', worker_id, video_name, total_frames, valid))

    result_queue.put(('finished', worker_id, '', processed, skipped))


def main():
    parser = argparse.ArgumentParser(
        description='Extract CLIP/DINOv3 per-frame features from bbox clips')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to remapped 480p pkl file')
    parser.add_argument('--clips_root', type=str, required=True,
                        help='Root dir with {video_stem}/clip_*.mp4 structure '
                             '(640x480 person-cropped clips from extract_bbox_clips.py)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['dinov3', 'clip'],
                        help='Feature extraction model')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Override default model variant')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npy feature files')
    parser.add_argument('--stride', type=int, default=156,
                        help='Clip stride (must match extract_bbox_clips.py)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of worker processes (each loads its own model)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size for GPU inference')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model name
    if args.model == 'dinov3':
        model_name = args.model_name or 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    elif args.model == 'clip':
        model_name = args.model_name or 'ViT-B/16'

    if args.model == 'dinov3':
        feat_dim = {'facebook/dinov3-vits16-pretrain-lvd1689m': 384,
                    'facebook/dinov3-vitb16-pretrain-lvd1689m': 768,
                    'facebook/dinov3-vitl16-pretrain-lvd1689m': 1024}.get(model_name, 768)
    elif args.model == 'clip':
        feat_dim = {'ViT-B/16': 512, 'ViT-B/32': 512,
                    'ViT-L/14': 768, 'ViT-L/14@336px': 768}[model_name]

    print(f"Model: {args.model} ({model_name}), feat_dim={feat_dim}")
    print(f"Workers: {args.workers}, batch_size: {args.batch_size}")

    # Load pkl — only keep total_frames per video
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    video_map = {}
    for ann in data['annotations']:
        frame_dir = ann['frame_dir']
        if frame_dir not in video_map:
            video_map[frame_dir] = ann['total_frames']
    del data
    gc.collect()

    video_list = list(video_map.items())
    print(f"Found {len(video_list)} unique videos")

    # Split videos across workers (round-robin for balanced load)
    worker_videos = [[] for _ in range(args.workers)]
    for i, item in enumerate(video_list):
        worker_videos[i % args.workers].append(item)

    for i, wv in enumerate(worker_videos):
        print(f"  Worker {i}: {len(wv)} videos")

    # Launch worker processes
    result_queue = Queue()
    processes = []
    for i in range(args.workers):
        p = Process(
            target=worker_fn,
            args=(i, worker_videos[i], args.clips_root, args.output_dir,
                  args.model, model_name, feat_dim,
                  args.stride, args.batch_size, args.device, result_queue),
        )
        p.start()
        processes.append(p)

    # Collect results from workers
    total_to_process = len(video_list)
    overall_bar = tqdm(total=total_to_process, desc="Videos", position=0,
                       unit="vid")
    worker_bars = {}
    for i in range(args.workers):
        worker_bars[i] = tqdm(total=0, desc=f"W{i}: idle", position=i + 1,
                              unit="fr", bar_format="{desc}: {n_fmt}/{total_fmt} frames [{rate_fmt}]")

    workers_done = 0
    total_processed = 0
    total_skipped = 0
    while workers_done < args.workers:
        msg = result_queue.get()
        status = msg[0]
        if status == 'video_start':
            _, wid, vname, total_frames, num_clips = msg
            wb = worker_bars[wid]
            wb.reset(total=total_frames)
            wb.set_description(f"W{wid}: {Path(vname).stem}")
        elif status == 'frame_progress':
            _, wid, _, frames_done, total_frames = msg
            wb = worker_bars[wid]
            wb.n = frames_done
            wb.refresh()
        elif status == 'done':
            _, wid, vname, total, valid = msg
            wb = worker_bars[wid]
            wb.n = total
            wb.refresh()
            overall_bar.update(1)
            overall_bar.set_postfix_str(
                f"last: {Path(vname).stem} ({valid}/{total} filled)")
        elif status == 'skip':
            overall_bar.update(1)
            total_skipped += 1
        elif status == 'no_clips':
            _, wid, vname, _, _ = msg
            overall_bar.update(1)
            overall_bar.write(
                f"  WARNING [W{wid}]: No clips found for {Path(vname).stem}")
        elif status == 'finished':
            _, wid, _, proc, skip = msg
            worker_bars[wid].set_description(f"W{wid}: done")
            worker_bars[wid].refresh()
            total_processed += proc
            total_skipped += skip
            workers_done += 1

    for wb in worker_bars.values():
        wb.close()
    overall_bar.close()

    for p in processes:
        p.join()

    print(f"\nDone! Processed {total_processed} videos, skipped {total_skipped} "
          f"(already done) -> {args.output_dir}/")


if __name__ == '__main__':
    main()

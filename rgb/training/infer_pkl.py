"""
Sliding-window inference for dino_clip_features models.

Same approach as inference/infer_pkl.py: per-window, stride-1 sliding across
each test annotation. Supports all dino_clip_features modes:
  - keypoint:  CNN1D on keypoints from pkl
  - feature:   MeanPool/Transformer on pre-extracted .npy features
  - fusion:    Features + keypoints -> fusion model

Output format matches inference/infer_pkl.py:
  result.pkl = {video_name: [[pred_label, gt_label, start_frame], ...], ...}
  Compatible with inference/calculate_accuracy.py for tolerance-based evaluation.

Usage:
  python infer_pkl.py \
      --run_dir runs/normal_dinov3_cnn1d_fusion_seq \
      --pkl_path ../pyskl/Pkl/aic_normal_dataset_with_3d_480p.pkl \
      --out_dict_dir ./eval_results/dinov3_cnn1d_fusion \
      --out_dict_name result.pkl
"""

import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from models import build_model
from transforms import ScaleWithNeckMotion, To1DInputShape

LABEL_NAMES = {
    'normal': [
        'no_label',
        'stand', 'stand_activity', 'walk', 'sit', 'sit_activity',
        'sitting_down', 'getting_up', 'bend', 'unstable', 'fall',
        'lie_down', 'lying_down', 'reach', 'run', 'jump',
    ],
    'wheelchair': [
        'no_label',
        'sit', 'propel', 'pick_place', 'sit_activity', 'bend',
        'getting_up', 'exercise', 'sitting_down', 'prepare_transfer',
        'transfer', 'fall', 'lie_down', 'lying_down', 'get_propelled',
        'stand',
    ],
}

WINDOW_SIZE = 48
CENTER_NUM_FRAMES = 5


def get_majority_labels(labels, num_frames=5, type="center"):
    """Same as inference/utils/dataset_utils.get_majority_labels."""
    if type == "center":
        start = len(labels) // 2 - num_frames // 2
        end = start + num_frames
        center_labels = labels[start:end]
    else:
        center_labels = labels[-num_frames:]
    label_counts = Counter(center_labels)
    return label_counts.most_common(1)[0][0]


def preprocess_keypoints(kp_window, image_width=1920, image_height=1080):
    """Preprocess a keypoint window: clip, scale, reshape.

    Mirrors cnn1d_infer from inference/utils/classification_model.py:
      1. Clip coordinates to valid range
      2. Apply ScaleWithNeckMotion (neck-relative + motion diff)
      3. Apply To1DInputShape -> (68, 48)

    Input:  (48, 17, 2) numpy array
    Output: (68, 48) tensor
    """
    kp = kp_window.copy()
    kp[..., 0] = np.clip(kp[..., 0], 0, image_width - 1)
    kp[..., 1] = np.clip(kp[..., 1], 0, image_height - 1)
    kp_tensor = torch.from_numpy(kp).float()
    kp_tensor = ScaleWithNeckMotion(image_width=image_width)(kp_tensor)
    kp_tensor = To1DInputShape()(kp_tensor)
    return kp_tensor


def run_inference(args):
    device = torch.device(args.device)

    # Load run config
    config_path = os.path.join(args.run_dir, 'config.json')
    with open(config_path) as f:
        cfg = json.load(f)

    mode = cfg['mode']
    model_type = cfg['model']
    feat_dim = cfg.get('feat_dim', 768)
    num_classes = cfg.get('num_classes', 15)
    model_frames = cfg.get('model_frames', 16)
    preprocess = cfg.get('preprocess', 'sequential')
    feature_dir = args.feature_dir or cfg.get('feature_dir')
    keypoint_field = cfg.get('keypoint_field', 'keypoint')
    dataset_type = cfg.get('dataset_type', 'normal')
    center_frame_only = cfg.get('center_frame_only', False)

    kp_num_frames = 48 if preprocess == 'sequential' else 144
    label_names = LABEL_NAMES[dataset_type]

    print(f"Run dir: {args.run_dir}")
    print(f"Mode: {mode}, Model: {model_type}, Feat dim: {feat_dim}")
    print(f"Dataset type: {dataset_type}")
    print(f"Feature dir: {feature_dir}")
    print(f"Keypoint field: {keypoint_field}")

    # Build and load model
    model = build_model(
        model_type, feat_dim=feat_dim, num_classes=num_classes,
        num_frames=model_frames, kp_num_frames=kp_num_frames)

    weight_path = os.path.join(args.run_dir, 'best.pth')
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load pkl data
    with open(args.pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    test_split = pkl_data['split']['sub_test']
    test_annotations = [
        ann for ann in pkl_data['annotations']
        if ann['frame_dir'] in set(test_split)
    ]

    # Handle annotation splitting for parallel inference
    if args.anno_split_num > 0:
        split_size = len(test_annotations) // (args.anno_split_num + 1)
        splits = [test_annotations[i * split_size:(i + 1) * split_size]
                  for i in range(args.anno_split_num)]
        splits.append(test_annotations[args.anno_split_num * split_size:])
        test_annotations = splits[args.anno_split_pos]

    # Determine image dimensions based on keypoint field
    if keypoint_field == 'keypoint_480p':
        image_height, image_width = 480, 640
    else:
        img_shape = test_annotations[0].get('img_shape') if test_annotations else None
        if img_shape:
            image_height, image_width = img_shape
        else:
            image_height, image_width = 1080, 1920
    print(f"Image dimensions: {image_width}x{image_height}")

    # Load feature cache if needed
    feature_cache = {}
    if mode in ('feature', 'fusion') and feature_dir:
        print(f"Loading features from: {feature_dir}")
        for ann in test_annotations:
            name = ann['frame_dir']
            if name not in feature_cache:
                feat_path = os.path.join(feature_dir, f"{name}.npy")
                if os.path.exists(feat_path):
                    feature_cache[name] = np.load(feat_path, mmap_mode='r')

    print(f"Test annotations: {len(test_annotations)}")
    print(f"Device: {device}")

    results = {}

    for ann in tqdm(test_annotations,
                    desc=f"Processing Annotations {args.anno_split_pos}/{args.anno_split_num}"):
        video_name = ann['frame_dir']
        results[video_name] = []

        ann_labels = ann['labels']

        # Get keypoints if needed
        has_keypoints = mode in ('keypoint', 'fusion')
        if has_keypoints:
            keypoints = ann[keypoint_field]  # (1, T, 17, 2)
            num_frames_total = keypoints.shape[1]
        else:
            # For feature-only modes, use label length as frame count
            num_frames_total = len(ann_labels)

        # Skip videos without features if we need them
        if mode in ('feature', 'fusion') and video_name not in feature_cache:
            continue

        features = feature_cache.get(video_name)

        num_windows = num_frames_total - WINDOW_SIZE + 1
        if num_windows <= 0:
            continue

        for start_frame in tqdm(range(num_windows),
                                desc=f"Sliding Windows {args.anno_split_pos}",
                                leave=False):
            # Ground truth label (majority vote from center frames)
            window_labels = ann_labels[start_frame:start_frame + WINDOW_SIZE]
            label_raw = get_majority_labels(
                window_labels, num_frames=CENTER_NUM_FRAMES, type=args.label_from)
            gt_label = label_names[label_raw]  # raw int -> string

            # Prepare inputs and run inference
            with torch.no_grad():
                if mode == 'keypoint':
                    wk = keypoints[0, start_frame:start_frame + WINDOW_SIZE, :, :]
                    kp_tensor = preprocess_keypoints(wk, image_width, image_height)
                    kp_tensor = kp_tensor.unsqueeze(0).to(device)
                    logits = model(kp_tensor.float())

                elif mode == 'feature':
                    if center_frame_only:
                        center_idx = start_frame + WINDOW_SIZE // 2
                        center_idx = min(center_idx, len(features) - 1)
                        clip_features = np.array(features[center_idx:center_idx + 1])
                    else:
                        # Subsample features (every 3rd frame for sequential mode)
                        sampled_indices = np.arange(start_frame, start_frame + WINDOW_SIZE, 3)
                        sampled_indices = sampled_indices[:model_frames]
                        sampled_indices = np.clip(sampled_indices, 0, len(features) - 1)
                        clip_features = np.array(features[sampled_indices])

                        if clip_features.shape[0] < model_frames:
                            pad = np.zeros(
                                (model_frames - clip_features.shape[0], feat_dim),
                                dtype=np.float32)
                            clip_features = np.concatenate([clip_features, pad], axis=0)

                    feat_tensor = torch.from_numpy(clip_features).float()
                    feat_tensor = feat_tensor.unsqueeze(0).to(device)
                    logits = model(feat_tensor)

                elif mode == 'fusion':
                    # Feature branch
                    if center_frame_only:
                        center_idx = start_frame + WINDOW_SIZE // 2
                        center_idx = min(center_idx, len(features) - 1)
                        clip_features = np.array(features[center_idx:center_idx + 1])
                    else:
                        sampled_indices = np.arange(start_frame, start_frame + WINDOW_SIZE, 3)
                        sampled_indices = sampled_indices[:model_frames]
                        sampled_indices = np.clip(sampled_indices, 0, len(features) - 1)
                        clip_features = np.array(features[sampled_indices])

                        if clip_features.shape[0] < model_frames:
                            pad = np.zeros(
                                (model_frames - clip_features.shape[0], feat_dim),
                                dtype=np.float32)
                            clip_features = np.concatenate([clip_features, pad], axis=0)

                    feat_tensor = torch.from_numpy(clip_features).float()
                    feat_tensor = feat_tensor.unsqueeze(0).to(device)

                    # Keypoint branch
                    wk = keypoints[0, start_frame:start_frame + WINDOW_SIZE, :, :]
                    kp_tensor = preprocess_keypoints(wk, image_width, image_height)
                    kp_tensor = kp_tensor.unsqueeze(0).to(device)

                    logits = model(feat_tensor, kp_tensor.float())

                else:
                    raise ValueError(f"Unsupported mode: {mode}")

            if isinstance(logits, dict):
                logits = logits['fused']
            pred_idx = logits.argmax(dim=1).item()
            pred_label = label_names[pred_idx + 1]  # 0-indexed model output -> 1-indexed label name

            results[video_name].append([pred_label, gt_label, start_frame])

    # Save results
    os.makedirs(args.out_dict_dir, exist_ok=True)
    output_path = os.path.join(args.out_dict_dir, args.out_dict_name)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    total_preds = sum(len(v) for v in results.values())
    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(results)} videos, {total_preds} predictions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sliding-window inference for dino_clip_features models')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory (contains config.json + best.pth)')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to annotation pkl file')
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Override feature directory from config')
    parser.add_argument('--out_dict_dir', type=str, required=True,
                        help='Output directory for result.pkl')
    parser.add_argument('--out_dict_name', type=str, default='result.pkl',
                        help='Output filename')
    parser.add_argument('--label_from', type=str, default='center',
                        help='Label source: center or end')
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--anno_split_num', type=int, default=0,
                        help='Total number of splits minus 1 (0 = no splitting)')
    parser.add_argument('--anno_split_pos', type=int, default=0,
                        help='Which split this process handles')
    args = parser.parse_args()

    WINDOW_SIZE = args.window_size
    run_inference(args)

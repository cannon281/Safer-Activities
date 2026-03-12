"""Batch 2D-to-3D pose lifting using MotionAGFormer.

Takes a SAFER-Activities pkl file with 2D keypoints (COCO 17-joint format)
and adds 3D keypoints lifted by MotionAGFormer-Base (243 frames, H3.6M).

The output pkl retains all original fields and adds:
  - keypoint_3d:       (num_people, num_frames, 17, 3)  hip-relative 3D coords
  - keypoint_3d_score: (num_people, num_frames)          mean 2D confidence

Usage (run from the MotionAGFormer repo root):
    python batch_3d_pose_processor.py \
        --input_pkl /path/to/aic_normal_dataset.pkl \
        --output_pkl /path/to/aic_normal_dataset_with_3d.pkl \
        --model_path checkpoint/motionagformer-b-h36m.pth.tr \
        --num_gpus 1
"""

import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add MotionAGFormer path
sys.path.append(os.getcwd())
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer


def coco_to_h36m(keypoints):
    """Convert COCO format keypoints to Human3.6M format."""
    new_keypoints = np.zeros_like(keypoints)

    # H36M joint mappings from COCO
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 12, :]
    new_keypoints[..., 2, :] = keypoints[..., 14, :]
    new_keypoints[..., 3, :] = keypoints[..., 16, :]
    new_keypoints[..., 4, :] = keypoints[..., 11, :]
    new_keypoints[..., 5, :] = keypoints[..., 13, :]
    new_keypoints[..., 6, :] = keypoints[..., 15, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 5, :]
    new_keypoints[..., 12, :] = keypoints[..., 7, :]
    new_keypoints[..., 13, :] = keypoints[..., 9, :]
    new_keypoints[..., 14, :] = keypoints[..., 6, :]
    new_keypoints[..., 15, :] = keypoints[..., 8, :]
    new_keypoints[..., 16, :] = keypoints[..., 10, :]

    return new_keypoints


def resample_keypoints(keypoints, target_frames=243):
    """Resample keypoints to target number of frames."""
    n_frames = keypoints.shape[0]
    if n_frames == target_frames:
        return keypoints

    indices = np.linspace(0, n_frames - 1, target_frames)
    indices = np.round(indices).astype(int)

    return keypoints[indices]


def check_valid_clip(clip, confidence_threshold=0.1):
    """Check if a clip contains valid pose data.

    Returns True if the clip has valid poses, False if all zeros.
    """
    # Check if all keypoints are zero (missing person)
    if np.all(clip[..., :2] == 0):
        return False

    # Check average confidence
    avg_confidence = np.mean(clip[..., 2])
    if avg_confidence < confidence_threshold:
        return False

    return True


def flip_data(data):
    """Flip data horizontally for test-time augmentation."""
    left_joints = [1, 2, 3, 14, 15, 16]
    right_joints = [4, 5, 6, 11, 12, 13]

    flipped_data = data.copy()
    flipped_data[..., 0] *= -1  # flip x coordinate

    # Swap left and right joints
    temp = flipped_data[..., left_joints, :].copy()
    flipped_data[..., left_joints, :] = flipped_data[..., right_joints, :]
    flipped_data[..., right_joints, :] = temp

    return flipped_data


def process_single_clip(clip, model, width, height, device='cuda'):
    """Process a single clip and return 3D poses."""
    # Check if clip is valid
    if not check_valid_clip(clip):
        # Return zeros for invalid clips
        return np.zeros((clip.shape[0], 17, 3)), np.zeros(clip.shape[0])

    # Separate coordinates and confidence
    clip_coords = clip[..., :2]
    clip_conf = clip[..., 2:3]

    # Normalize screen coordinates
    clip_normalized = normalize_screen_coordinates(clip_coords[None, ...], w=width, h=height)

    # Add confidence back
    clip_normalized = np.concatenate([clip_normalized, clip_conf[None, ...]], axis=-1)

    # Apply test-time augmentation
    clip_flipped = flip_data(clip_normalized)

    # Convert to tensor
    clip_tensor = torch.from_numpy(clip_normalized.astype('float32')).to(device)
    clip_flipped_tensor = torch.from_numpy(clip_flipped.astype('float32')).to(device)

    # Get predictions
    with torch.no_grad():
        output_3d = model(clip_tensor)
        output_3d_flipped = model(clip_flipped_tensor)

    # Convert to numpy and flip back
    output_3d = output_3d.cpu().numpy()
    output_3d_flipped = flip_data(output_3d_flipped.cpu().numpy())

    # Average predictions
    output_3d = (output_3d + output_3d_flipped) / 2

    # Set hip to origin
    output_3d[:, :, 0, :] = 0

    # Calculate confidence for 3D poses (average of 2D confidence)
    confidence_3d = np.mean(clip[..., 2], axis=1)

    return output_3d[0], confidence_3d


def process_video_annotation(annotation, model, device='cuda'):
    """Process a single video annotation and add 3D keypoints."""
    video_name = annotation['frame_dir']

    # Extract keypoints and scores
    keypoints = annotation['keypoint']  # Shape: [num_people, num_frames, 17, 2]
    scores = annotation['keypoint_score']  # Shape: [num_people, num_frames, 17]
    width = annotation['width']
    height = annotation['height']

    # Handle multi-person format
    if len(keypoints.shape) == 4:
        num_people = keypoints.shape[0]
    else:
        # Single person - add person dimension
        keypoints = keypoints[None, ...]
        scores = scores[None, ...]
        num_people = 1

    all_3d_keypoints = []
    all_3d_scores = []

    # Process each person
    for person_idx in range(num_people):
        person_keypoints = keypoints[person_idx]  # [frames, 17, 2]
        person_scores = scores[person_idx]  # [frames, 17]

        # Convert COCO to H36M format
        keypoints_h36m = coco_to_h36m(person_keypoints)

        # Add confidence as third dimension
        keypoints_with_conf = np.concatenate([keypoints_h36m, person_scores[..., None]], axis=-1)

        # Process in clips
        n_frames = keypoints_with_conf.shape[0]
        clip_len = 243
        poses_3d = []
        confidence_3d = []

        print(f"    Processing {n_frames} frames in {(n_frames + clip_len - 1) // clip_len} clips...")

        # Process clips
        for start_idx in range(0, n_frames, clip_len):
            end_idx = min(start_idx + clip_len, n_frames)
            clip = keypoints_with_conf[start_idx:end_idx]

            # Resample if needed
            if clip.shape[0] < clip_len:
                clip = resample_keypoints(clip, clip_len)

            # Process clip
            clip_3d, clip_conf = process_single_clip(clip, model, width, height, device)

            # Trim to original length
            frames_to_keep = min(end_idx - start_idx, clip_len)
            poses_3d.append(clip_3d[:frames_to_keep])
            confidence_3d.append(clip_conf[:frames_to_keep])

        # Concatenate all clips
        person_poses_3d = np.concatenate(poses_3d, axis=0)[:n_frames]
        person_confidence_3d = np.concatenate(confidence_3d, axis=0)[:n_frames]

        all_3d_keypoints.append(person_poses_3d)
        all_3d_scores.append(person_confidence_3d)

    # Stack all people
    all_3d_keypoints = np.stack(all_3d_keypoints, axis=0)  # [people, frames, 17, 3]
    all_3d_scores = np.stack(all_3d_scores, axis=0)  # [people, frames]

    # If original was single person, remove the person dimension
    if num_people == 1 and len(annotation['keypoint'].shape) == 3:
        all_3d_keypoints = all_3d_keypoints[0]
        all_3d_scores = all_3d_scores[0]

    # Add to annotation
    annotation['keypoint_3d'] = all_3d_keypoints.astype(np.float32)
    annotation['keypoint_3d_score'] = all_3d_scores.astype(np.float32)

    return annotation, video_name


def process_batch(annotations_batch, model_path, gpu_id, batch_idx=0, total_batches=1):
    """Process a batch of annotations on a specific GPU."""
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    print(f"\n[Batch {batch_idx + 1}/{total_batches}] Initializing model on {device}...")

    # Initialize MotionAGFormer-Base (243 frames)
    args = {
        'n_layers': 16, 'dim_in': 3, 'dim_feat': 128, 'dim_rep': 512, 'dim_out': 3,
        'mlp_ratio': 4, 'act_layer': nn.GELU, 'attn_drop': 0.0, 'drop': 0.0, 'drop_path': 0.0,
        'use_layer_scale': True, 'layer_scale_init_value': 0.00001, 'use_adaptive_fusion': True,
        'num_heads': 8, 'qkv_bias': False, 'qkv_scale': None, 'hierarchical': False,
        'use_temporal_similarity': True, 'neighbour_num': 2, 'temporal_connection_len': 1,
        'use_tcn': False, 'graph_only': False, 'n_frames': 243
    }

    model = MotionAGFormer(**args).to(device)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[gpu_id])

    # Load checkpoint
    pre_dict = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    print(f"[Batch {batch_idx + 1}/{total_batches}] Processing {len(annotations_batch)} videos...")

    # Process annotations
    results = []
    for i, annotation in enumerate(annotations_batch):
        video_name = annotation.get('frame_dir', 'unknown')
        print(f"[Batch {batch_idx + 1}/{total_batches}] Processing video {i+1}/{len(annotations_batch)}: {video_name}")

        try:
            processed_ann, video_name = process_video_annotation(annotation, model, device)
            results.append((processed_ann, video_name))

            if 'keypoint_3d' in processed_ann:
                shape = processed_ann['keypoint_3d'].shape
                print(f"  Successfully added 3D poses with shape {shape}")
            else:
                print(f"  Failed to add 3D poses")

        except Exception as e:
            print(f"  Error processing {video_name}: {str(e)}")
            results.append((annotation, video_name))

    print(f"[Batch {batch_idx + 1}/{total_batches}] Completed batch processing")
    return results


def main():
    parser = argparse.ArgumentParser(description='Add 3D poses to existing pkl file')
    parser.add_argument('--input_pkl', type=str, required=True,
                       help='Path to input pkl file with 2D poses')
    parser.add_argument('--output_pkl', type=str, required=True,
                       help='Path to output pkl file with 3D poses added')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to MotionAGFormer checkpoint')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use for parallel processing')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (will be slow)")
        args.num_gpus = 1
    else:
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")

    # Load pkl file
    print(f"\nLoading pkl file from {args.input_pkl}...")
    with open(args.input_pkl, 'rb') as f:
        data = pickle.load(f)

    # Extract components
    split_info = data['split']
    annotations = data['annotations']

    print(f"Found {len(annotations)} videos to process")

    # Split annotations into batches for parallel processing
    num_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    batch_size = (len(annotations) + num_gpus - 1) // num_gpus

    annotation_batches = []
    for i in range(0, len(annotations), batch_size):
        annotation_batches.append(annotations[i:i + batch_size])

    print(f"\nProcessing with {num_gpus} GPU(s) in {len(annotation_batches)} batches")
    print(f"Batch size: ~{batch_size} videos per batch")

    # Process in parallel
    if num_gpus > 1:
        print("\nStarting multi-GPU processing...")
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, batch in enumerate(annotation_batches):
                gpu_id = i % num_gpus
                future = executor.submit(process_batch, batch, args.model_path, gpu_id, i, len(annotation_batches))
                futures.append(future)

            all_results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_results = future.result()
                all_results.extend(batch_results)
    else:
        print("\nStarting single GPU/CPU processing...")
        all_results = []
        for i, batch in enumerate(annotation_batches):
            print(f"\n{'='*60}")
            print(f"Processing batch {i+1}/{len(annotation_batches)}")
            print(f"{'='*60}")
            batch_results = process_batch(batch, args.model_path, 0, i, len(annotation_batches))
            all_results.extend(batch_results)

            processed_so_far = len(all_results)
            print(f"\nOverall progress: {processed_so_far}/{len(annotations)} videos processed ({processed_so_far/len(annotations)*100:.1f}%)")

    # Sort results to maintain original order
    print("\nSorting results...")
    video_order = {ann['frame_dir']: i for i, ann in enumerate(annotations)}
    all_results.sort(key=lambda x: video_order.get(x[1], float('inf')))

    processed_annotations = []
    success_count = 0
    for processed_ann, video_name in all_results:
        processed_annotations.append(processed_ann)
        if 'keypoint_3d' in processed_ann:
            success_count += 1

    # Save updated data
    updated_data = {
        'split': split_info,
        'annotations': processed_annotations
    }

    print(f"\nSaving updated pkl file to {args.output_pkl}...")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(updated_data, f)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"- Total videos: {len(processed_annotations)}")
    print(f"- Successfully added 3D poses: {success_count}")
    print(f"- Failed: {len(processed_annotations) - success_count}")
    print(f"- Success rate: {success_count/len(processed_annotations)*100:.1f}%")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()

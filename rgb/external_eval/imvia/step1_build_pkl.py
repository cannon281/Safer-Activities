"""
Step 1: Extract keypoints from ImViA videos and build a SAFER-compatible pkl.

For each video:
  1. Read .avi frames, upscale 320x240 -> 1920x1080
  2. Run YOLOv8 + ViTPose per frame -> keypoints, scores, bboxes
  3. Parse ImViA annotation (fall_start, fall_end) -> per-frame labels
  4. Save as a single pkl file with all annotations

Usage:
    python step1_build_pkl.py --imvia_root /path/to/ImViA --output imvia_dataset.pkl
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from tqdm import tqdm

# Add inference directory to path for ViTPose imports
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
INFERENCE_DIR = os.path.join(PROJ_ROOT, 'inference')
sys.path.insert(0, INFERENCE_DIR)

from tools.src.vitpose_infer.main import VitInference
from tools.utils import resize_frame

ROOMS = ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02']

# SAFER label indices
LABEL_FALL = 10
LABEL_NO_LABEL = 0

# ViTPose model paths (relative to inference dir)
POSE_PATH = os.path.join(INFERENCE_DIR, 'det_pose_models', 'vitpose-b-multi-coco.pth')
DET_PATH = os.path.join(INFERENCE_DIR, 'det_pose_models', 'yolov8x.pt')


def get_annotation_dir(imvia_root, room):
    """Handle Coffee_room_02 using 'Annotations_files' (with 's')."""
    if room == 'Coffee_room_02':
        return os.path.join(imvia_root, room, 'Annotations_files')
    return os.path.join(imvia_root, room, 'Annotation_files')


def parse_imvia_annotation(txt_path, total_frames):
    """Parse ImViA annotation file to get per-frame labels.

    Format: line 1 = fall_start, line 2 = fall_end,
    lines 3+ = frame_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h

    Returns labels array of shape (total_frames,) with LABEL_FALL or LABEL_NO_LABEL.
    """
    labels = np.full(total_frames, LABEL_NO_LABEL, dtype=np.int64)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"  WARNING: Annotation {txt_path} has < 2 lines, skipping labels")
        return labels

    # Some videos have no falls — first line is bbox data (contains commas)
    # instead of a fall_start integer
    if ',' in lines[0]:
        print(f"  INFO: {txt_path} has no fall (bbox-only annotation)")
        return labels

    fall_start = int(lines[0].strip())
    fall_end = int(lines[1].strip())

    # Mark fall frames (1-indexed in annotation, 0-indexed in our array)
    for frame_idx in range(total_frames):
        frame_num = frame_idx + 1  # ImViA annotations are 1-indexed
        if fall_start <= frame_num <= fall_end:
            labels[frame_idx] = LABEL_FALL

    return labels


def process_video(video_path, txt_path, pose_model, frame_dir_name):
    """Process a single ImViA video through ViTPose.

    Returns an annotation dict compatible with SAFER pkl format, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"  WARNING: {video_path} has 0 frames")
        cap.release()
        return None

    # Preallocate arrays
    all_keypoints = np.zeros((total_frames, 17, 2), dtype=np.float32)
    all_scores = np.zeros((total_frames, 17), dtype=np.float32)
    all_bboxes = np.zeros((total_frames, 4), dtype=np.float32)

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"  {frame_dir_name}", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Upscale to 1080p for ViTPose
        frame_1080p = resize_frame(frame, "1080p")

        # Run ViTPose
        pts, tids, bboxes, drawn_frame, orig_frame, scores = pose_model.inference(
            frame_1080p, frame_idx)

        if len(pts) > 0:
            # Take the first detected person (ImViA is single-person)
            kp = pts[0]  # (17, 3) -> ViTPose outputs [Y, X, confidence]
            # Swap columns: ViTPose returns [Y, X] but we need [X, Y]
            all_keypoints[frame_idx, :, 0] = kp[:, 1]  # X from ViTPose col 1
            all_keypoints[frame_idx, :, 1] = kp[:, 0]  # Y from ViTPose col 0
            all_scores[frame_idx] = kp[:, 2]       # confidence

            # Bbox: bboxes from ViTPose are [x, y, w, h] (tlwh)
            bbox = bboxes[0]
            all_bboxes[frame_idx] = [bbox[0], bbox[1], bbox[2], bbox[3]]

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Trim to actual frame count (in case total_frames was wrong)
    actual_frames = frame_idx
    all_keypoints = all_keypoints[:actual_frames]
    all_scores = all_scores[:actual_frames]
    all_bboxes = all_bboxes[:actual_frames]

    # Parse annotation labels
    labels = parse_imvia_annotation(txt_path, actual_frames)

    # Build annotation dict matching SAFER pkl format
    annotation = {
        'frame_dir': frame_dir_name,
        'keypoint': all_keypoints[np.newaxis],       # (1, T, 17, 2)
        'keypoint_score': all_scores[np.newaxis],     # (1, T, 17)
        'labels': labels,                             # (T,)
        'img_shape': (1080, 1920),
        'width': 1920,
        'height': 1080,
        'bboxes': all_bboxes,                         # (T, 4)
        'total_frames': actual_frames,
    }

    return annotation


def main():
    parser = argparse.ArgumentParser(description='Build ImViA pkl from ViTPose keypoints')
    parser.add_argument('--imvia_root', type=str, required=True,
                        help='Path to ImViA dataset root')
    parser.add_argument('--output', type=str, default='imvia_dataset.pkl',
                        help='Output pkl path')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print("Initializing ViTPose + YOLOv8...")
    pose_model = VitInference(
        pose_path=POSE_PATH,
        det_type="yolov8",
        det_path=DET_PATH,
        tensorrt=False,
        tracking_method="bytetrack",
    )

    annotations = []
    video_names = []

    for room in ROOMS:
        video_dir = os.path.join(args.imvia_root, room, 'Videos')
        anno_dir = get_annotation_dir(args.imvia_root, room)

        if not os.path.isdir(video_dir):
            print(f"WARNING: Video dir not found: {video_dir}")
            continue

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.avi')])
        print(f"\n{room}: {len(video_files)} videos")

        for vf in video_files:
            video_path = os.path.join(video_dir, vf)
            stem = os.path.splitext(vf)[0]
            txt_path = os.path.join(anno_dir, stem + '.txt')

            if not os.path.exists(txt_path):
                print(f"  WARNING: No annotation for {vf}, skipping")
                continue

            frame_dir_name = f"{room}_{stem}"

            ann = process_video(video_path, txt_path, pose_model, frame_dir_name)
            if ann is not None:
                annotations.append(ann)
                video_names.append(frame_dir_name)
                fall_count = np.sum(ann['labels'] == LABEL_FALL)
                print(f"  {frame_dir_name}: {ann['total_frames']} frames, "
                      f"{fall_count} fall frames")

    # Build pkl
    pkl_data = {
        'split': {'sub_test': video_names},
        'annotations': annotations,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(pkl_data, f)

    print(f"\nSaved {args.output}")
    print(f"  {len(annotations)} annotations")
    print(f"  {len(video_names)} videos in sub_test split")

    # Quick verification
    if annotations:
        ann0 = annotations[0]
        print(f"\nSample annotation: {ann0['frame_dir']}")
        print(f"  keypoint shape: {ann0['keypoint'].shape}")
        print(f"  keypoint_score shape: {ann0['keypoint_score'].shape}")
        print(f"  labels shape: {ann0['labels'].shape}")
        print(f"  bboxes shape: {ann0['bboxes'].shape}")
        print(f"  total_frames: {ann0['total_frames']}")


if __name__ == '__main__':
    main()

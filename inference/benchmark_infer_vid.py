#!/usr/bin/env python3

import argparse
import torch
import cv2
import numpy as np
import time
import json
import os
from collections import deque
from scipy.signal import butter, filtfilt
from datetime import datetime

from tools.src.vitpose_infer.main import VitInference
from tools.model_setup import load_model_with_transforms
from tools.utils import nan_helper, resize_frame
from utils import ConfigParser
from utils.classification_model import cnn1d_infer

import sys
pyskl_path = "/home/work/inference/pyskl"
if pyskl_path not in sys.path:
    sys.path.append(pyskl_path)
import mmcv
from pyskl.apis import inference_recognizer, init_recognizer

MAIN_LABELS = ["stand", "stand_activity", "walk", "sit", "sit_activity", "sitting_down", "getting_up",
               "bend", "unstable", "fall", "lie_down", "lying_down", "reach", "run", "jump"]


def benchmark_model_on_video(video_path, model_name, model_type, config_path, weight_path, 
                            pose_model, device='cuda:0', num_frames=500):
    """Benchmark a model using actual video processing with timing measurements."""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*80}")
    
    # Load model
    if model_type == '1dcnn':
        cfg_parser = ConfigParser(config_path)
        with open(cfg_parser.dataset_cfg["mappings_json_file"], "r") as f:
            mappings = json.loads(f.read())[cfg_parser.dataset_cfg["dataset_type"]]
        num_classes = len(list(mappings["labels"].values())[1:])
        model, transforms = load_model_with_transforms(cfg_parser, num_classes, weight_path, device)
    else:  # pyskl
        config = mmcv.Config.fromfile(config_path)
        config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
        model = init_recognizer(config, weight_path, device)
        transforms = None
    
    model.eval()
    
    # Count parameters
    try:
        if model_type != '1dcnn' and hasattr(model, 'backbone'):
            param_source = model.backbone
        else:
            param_source = model
        total_params = sum(p.numel() for p in param_source.parameters() if p.requires_grad)
    except AttributeError:
        total_params = -1
    
    # Initialize variables (from your code)
    lstm_framecount = 48
    lstm_posedict = {}
    
    processing_times = {
        "pose_detection": [],
        "feature_extraction": [],
        "inference": [],
        "rendering": []
    }
    
    framecount = 0
    buffercount = 10
    number_joints = 17
    pose_results_buffer = deque([], maxlen=buffercount)
    pose_results_buffer_list = []
    pose_results_buffer_list_update = 0
    
    id_joints = {}
    
    # Filter setup
    fs = 17
    fc = 3
    wn = fc / (fs / 2)
    butter_b, butter_a = butter(2, wn)
    
    # Open video
    vid = cv2.VideoCapture(video_path)
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {num_frames} frames from video...")
    print(f"Video FPS: {video_fps:.2f}")
    
    # Process frames
    frames_processed = 0
    while frames_processed < num_frames:
        ret, frame = vid.read()
        if not ret:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Pose detection timing
        pose_start = time.time()
        frame = resize_frame(frame, "1080p")
        pts, tids, bboxes, drawn_frame, orig_frame, scores = pose_model.inference(frame, framecount)
        pose_end = time.time()
        processing_times["pose_detection"].append(pose_end - pose_start)
        
        # Convert to pose_results format
        pose_results = []
        for tid, bbox, pt in zip(tids, bboxes, pts):
            pt[:,[0, 1]] = pt[:,[1, 0]]
            pose_dict = {
                "bbox": bbox,
                "keypoints": pt,
                "track_id": tid,
            }
            pose_results.append(pose_dict)
        
        # Feature extraction timing
        feature_start = time.time()
        
        if framecount >= buffercount:
            if pose_results_buffer_list_update == 0:
                pose_results_buffer_list = list(pose_results_buffer)
                
                # Track IDs
                tracked_id = {}
                for frame in pose_results_buffer_list:
                    for person in frame:
                        if person['track_id'] in tracked_id:
                            tracked_id[person['track_id']] += 1
                        else:
                            tracked_id[person['track_id']] = 0
                
                # Initialize joints
                id_joints = {}
                for id in tracked_id.keys():
                    if tracked_id[id] < 2 or id == -1:
                        continue
                    id_joints[id] = [[0] * buffercount for n in range(number_joints * 3)]
                
                # Fill joint data
                frameidx = 0
                for frame in pose_results_buffer_list:
                    for person in frame:
                        person_id = person['track_id']
                        if person_id not in id_joints:
                            continue
                        
                        idx = 0
                        for kp in person['keypoints']:
                            id_joints[person_id][idx][frameidx] = kp[0]
                            id_joints[person_id][idx + 1][frameidx] = kp[1]
                            id_joints[person_id][idx + 2][frameidx] = kp[2]
                            idx += 3
                    frameidx += 1
                
                # Filter joints
                for track_id in id_joints.keys():
                    jidx = 0
                    for joint in id_joints[track_id]:
                        if not any(joint):
                            filter_joint = joint
                        else:
                            interp_joint = np.array(joint)
                            interp_joint[interp_joint == 0] = np.nan
                            nans, x = nan_helper(interp_joint)
                            if len(interp_joint[~nans]) > 0:
                                interp_joint[nans] = np.interp(x(nans), x(~nans), interp_joint[~nans])
                                filter_joint = filtfilt(butter_b, butter_a, interp_joint)
                            else:
                                filter_joint = joint
                        id_joints[track_id][jidx] = filter_joint
                        jidx += 1
                
                pose_results_buffer_list_update += 1
            
            frameidx = pose_results_buffer_list_update - 1
            feature_end = time.time()
            processing_times["feature_extraction"].append(feature_end - feature_start)
            
            # Inference timing
            inference_start = time.time()
            
            for track_id in id_joints.keys():
                format_joint = []
                
                for j in range(0, number_joints * 3, 3):
                    format_joint.append([
                        id_joints[track_id][j][frameidx],
                        id_joints[track_id][j+1][frameidx],
                        id_joints[track_id][j+2][frameidx]
                    ])
                
                if any(format_joint):
                    lstm_joints = []
                    for j in format_joint:
                        if j[0] == 0 and j[1] == 0:
                            continue
                        lstm_joints.extend([j[0], j[1], j[2]])
                    
                    if track_id in lstm_posedict:
                        lstm_posedict[track_id].append(lstm_joints)
                    else:
                        lstm_posedict[track_id] = deque([], maxlen=lstm_framecount)
                        lstm_posedict[track_id].append(lstm_joints)
                    
                    # Run inference when we have enough frames
                    if len(lstm_posedict[track_id]) == lstm_framecount:
                        if model_type == 'pyskl':
                            # PySkl inference
                            a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                            reshaped_array = a_input.reshape(1, 48, 17, 3)
                            window_keypoints = reshaped_array[:, :, :, :2]
                            window_scores = reshaped_array[:, :, :, 2]
                            
                            # Clip values
                            window_keypoints[..., 0] = np.clip(window_keypoints[..., 0], 0, 1920)
                            window_keypoints[..., 1] = np.clip(window_keypoints[..., 1], 0, 1080)
                            window_scores = np.nan_to_num(window_scores)
                            window_scores = np.clip(window_scores, 0, 1)
                            
                            fake_anno = dict(
                                frame_dir='',
                                label=-1,
                                img_shape=(1080, 1920),
                                original_shape=(1080, 1920),
                                start_index=0,
                                modality='Pose',
                                total_frames=window_keypoints.shape[1],
                                test_mode=True,
                                usable_indices=np.array([0]),
                                usable_label=np.array([1]),
                                keypoint=window_keypoints,
                                keypoint_score=window_scores
                            )
                            
                            _ = inference_recognizer(model, fake_anno)
                        else:
                            # 1D-CNN inference
                            a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                            _ = cnn1d_infer(model, transforms, device, a_input[0],
                                          return_secondary=False, return_confidence=True)
                        
                        frames_processed += 1
                        break  # Only process one person per frame for benchmarking
            
            inference_end = time.time()
            processing_times["inference"].append(inference_end - inference_start)
            
            # Update buffer
            pose_results_buffer_list_update += 1
            if pose_results_buffer_list_update == buffercount + 1:
                pose_results_buffer_list_update = 0
            
            pose_results_buffer.popleft()
            prev_pos_results = pose_results.copy()
            pose_results_buffer.append(prev_pos_results)
        
        else:
            # Still building up buffer
            feature_end = time.time()
            processing_times["feature_extraction"].append(feature_end - feature_start)
            
            inference_start = time.time()
            inference_end = time.time()
            processing_times["inference"].append(inference_end - inference_start)
            
            prev_pos_results = pose_results.copy()
            pose_results_buffer.append(prev_pos_results)
        
        # Minimal rendering timing (no actual display), only placeholder for now
        render_start = time.time()
        render_end = time.time()
        processing_times["rendering"].append(render_end - render_start)
        
        framecount += 1
        
        # Print progress
        if framecount % 100 == 0:
            print(f"  Processed {framecount} frames...")
    
    vid.release()
    
    # Calculate statistics
    results = {}
    for component, times in processing_times.items():
        if times:
            results[component] = {
                'mean_ms': np.mean(times) * 1000,
                'std_ms': np.std(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'max_ms': np.max(times) * 1000,
                'count': len(times)
            }
    
    # Calculate total and percentages
    total_mean = sum(r['mean_ms'] for r in results.values())
    for component, stats in results.items():
        stats['percentage'] = (stats['mean_ms'] / total_mean) * 100
    
    # Overall stats
    overall_stats = {
        'model_name': model_name,
        'total_parameters': total_params,
        'frames_processed': frames_processed,
        'total_mean_ms': total_mean,
        'fps': 1000 / total_mean if total_mean > 0 else 0,
        'video_fps': video_fps,
        'components': results
    }
    
    # Print summary
    print(f"\nPerformance Summary for {model_name}:")
    print(f"  Parameters: {total_params/1e6:.2f}M" if total_params > 0 else "  Parameters: N/A")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Total time per frame: {total_mean:.1f}ms")
    print(f"  Processing FPS: {overall_stats['fps']:.1f}")
    print(f"\nComponent Breakdown:")
    for component, stats in results.items():
        print(f"  {component}: {stats['mean_ms']:.1f}ms ({stats['percentage']:.1f}%)")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return overall_stats


def main():
    parser = argparse.ArgumentParser(description='Benchmark models using actual video processing')
    parser.add_argument('--video_path', type=str, default='videos/day_normal_p05_cam7.mp4',
                       help='Path to video file for benchmarking')
    parser.add_argument('--num_frames', type=int, default=500,
                       help='Number of frames to process per model')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Model Performance Benchmark - Real Video Processing")
    print("="*80)
    print(f"Video: {args.video_path}")
    print(f"Frames per model: {args.num_frames}")
    print(f"Device: {args.device}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize pose model (shared for all benchmarks)
    print("\nInitializing pose detection model...")
    pose_model = VitInference(
        pose_path='det_pose_models/vitpose-b-multi-coco.pth',
        det_type="yolov8",
        det_path='det_pose_models/yolov8x.pt',
        tensorrt=False,
        tracking_method="bytetrack"
    )
    
    # Define models to benchmark
    models = {
        '1D-CNN': {
            'type': '1dcnn',
            'config': 'configs/CNN1D_kp.py',
            'weight': 'weights/CNN1D_kp.pt'
        },
        'ST-GCN++': {
            'type': 'pyskl',
            'config': 'pyskl/configs/stgcn++/safer_activity_xsub/non-wheelchair.py',
            'weight': 'weights/stgcn-non-wheelchair-epoch_16.pth'
        },
        'MSG-3D': {
            'type': 'pyskl',
            'config': 'pyskl/configs/msg3d/safer_activity_xsub/non-wheelchair.py',
            'weight': 'weights/msg3d-non-wheelchair-epoch_16.pth'
        },
        'PoseC3D': {
            'type': 'pyskl',
            'config': 'pyskl/configs/posec3d/safer_activity_xsub/non-wheelchair.py',
            'weight': 'weights/non-wheelchair-epoch_44.pth'
        }
    }
    
    # Run benchmarks
    all_results = {}
    for model_name, model_info in models.items():
        try:
            results = benchmark_model_on_video(
                args.video_path,
                model_name,
                model_info['type'],
                model_info['config'],
                model_info['weight'],
                pose_model,
                args.device,
                args.num_frames
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    # Model comparison table
    print(f"\n{'Model':<15} {'Parameters':<12} {'Total (ms)':<12} {'FPS':<10} {'Inference (ms)':<15} {'Inference %':<12}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        params = f"{results['total_parameters']/1e6:.2f}M" if results['total_parameters'] > 0 else "N/A"
        total_ms = f"{results['total_mean_ms']:.1f}"
        fps = f"{results['fps']:.1f}"
        
        inference_ms = "N/A"
        inference_pct = "N/A"
        if 'inference' in results['components']:
            inference_ms = f"{results['components']['inference']['mean_ms']:.1f}"
            inference_pct = f"{results['components']['inference']['percentage']:.1f}%"
        
        print(f"{model_name:<15} {params:<12} {total_ms:<12} {fps:<10} {inference_ms:<15} {inference_pct:<12}")
    
    # Component breakdown comparison
    print("\n" + "="*80)
    print("COMPONENT BREAKDOWN (ms)")
    print("="*80)
    
    components = ['pose_detection', 'feature_extraction', 'inference', 'rendering']
    print(f"{'Model':<15}", end='')
    for comp in components:
        print(f" {comp[:8]:<10}", end='')
    print()
    print("-"*80)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15}", end='')
        for comp in components:
            if comp in results['components']:
                ms = results['components'][comp]['mean_ms']
                print(f" {ms:<10.1f}", end='')
            else:
                print(f" {'N/A':<10}", end='')
        print()
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'benchmark_results/benchmark_result_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Bottleneck analysis
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        components = results['components']
        sorted_components = sorted(components.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        for comp, stats in sorted_components[:2]:  # Top 2 bottlenecks
            print(f"  - {comp}: {stats['mean_ms']:.1f}ms ({stats['percentage']:.1f}%)")
        
        if results['fps'] < results['video_fps']:
            print(f"Processing FPS ({results['fps']:.1f}) < Video FPS ({results['video_fps']:.1f})")
            print(f"Cannot process in real-time!")
        else:
            print(f"Can process in real-time ({results['fps']:.1f} FPS)")


if __name__ == "__main__":
    main()
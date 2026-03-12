"""
Datasets for training on pre-extracted CLIP/DINOv3/VideoMAE features
and multimodal skeleton+RGB fusion.

Sampling logic matches keypoints_train/datasets/datasets.py:
  - Same sliding windows, same usable_indices, same epoch-based seeding
  - Same splits (sub_train/sub_test), same label extraction (center majority)

Dataset modes:
  - "keypoint":      Keypoints only (B, 68, 48) — for standalone 1D-CNN
  - "feature":       Pre-extracted CLIP/DINOv3/VideoMAE features (B, 16, D)
  - "fusion":        Features + keypoints (B, 16, D) + (B, 68, 48)
"""

import os
import pickle
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from transforms import (apply_transforms, get_test_transforms,
                        get_train_transforms)


def get_majority_labels(labels, center_frames, label_from="center"):
    """Identical to keypoints_train/utils/dataset_utils.get_majority_labels."""
    if label_from == "center":
        start = len(labels) // 2 - center_frames // 2
        end = start + center_frames
        center_labels = labels[start:end]
    else:
        center_labels = labels[-center_frames:]
    label_counts = Counter(center_labels)
    return label_counts.most_common(1)[0][0]


# =============================================================================
# Feature-based datasets
# =============================================================================

class FeatureDataset(Dataset):
    """Training dataset for pre-extracted CLIP/DINOv3/VideoMAE features.
    Matches keypoints_train AICActivityDataset sampling exactly.
    """

    def __init__(self, ann_file, feature_dir, split=None,
                 clip_length=48, clip_majority_frames=5,
                 skip_window_length=20, skip_frames=False, skip_stride=1,
                 model_frames=16, feat_dim=768, center_frame_only=False):
        self.feature_dir = feature_dir
        self.split_name = split
        self.clip_length = clip_length
        self.clip_majority_frames = clip_majority_frames
        self.skip_window_length = skip_window_length
        self.skip_frames = skip_frames
        self.skip_stride = skip_stride
        self.model_frames = model_frames
        self.feat_dim = feat_dim
        self.center_frame_only = center_frame_only

        self.is_train = split is not None and 'train' in split
        self.epoch = 0

        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        split_set = set(data['split'][split]) if split else None
        self.annotations = [
            a for a in data['annotations']
            if split_set is None or a['frame_dir'] in split_set
        ]

        print(f"{split} split len: {len(split_set) if split_set else 'all'}")
        print(f"Total annotations in {split}: {len(self.annotations)}")

        self.feature_cache = {}
        self._load_features()

        # Cache keypoints for validity check (matching keypoints_train)
        self.keypoint_cache = {}
        for ann in self.annotations:
            name = ann['frame_dir']
            if name not in self.keypoint_cache:
                self.keypoint_cache[name] = np.array(ann['keypoint'])[0]

        self.clips = []
        for ann in self.annotations:
            self._process_annotation(ann)

        print(f"Total usable indices in the {split}: "
              f"{self._count_usable_indices()}")
        print(f"Total clips: {len(self.clips)}")

    def _load_features(self):
        loaded = set()
        missing = 0
        for ann in self.annotations:
            name = ann['frame_dir']
            if name in loaded:
                continue
            feat_path = os.path.join(self.feature_dir, f"{name}.npy")
            if os.path.exists(feat_path):
                self.feature_cache[name] = np.load(feat_path, mmap_mode='r')
                loaded.add(name)
            else:
                missing += 1
        if missing:
            print(f"  WARNING: {missing} feature files not found")

    def _process_annotation(self, annotation):
        total_frames = annotation['total_frames']
        frame_dir = annotation['frame_dir']

        if frame_dir not in self.feature_cache:
            return

        kp = self.keypoint_cache.get(frame_dir)
        if kp is None:
            return

        for start_frame in range(0, total_frames - self.clip_length + 1,
                                 self.skip_window_length):
            end_frame = start_frame + self.clip_length + \
                self.skip_window_length
            end_frame = min(end_frame, total_frames)

            labels = annotation['labels'][start_frame:end_frame]
            usable_indices = []

            for sub_start in range(
                    0, min(self.skip_window_length,
                           end_frame - start_frame - self.clip_length + 1)):
                sub_end = sub_start + self.clip_length
                sub_labels = labels[sub_start:sub_end]

                if get_majority_labels(
                        sub_labels, self.clip_majority_frames, "center") == 0:
                    continue

                # Keypoint validity check (matching keypoints_train)
                sub_kp = kp[start_frame + sub_start:
                            start_frame + sub_end]
                num_zero_keypoints = np.sum(sub_kp == 0)
                if num_zero_keypoints <= 170:
                    usable_indices.append(sub_start)

            if len(usable_indices) > 0:
                self.clips.append({
                    'frame_dir': frame_dir,
                    'start_frame': start_frame,
                    'labels': np.array(labels, dtype=np.int8),
                    'usable_indices': usable_indices,
                })

    def _count_usable_indices(self):
        return sum(len(c['usable_indices']) for c in self.clips)

    def _get_clip_params(self, idx):
        """Get sub-window parameters for this sample (shared across modes)."""
        clip = self.clips[idx]
        if self.is_train:
            random.seed(self.epoch * 10000 + idx)
        else:
            random.seed(idx)

        random_start_index = random.choice(clip['usable_indices'])
        sub_clip_start = random_start_index
        sub_labels = clip['labels'][sub_clip_start:
                                     sub_clip_start + self.clip_length].tolist()
        label = get_majority_labels(
            sub_labels, self.clip_majority_frames, "center") - 1
        abs_start = clip['start_frame'] + sub_clip_start
        return clip['frame_dir'], abs_start, label

    def _get_subsampled_features(self, frame_dir, abs_start):
        """Get subsampled features for this clip."""
        features = self.feature_cache[frame_dir]
        abs_end = abs_start + self.clip_length

        if self.center_frame_only:
            center_idx = abs_start + self.clip_length // 2
            center_idx = np.clip(center_idx, 0, len(features) - 1)
            return np.array(features[center_idx:center_idx + 1])

        if self.skip_frames:
            effective_stride = self.skip_stride * 3
            sampled_indices = np.arange(abs_start, abs_end, effective_stride)
        else:
            sampled_indices = np.arange(abs_start, abs_end, 3)

        sampled_indices = sampled_indices[:self.model_frames]
        sampled_indices = np.clip(sampled_indices, 0, len(features) - 1)
        clip_features = np.array(features[sampled_indices])

        if clip_features.shape[0] < self.model_frames:
            pad = np.zeros((self.model_frames - clip_features.shape[0],
                            self.feat_dim), dtype=np.float32)
            clip_features = np.concatenate([clip_features, pad], axis=0)

        return clip_features

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_dir, abs_start, label = self._get_clip_params(idx)
        clip_features = self._get_subsampled_features(frame_dir, abs_start)

        return {
            'features': torch.from_numpy(clip_features).float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


class FeatureTestDataset(Dataset):
    """Test dataset for pre-extracted features (stride-1 sliding window)."""

    def __init__(self, ann_file, feature_dir, split=None,
                 clip_length=48, clip_majority_frames=5,
                 skip_window_length=20, skip_frames=False, skip_stride=1,
                 model_frames=16, feat_dim=768, center_frame_only=False):
        self.feature_dir = feature_dir
        self.clip_length = clip_length
        self.clip_majority_frames = clip_majority_frames
        self.skip_frames = skip_frames
        self.skip_stride = skip_stride
        self.model_frames = model_frames
        self.feat_dim = feat_dim
        self.center_frame_only = center_frame_only

        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        split_set = set(data['split'][split]) if split else None
        annotations = [
            a for a in data['annotations']
            if split_set is None or a['frame_dir'] in split_set
        ]

        self.feature_cache = {}
        loaded = set()
        for ann in annotations:
            name = ann['frame_dir']
            if name not in loaded:
                feat_path = os.path.join(feature_dir, f"{name}.npy")
                if os.path.exists(feat_path):
                    self.feature_cache[name] = np.load(feat_path,
                                                        mmap_mode='r')
                    loaded.add(name)

        # Cache keypoints for validity check (matching keypoints_train)
        self.keypoint_cache = {}
        for ann in annotations:
            name = ann['frame_dir']
            if name not in self.keypoint_cache:
                self.keypoint_cache[name] = np.array(ann['keypoint'])[0]

        self.clips = []
        for ann in annotations:
            self._process_annotation(ann)

        print(f"[{split}] Test clips: {len(self.clips)}")

    def _process_annotation(self, annotation):
        total_frames = annotation['total_frames']
        frame_dir = annotation['frame_dir']

        if frame_dir not in self.feature_cache:
            return

        kp = self.keypoint_cache.get(frame_dir)
        if kp is None:
            return

        for start_frame in range(0, total_frames - self.clip_length + 1, 1):
            end_frame = start_frame + self.clip_length
            labels = annotation['labels'][start_frame:end_frame]

            if get_majority_labels(
                    labels, self.clip_majority_frames, "center") == 0:
                continue

            # Keypoint validity check (matching keypoints_train)
            sub_kp = kp[start_frame:end_frame]
            num_zero_keypoints = np.sum(sub_kp == 0)
            if num_zero_keypoints >= 170:
                continue

            self.clips.append({
                'frame_dir': frame_dir,
                'start_frame': start_frame,
                'labels': labels,
            })

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        features = self.feature_cache[clip['frame_dir']]
        abs_start = clip['start_frame']
        abs_end = abs_start + self.clip_length

        label = get_majority_labels(
            clip['labels'], self.clip_majority_frames, "center") - 1

        if self.center_frame_only:
            center_idx = abs_start + self.clip_length // 2
            center_idx = np.clip(center_idx, 0, len(features) - 1)
            clip_features = np.array(features[center_idx:center_idx + 1])
        else:
            if self.skip_frames:
                effective_stride = self.skip_stride * 3
                sampled_indices = np.arange(abs_start, abs_end,
                                            effective_stride)
            else:
                sampled_indices = np.arange(abs_start, abs_end, 3)

            sampled_indices = sampled_indices[:self.model_frames]
            sampled_indices = np.clip(sampled_indices, 0, len(features) - 1)
            clip_features = np.array(features[sampled_indices])

            if clip_features.shape[0] < self.model_frames:
                pad = np.zeros((self.model_frames - clip_features.shape[0],
                                self.feat_dim), dtype=np.float32)
                clip_features = np.concatenate([clip_features, pad], axis=0)

        return {
            'features': torch.from_numpy(clip_features).float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# Keypoint-only datasets (for standalone 1D-CNN)
# =============================================================================

class KeypointDataset(Dataset):
    """Training dataset for keypoint-only models (e.g., standalone 1D-CNN).

    Same sliding window / usable_indices / epoch-based seeding as FeatureDataset,
    but loads only keypoints from the pkl (no pre-extracted features needed).
    """

    def __init__(self, ann_file, split=None,
                 clip_length=48, clip_majority_frames=5,
                 skip_window_length=20, skip_frames=False, skip_stride=1,
                 image_width=None, keypoint_field='keypoint', **kwargs):
        self.split_name = split
        self.clip_length = clip_length
        self.clip_majority_frames = clip_majority_frames
        self.skip_window_length = skip_window_length
        self.skip_frames = skip_frames
        self.skip_stride = skip_stride

        self.is_train = split is not None and 'train' in split
        self.epoch = 0

        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        split_set = set(data['split'][split]) if split else None
        self.annotations = [
            a for a in data['annotations']
            if split_set is None or a['frame_dir'] in split_set
        ]

        print(f"{split} split len: {len(split_set) if split_set else 'all'}")
        print(f"Total annotations in {split}: {len(self.annotations)}")

        # Cache keypoints per video: (1, T, 17, 2) → (T, 17, 2)
        self.keypoint_cache = {}
        for ann in self.annotations:
            name = ann['frame_dir']
            if name not in self.keypoint_cache:
                self.keypoint_cache[name] = np.array(ann[keypoint_field])[0]

        # Determine image width based on which keypoint field is used
        if image_width is None:
            if keypoint_field == 'keypoint_480p':
                image_width = 640
            else:
                img_shape = self.annotations[0].get('img_shape')
                image_width = img_shape[1] if img_shape else 1920
        self.image_width = image_width

        self.kp_train_transforms = get_train_transforms(image_width=image_width)
        self.kp_test_transforms = get_test_transforms(image_width=image_width)

        self.clips = []
        for ann in self.annotations:
            self._process_annotation(ann)

        print(f"Total usable indices in the {split}: "
              f"{self._count_usable_indices()}")
        print(f"Total clips: {len(self.clips)}")

    def _process_annotation(self, annotation):
        total_frames = annotation['total_frames']
        frame_dir = annotation['frame_dir']

        kp = self.keypoint_cache.get(frame_dir)
        if kp is None:
            return

        for start_frame in range(0, total_frames - self.clip_length + 1,
                                 self.skip_window_length):
            end_frame = start_frame + self.clip_length + \
                self.skip_window_length
            end_frame = min(end_frame, total_frames)

            labels = annotation['labels'][start_frame:end_frame]
            usable_indices = []

            for sub_start in range(
                    0, min(self.skip_window_length,
                           end_frame - start_frame - self.clip_length + 1)):
                sub_end = sub_start + self.clip_length
                sub_labels = labels[sub_start:sub_end]

                if get_majority_labels(
                        sub_labels, self.clip_majority_frames, "center") == 0:
                    continue

                sub_kp = kp[start_frame + sub_start:
                            start_frame + sub_end]
                num_zero_keypoints = np.sum(sub_kp == 0)
                if num_zero_keypoints <= 170:
                    usable_indices.append(sub_start)

            if len(usable_indices) > 0:
                self.clips.append({
                    'frame_dir': frame_dir,
                    'start_frame': start_frame,
                    'labels': np.array(labels, dtype=np.int8),
                    'usable_indices': usable_indices,
                })

    def _count_usable_indices(self):
        return sum(len(c['usable_indices']) for c in self.clips)

    def _get_clip_params(self, idx):
        clip = self.clips[idx]
        if self.is_train:
            random.seed(self.epoch * 10000 + idx)
        else:
            random.seed(idx)

        random_start_index = random.choice(clip['usable_indices'])
        sub_clip_start = random_start_index
        sub_labels = clip['labels'][sub_clip_start:
                                     sub_clip_start + self.clip_length].tolist()
        label = get_majority_labels(
            sub_labels, self.clip_majority_frames, "center") - 1
        abs_start = clip['start_frame'] + sub_clip_start
        return clip['frame_dir'], abs_start, label

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_dir, abs_start, label = self._get_clip_params(idx)

        kp = self.keypoint_cache[frame_dir]
        abs_end = abs_start + self.clip_length
        clip_kp = kp[abs_start:abs_end].copy()

        if clip_kp.shape[0] < self.clip_length:
            pad = np.zeros((self.clip_length - clip_kp.shape[0], 17, 2),
                           dtype=np.float32)
            clip_kp = np.concatenate([clip_kp, pad], axis=0)

        clip_kp = torch.from_numpy(clip_kp).float()
        transforms = self.kp_train_transforms if self.is_train \
            else self.kp_test_transforms
        clip_kp = apply_transforms(clip_kp, transforms)

        return {
            'keypoints': clip_kp.float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


class KeypointTestDataset(Dataset):
    """Test dataset for keypoint-only models (stride-1 sliding window)."""

    def __init__(self, ann_file, split=None,
                 clip_length=48, clip_majority_frames=5,
                 skip_window_length=20, skip_frames=False, skip_stride=1,
                 image_width=None, keypoint_field='keypoint', **kwargs):
        self.clip_length = clip_length
        self.clip_majority_frames = clip_majority_frames

        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        split_set = set(data['split'][split]) if split else None
        annotations = [
            a for a in data['annotations']
            if split_set is None or a['frame_dir'] in split_set
        ]

        self.keypoint_cache = {}
        for ann in annotations:
            name = ann['frame_dir']
            if name not in self.keypoint_cache:
                self.keypoint_cache[name] = np.array(ann[keypoint_field])[0]

        # Determine image width based on which keypoint field is used
        if image_width is None:
            if keypoint_field == 'keypoint_480p':
                image_width = 640
            else:
                img_shape = annotations[0].get('img_shape')
                image_width = img_shape[1] if img_shape else 1920
        self.image_width = image_width

        self.kp_transforms = get_test_transforms(image_width=image_width)

        self.clips = []
        for ann in annotations:
            total_frames = ann['total_frames']
            frame_dir = ann['frame_dir']

            kp = self.keypoint_cache.get(frame_dir)
            if kp is None:
                continue

            for start_frame in range(0, total_frames - clip_length + 1, 1):
                end_frame = start_frame + clip_length
                labels = ann['labels'][start_frame:end_frame]

                if get_majority_labels(
                        labels, clip_majority_frames, "center") == 0:
                    continue

                sub_kp = kp[start_frame:end_frame]
                num_zero_keypoints = np.sum(sub_kp == 0)
                if num_zero_keypoints >= 170:
                    continue

                self.clips.append({
                    'frame_dir': frame_dir,
                    'start_frame': start_frame,
                    'labels': labels,
                })

        print(f"[{split}] Keypoint test clips: {len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        abs_start = clip['start_frame']
        abs_end = abs_start + self.clip_length

        label = get_majority_labels(
            clip['labels'], self.clip_majority_frames, "center") - 1

        kp = self.keypoint_cache[clip['frame_dir']]
        clip_kp = kp[abs_start:abs_end].copy()
        if clip_kp.shape[0] < self.clip_length:
            pad = np.zeros((self.clip_length - clip_kp.shape[0], 17, 2),
                           dtype=np.float32)
            clip_kp = np.concatenate([clip_kp, pad], axis=0)
        clip_kp = torch.from_numpy(clip_kp).float()
        clip_kp = apply_transforms(clip_kp, self.kp_transforms)

        return {
            'keypoints': clip_kp.float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# Fusion datasets (pre-extracted features + keypoints)
# =============================================================================

class FusionDataset(FeatureDataset):
    """Training dataset returning both pre-extracted features AND keypoints.

    Extends FeatureDataset: same clips, same sampling.
    Additionally loads keypoints from pkl and applies transforms.
    """

    def __init__(self, *args, image_width=None, keypoint_field='keypoint',
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Cache keypoints per video: (1, T, 17, 2) → (T, 17, 2)
        self.keypoint_cache = {}
        for ann in self.annotations:
            name = ann['frame_dir']
            if name not in self.keypoint_cache and name in self.feature_cache:
                kp = np.array(ann[keypoint_field])[0]  # Remove person dim
                self.keypoint_cache[name] = kp

        # Determine image width based on which keypoint field is used
        if image_width is None:
            if keypoint_field == 'keypoint_480p':
                image_width = 640
            else:
                img_shape = self.annotations[0].get('img_shape')
                image_width = img_shape[1] if img_shape else 1920
        self.image_width = image_width

        self.kp_train_transforms = get_train_transforms(image_width=image_width)
        self.kp_test_transforms = get_test_transforms(image_width=image_width)

    def __getitem__(self, idx):
        frame_dir, abs_start, label = self._get_clip_params(idx)

        # Features (subsampled to 16 frames)
        clip_features = self._get_subsampled_features(frame_dir, abs_start)

        # Keypoints (full clip_length frames for 1D-CNN)
        kp = self.keypoint_cache[frame_dir]
        abs_end = abs_start + self.clip_length
        clip_kp = kp[abs_start:abs_end].copy()  # (clip_length, 17, 2)

        # Pad if short
        if clip_kp.shape[0] < self.clip_length:
            pad = np.zeros((self.clip_length - clip_kp.shape[0], 17, 2),
                           dtype=np.float32)
            clip_kp = np.concatenate([clip_kp, pad], axis=0)

        clip_kp = torch.from_numpy(clip_kp).float()
        transforms = self.kp_train_transforms if self.is_train \
            else self.kp_test_transforms
        clip_kp = apply_transforms(clip_kp, transforms)  # (68, clip_length)

        return {
            'features': torch.from_numpy(clip_features).float(),
            'keypoints': clip_kp.float(),
            'label': torch.tensor(label, dtype=torch.long),
            'idx': idx,
        }


class FusionTestDataset(FeatureTestDataset):
    """Test dataset returning both pre-extracted features AND keypoints."""

    def __init__(self, *args, image_width=None, keypoint_field='keypoint',
                 **kwargs):
        # Need annotations for keypoints — store before super().__init__
        # Re-load pkl to get keypoints
        ann_file = args[0] if args else kwargs.get('ann_file')
        split = kwargs.get('split')

        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        split_set = set(data['split'][split]) if split else None
        annotations = [
            a for a in data['annotations']
            if split_set is None or a['frame_dir'] in split_set
        ]

        self._keypoint_cache = {}
        for ann in annotations:
            name = ann['frame_dir']
            if name not in self._keypoint_cache:
                kp = np.array(ann[keypoint_field])[0]
                self._keypoint_cache[name] = kp

        # Determine image width based on which keypoint field is used
        if image_width is None:
            if keypoint_field == 'keypoint_480p':
                image_width = 640
            else:
                img_shape = annotations[0].get('img_shape')
                image_width = img_shape[1] if img_shape else 1920
        self._image_width = image_width

        super().__init__(*args, **kwargs)
        self.keypoint_cache = self._keypoint_cache
        del self._keypoint_cache
        self.image_width = self._image_width

        self.kp_transforms = get_test_transforms(image_width=self.image_width)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        features = self.feature_cache[clip['frame_dir']]
        abs_start = clip['start_frame']
        abs_end = abs_start + self.clip_length

        label = get_majority_labels(
            clip['labels'], self.clip_majority_frames, "center") - 1

        # Features
        if self.center_frame_only:
            center_idx = abs_start + self.clip_length // 2
            center_idx = np.clip(center_idx, 0, len(features) - 1)
            clip_features = np.array(features[center_idx:center_idx + 1])
        else:
            if self.skip_frames:
                effective_stride = self.skip_stride * 3
                sampled_indices = np.arange(abs_start, abs_end,
                                            effective_stride)
            else:
                sampled_indices = np.arange(abs_start, abs_end, 3)
            sampled_indices = sampled_indices[:self.model_frames]
            sampled_indices = np.clip(sampled_indices, 0, len(features) - 1)
            clip_features = np.array(features[sampled_indices])
            if clip_features.shape[0] < self.model_frames:
                pad = np.zeros((self.model_frames - clip_features.shape[0],
                                self.feat_dim), dtype=np.float32)
                clip_features = np.concatenate([clip_features, pad], axis=0)

        # Keypoints
        kp = self.keypoint_cache[clip['frame_dir']]
        clip_kp = kp[abs_start:abs_end].copy()
        if clip_kp.shape[0] < self.clip_length:
            pad = np.zeros((self.clip_length - clip_kp.shape[0], 17, 2),
                           dtype=np.float32)
            clip_kp = np.concatenate([clip_kp, pad], axis=0)
        clip_kp = torch.from_numpy(clip_kp).float()
        clip_kp = apply_transforms(clip_kp, self.kp_transforms)

        return {
            'features': torch.from_numpy(clip_features).float(),
            'keypoints': clip_kp.float(),
            'label': torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# Unified dataloader builder
# =============================================================================

def build_dataloaders(ann_file, mode='feature', preprocess='sequential',
                      model_frames=16, feat_dim=768, batch_size=128,
                      num_workers=4, seed=111,
                      # Feature-specific
                      feature_dir=None,
                      center_frame_only=False,
                      # Keypoint field selection
                      keypoint_field='keypoint',
                      # Split names
                      train_split='sub_train', test_split='sub_test',
                      # Eval settings (default to train values)
                      eval_batch_size=None, eval_num_workers=None):
    """Build train, val, and test dataloaders for any mode.

    Returns 3 loaders:
      - train_loader: windowed (stride-20) on train split, epoch-based seeding
      - val_loader:   windowed (stride-20) on test split, fixed seed
                      (matches original keypoints_train: same dataset class
                       for both train and val during training)
      - test_loader:  stride-1 on test split, for final evaluation only

    Args:
        mode: 'keypoint', 'feature', or 'fusion'
        preprocess: 'sequential' (48→16) or 'skip' (144→16)
    """

    if preprocess == 'sequential':
        ds_kwargs = dict(clip_length=48, clip_majority_frames=5,
                         skip_frames=False, skip_stride=1)
    else:
        ds_kwargs = dict(clip_length=144, clip_majority_frames=15,
                         skip_frames=True, skip_stride=3)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if mode == 'keypoint':
        common = dict(ann_file=ann_file, skip_window_length=20,
                      keypoint_field=keypoint_field, **ds_kwargs)
        train_dataset = KeypointDataset(split=train_split, **common)
        val_dataset = KeypointDataset(split=test_split, **common)
        test_dataset = KeypointTestDataset(split=test_split, **common)

    elif mode == 'feature':
        common = dict(ann_file=ann_file, feature_dir=feature_dir,
                      skip_window_length=20, model_frames=model_frames,
                      feat_dim=feat_dim,
                      center_frame_only=center_frame_only, **ds_kwargs)
        train_dataset = FeatureDataset(split=train_split, **common)
        val_dataset = FeatureDataset(split=test_split, **common)
        test_dataset = FeatureTestDataset(split=test_split, **common)

    elif mode == 'fusion':
        common = dict(ann_file=ann_file, feature_dir=feature_dir,
                      skip_window_length=20, model_frames=model_frames,
                      feat_dim=feat_dim,
                      center_frame_only=center_frame_only,
                      keypoint_field=keypoint_field, **ds_kwargs)
        train_dataset = FusionDataset(split=train_split, **common)
        val_dataset = FusionDataset(split=test_split, **common)
        test_dataset = FusionTestDataset(split=test_split, **common)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if eval_batch_size is None:
        eval_batch_size = batch_size
    if eval_num_workers is None:
        eval_num_workers = num_workers

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=eval_num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=eval_num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader

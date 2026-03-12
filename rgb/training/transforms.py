"""
Keypoint transforms for the 1D-CNN branch in fusion models.

Copied from keypoints_train/internal_transforms/transforms.py to keep
this codebase self-contained. These are applied identically to the
keypoints_train pipeline for fair comparison.
"""

import random

import torch


class HorizontalFlip:
    """Flip keypoints horizontally with given probability."""

    def __init__(self, probability=0.5, image_width=1920):
        self.probability = probability
        self.left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
        self.right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
        self.image_width = image_width

    def __call__(self, keypoints):
        if random.random() < self.probability:
            keypoints[..., 0] = self.image_width - keypoints[..., 0]
            keypoints_copy = keypoints.clone()
            for left_idx, right_idx in zip(self.left_kp, self.right_kp):
                keypoints[:, left_idx, :] = keypoints_copy[:, right_idx, :]
                keypoints[:, right_idx, :] = keypoints_copy[:, left_idx, :]
        return keypoints


class GaussianNoise:
    """Add Gaussian noise to 2D keypoints."""

    def __init__(self, mean=0, std=0.5, pixel_range=2):
        self.mean = mean
        self.std = std
        self.pixel_range = pixel_range

    def __call__(self, keypoints):
        num_frames, num_joints, _ = keypoints.shape
        for frame_idx in range(num_frames):
            frame = keypoints[frame_idx]
            if not torch.all(frame == 0):
                noise = torch.normal(self.mean, self.std,
                                     size=(num_joints, 2))
                noise = torch.clamp(noise, -self.pixel_range, self.pixel_range)
                keypoints[frame_idx] += noise
        return keypoints


class ScaleWithNeckMotion:
    """Scale keypoints relative to neck and compute motion differences.

    Output: (T, 17, 4) — [scaled_x, scaled_y, motion_x, motion_y]
    """

    def __init__(self, image_width=1920):
        self.image_width = image_width

    def __call__(self, keypoints):
        num_frames, num_joints, _ = keypoints.shape

        motion_info = torch.zeros_like(keypoints)
        scaled_keypoints = torch.zeros_like(keypoints)

        # Motion: frame-to-frame difference, normalized by image width
        for frame_idx in range(num_frames - 1):
            current_frame = keypoints[frame_idx]
            next_frame = keypoints[frame_idx + 1]
            if not torch.all(current_frame == 0) and \
               not torch.all(next_frame == 0):
                motion_info[frame_idx] = \
                    (next_frame - current_frame) / self.image_width

        # Scale relative to neck position
        for frame_idx in range(num_frames):
            frame = keypoints[frame_idx].clone()
            if not torch.all(frame == 0):
                neck_x = (frame[5, 0] + frame[6, 0]) / 2
                neck_y = (frame[5, 1] + frame[6, 1]) / 2
                max_y, min_y = frame[:, 1].max(), frame[:, 1].min()
                y_range = max_y - min_y if max_y - min_y > 0 else 1
                frame[:, 0] = (frame[:, 0] - neck_x) / y_range
                frame[:, 1] = (frame[:, 1] - neck_y) / y_range
                scaled_keypoints[frame_idx] = frame

        # Concatenate: (T, 17, 2) + (T, 17, 2) → (T, 17, 4)
        return torch.cat((scaled_keypoints, motion_info), dim=-1)


class To1DInputShape:
    """Reshape keypoints for 1D-CNN input.

    (T, 17, C) → flatten → (T, 17*C) → transpose → (17*C, T)
    """

    def __call__(self, keypoints):
        num_frames, num_joints, num_info = keypoints.shape
        keypoints_flattened = keypoints.reshape(num_frames,
                                                num_joints * num_info)
        return keypoints_flattened.transpose(0, 1)


def get_train_transforms(image_width=1920):
    """Training transforms for keypoint preprocessing."""
    return [
        HorizontalFlip(probability=0.5, image_width=image_width),
        GaussianNoise(mean=0, std=0.5, pixel_range=2),
        ScaleWithNeckMotion(image_width=image_width),
        To1DInputShape(),
    ]


def get_test_transforms(image_width=1920):
    """Test transforms for keypoint preprocessing."""
    return [
        ScaleWithNeckMotion(image_width=image_width),
        To1DInputShape(),
    ]


def apply_transforms(keypoints, transforms):
    """Apply a list of transforms sequentially."""
    for t in transforms:
        keypoints = t(keypoints)
    return keypoints

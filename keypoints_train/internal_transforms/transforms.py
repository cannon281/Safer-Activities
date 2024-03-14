import torch
import random

class BaseTransform:
    def __call__(self, x, y = None):
        raise NotImplementedError


class HorizontalFlip(BaseTransform):
    def __init__(self, probability=0.5, image_width = 1920):
        self.probability = probability
        # Update these indices according to your keypoints format
        self.left_kp = [1, 3, 5, 7, 9, 11, 13, 15]  # Left keypoints indices
        self.right_kp = [2, 4, 6, 8, 10, 12, 14, 16]  # Right keypoints indices
        self.image_width = image_width  # Width of a 1080p image

    def __call__(self, keypoints):
        if random.random() < self.probability:
            # Flip x-coordinates
            keypoints[..., 0] = self.image_width - keypoints[..., 0]
            
            # Create a copy to avoid mixing up when swapping
            keypoints_copy = keypoints.clone()

            # Swap left and right body parts
            for left_idx, right_idx in zip(self.left_kp, self.right_kp):
                keypoints[:, left_idx, :] = keypoints_copy[:, right_idx, :]
                keypoints[:, right_idx, :] = keypoints_copy[:, left_idx, :]
                
        return keypoints
    

class NormalizeKeypoints(BaseTransform):
    def __call__(self, keypoints, image_width=1920):
        return keypoints/image_width

    
class ScaleWithNeck(BaseTransform):
    def __call__(self, keypoints):
        num_frames, num_joints, _ = keypoints.shape

        for frame_idx in range(num_frames):
            frame = keypoints[frame_idx]
            if not torch.all(frame == 0):  # Apply transformation if frame is not all zeros
                # Calculate neck coordinates
                neck_x = (frame[5, 0] + frame[6, 0]) / 2
                neck_y = (frame[5, 1] + frame[6, 1]) / 2

                # Scale coordinates relative to neck position
                max_y, min_y = frame[:, 1].max(), frame[:, 1].min()
                y_range = max_y - min_y if max_y - min_y > 0 else 1  # Prevent division by zero
                
                frame[:, 0] = (frame[:, 0] - neck_x) / y_range
                frame[:, 1] = (frame[:, 1] - neck_y) / y_range

        return keypoints



class ScaleWithNeckMotion(BaseTransform):
    def __init__(self, image_width=1920):
        self.image_width = image_width  # Set the normalization factor as the image width

    def __call__(self, keypoints):
        num_frames, num_joints, _ = keypoints.shape
        
        # Initialize a tensor to store the motion information
        motion_info = torch.zeros_like(keypoints)
        scaled_keypoints = torch.zeros_like(keypoints)
        
        for frame_idx in range(num_frames):
            frame = keypoints[frame_idx]
            if not torch.all(frame == 0):  # Apply transformation if frame is not all zeros
                # Calculate neck coordinates
                neck_x = (frame[5, 0] + frame[6, 0]) / 2
                neck_y = (frame[5, 1] + frame[6, 1]) / 2

                # Scale coordinates relative to neck position
                max_y, min_y = frame[:, 1].max(), frame[:, 1].min()
                y_range = max_y - min_y if max_y - min_y > 0 else 1  # Prevent division by zero
                
                frame[:, 0] = (frame[:, 0] - neck_x) / y_range
                frame[:, 1] = (frame[:, 1] - neck_y) / y_range

                # Store scaled frame back
                scaled_keypoints[frame_idx] = frame  # Assume scaling logic is applied

                if frame_idx < num_frames - 1:
                    next_frame = keypoints[frame_idx + 1]
                    if not torch.all(next_frame == 0):
                        # Calculate motion difference
                        motion_info[frame_idx] = next_frame - frame
        
        # Normalize motion information
        motion_info = motion_info / self.image_width  # Normalize to [-1, 1] range
        
        # Concatenate scaled keypoints with normalized motion information
        augmented_keypoints = torch.cat((scaled_keypoints, motion_info), dim=-1)
        
        return augmented_keypoints





class GaussianNoise(BaseTransform):
    def __init__(self, mean=0, std=0.5, pixel_range=2):
        self.mean = mean
        self.std = std
        self.pixel_range = pixel_range

    def __call__(self, keypoints):
        num_frames, num_joints, _ = keypoints.shape

        for frame_idx in range(num_frames):
            frame = keypoints[frame_idx]
            if not torch.all(frame == 0):  # Only apply noise to non-zero frames
                noise = torch.normal(self.mean, self.std, size=(num_joints, 2))
                noise = torch.clamp(noise, -self.pixel_range, self.pixel_range)
                
                # Add noise
                keypoints[frame_idx] += noise

        return keypoints


class To1DInputShape(BaseTransform):
    def __call__(self, keypoints):
        num_frames, num_joints, num_info = keypoints.shape
        # Flatten the keypoints from [num_frames, 17, 2] to [num_frames, 34]
        # Flatten the keypoints from [num_frames, 17, 4] to [num_frames, 68] If there is motion info
        keypoints_flattened = keypoints.reshape(num_frames, num_joints * num_info)
        # Transpose the tensor to have shape [_, num_frames]
        keypoints_model_input = keypoints_flattened.transpose(0, 1)
        return keypoints_model_input
    
    
class Compose(BaseTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y=None):
        for transform in self.transforms:
            x = transform(x,y) if y is not None else transform(x)
        return x

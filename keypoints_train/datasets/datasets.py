from torch.utils.data import Dataset
import numpy as np
import random
import pickle
import torch
from utils.dataset_utils import get_majority_labels

# --------------------------------------------------------------------------------------------------------------------------------------------

class BaseAICActivityDataset(Dataset):
    def __init__(self, pickle_file_path, split_type, 
                 clip_length=144, clip_majority_frames = 15, skip_frames=False, skip_stride = 0,
                 skip_window_length=20, 
                 transforms=None,
                 apply_transforms = False, 
                 label_from="center",
                 target_type="classification"):
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)

        self.split = data["split"][split_type]
        self.clip_length = clip_length
        self.clip_majority_frames = clip_majority_frames # Length of the center frames of the clip (Set to 15)
        self.skip_window_length = skip_window_length # 20 for skipping every 20 frames for fetching a clip
        self.skip_frames = skip_frames # True or False, whether to skip frames during training
        self.skip_stride = skip_stride
        
        self.label_from = label_from # If "center", then will fetch the label from the center frames, if "end" will do so from the end
        
        self.transforms = transforms
        self.apply_transforms = apply_transforms
        
        self.is_train = True if "train" in split_type else False
        
        self.epoch = 0  # Initial epoch
        self.annotations = [annotation for annotation in data["annotations"] if annotation["frame_dir"] in self.split]
        print(split_type, "split len:", len(self.split))
        print(f"Total annotations in {split_type}: {len(self.annotations)}")
        
        self.target_type = target_type

        # Create a list to store the sub-clips information
        self.clips = []
        for annotation in self.annotations:
            self.process_annotation(annotation)
            
        print(f"Total usable indices in the {split_type}: {self.count_usable_indices()}")


    def process_annotation(self, annotation):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
                    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        
    def __len__(self):
        return len(self.clips)
    
    
    def count_usable_indices(self):
        if len(self.clips) == 0:
            print("self.clips is not populated yet")
            return 0
        
        count_usable_indices = 0
        for i in range(len(self.clips)):
            count_usable_indices += len(self.clips[i]['usable_indices'])
        
        return count_usable_indices


# --------------------------------------------------------------------------------------------------------------------------------------------



class AICActivityDataset(BaseAICActivityDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process_annotation(self, annotation):
        total_frames = annotation["total_frames"]
        
        for start_frame in range(0, total_frames - self.clip_length + 1, self.skip_window_length):
            end_frame = start_frame + self.clip_length + self.skip_window_length
            end_frame = min(end_frame, total_frames)
            
            labels = annotation["labels"][start_frame:end_frame]
            keypoints = annotation["keypoint"][0, start_frame:end_frame]
            

            # Initialize usable_indices for this window
            usable_indices = []
        
            for sub_start in range(0, min(self.skip_window_length, end_frame - start_frame - self.clip_length + 1)):
                # Correct calculation of sub_end to use it for slicing labels and keypoints
                sub_end = sub_start + self.clip_length
                sub_labels = labels[sub_start:sub_end]  # Use sub_end for slicing
                sub_keypoints = keypoints[sub_start:sub_end, :, :]  # Correctly slicing according to the shape
                
                # Check for the majority of labels being 0
                if get_majority_labels(sub_labels, self.clip_majority_frames, self.label_from) == 0:
                    continue  # Skip this sub-window
                
                # Count zero keypoints in the sub-window
                num_zero_keypoints = np.sum(sub_keypoints == 0)
                
                # Ensure that label exists for the given sub-window
                # Ensure keypoints equivalent to less than 5 frames (34 * 5 = 170) only have zero keypoints, otherwise, reject
                if num_zero_keypoints <= 170:
                    # If this sub-window is valid, add its start index (relative to the beginning of the window) to usable_indices
                    usable_indices.append(sub_start)

            # If there are any usable indices, append the window information to self.clips
            if len(usable_indices) > 0:
                keypoints = keypoints.astype(np.float16)
                labels = np.array(labels, dtype=np.int8)
                self.clips.append({
                    "keypoints": keypoints,
                    "labels": labels,
                    "frame_dir": annotation["frame_dir"],
                    "usable_indices": usable_indices
                })


    def __getitem__(self, idx):
        clip = self.clips[idx]
        
        if self.is_train:
            random.seed(self.epoch * 10000 + idx)  # Ensure unique seed for each combination of epoch and idx
        else:
            random.seed(idx) # Else the same seed for every idx

        # Select a random starting point for the sub-clip
        random_start_index = random.choice(clip['usable_indices'])
        sub_clip_start = random_start_index
        sub_clip_end = sub_clip_start + self.clip_length
        
        # Always extract the keypoints for the entire window
        keypoints = torch.from_numpy(clip["keypoints"][sub_clip_start:sub_clip_end, :, :]).float()
        labels = clip["labels"].tolist()[sub_clip_start:sub_clip_end]
        
            
        if self.skip_frames:
            # Skip frames according to skip_frame_count
            keypoints = keypoints[::self.skip_stride, :, :]
            
        # If classification
        if self.target_type == "classification":
            # Determine the label based on the specified method
            label = get_majority_labels(labels, self.clip_majority_frames, self.label_from)
        
        # If segmentation
        else:
            if self.skip_frames:
                # Each label corresponds to each frame
                label = labels[::self.skip_stride]    
            else:
                label = labels
                
            # The length of labels must match the length of keypoints
            assert len(label) == keypoints.shape[0], "Length of labels and keypoints do not match for segmentation"
        
        if self.transforms and self.apply_transforms:
            keypoints = self.transforms(keypoints)
        
        if isinstance(label, list):
            label = torch.tensor([l-1 for l in label])
        else:    
            label = label - 1  # Adjust for starting the labels from 0. Mapping needs adjustment during inference for predictions
        
        return keypoints, label
    
    
# --------------------------------------------------------------------------------------------------------------------------------------------


class AICActivityTestDataset(BaseAICActivityDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process_annotation(self, annotation):
        total_frames = annotation["total_frames"]
        
        step_size = self.skip_window_length//2 if self.target_type == "segmentation" else 1
        
        for start_frame in range(0, total_frames - self.clip_length + 1, step_size):
            end_frame = start_frame + self.clip_length
            
            labels = annotation["labels"][start_frame:end_frame]
            keypoints = annotation["keypoint"][0, start_frame:end_frame]

            # Check for the majority of labels being 0
            if get_majority_labels(labels, self.clip_majority_frames, self.label_from) == 0:
                continue  # Skip this sub-window
                
            # Count zero keypoints in the sub-window
            num_zero_keypoints = np.sum(keypoints == 0)
                
            # Ensure that label exists for the given window
            # Ensure keypoints equivalent to more than 5 frames (34 * 5 = 170) are not zero
            if num_zero_keypoints >= 170:
                continue
            
            if self.skip_frames:
                # Skip frames according to skip_frame_count
                # For test, done here so that it is more memory-efficient
                keypoints = keypoints[::self.skip_stride, :, :]
                
            self.clips.append({
                "keypoints": keypoints,
                "labels": labels,
                "frame_dir": annotation["frame_dir"],
                "start_frame": start_frame
            })

    def count_usable_indices(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        keypoints = torch.from_numpy(clip["keypoints"]).float()
        labels = clip["labels"]

        # If classification
        if self.target_type == "classification":
            # Determine the label based on the specified method
            label = get_majority_labels(labels, self.clip_majority_frames, self.label_from)
        
        # If segmentation
        else:
            if self.skip_frames:
                # Each label corresponds to each frame
                label = labels[::self.skip_stride]    
            else:
                label = labels
                
            # The length of labels must match the length of keypoints
            assert len(label) == keypoints.shape[0], "Length of labels and keypoints do not match for segmentation"


        if self.transforms and self.apply_transforms:
            keypoints = self.transforms(keypoints)
            
        if isinstance(label, list):
            label = torch.tensor([l-1 for l in label])
        else:    
            label = label - 1  # Adjust for starting the labels from 0. Mapping needs adjustment during inference for predictions
        
        
        return keypoints, label
    
    
# --------------------------------------------------------------------------------------------------------------------------------------------
    
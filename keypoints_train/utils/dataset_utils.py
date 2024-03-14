from torch.utils.data import random_split
from internal_transforms import Compose
from torch.utils.data import DataLoader
import copy
from collections import Counter


def get_majority_center_label(labels, num_frames):
    # Calculate the start and end indices for the center frames
    start = len(labels) // 2 - num_frames // 2
    end = start + num_frames

    # Extract the center labels
    center_labels = labels[start:end]

    # Count the frequency of each label in the center frames
    label_counts = Counter(center_labels)

    # Return the most common label
    return label_counts.most_common(1)[0][0]


def get_majority_end_label(labels, num_frames):
    # Calculate the start and end indices for the center frames
    start = len(labels) - num_frames
    end = start + num_frames

    # Extract the center labels
    center_labels = labels[start:end]

    # Count the frequency of each label in the center frames
    label_counts = Counter(center_labels)

    # Return the most common label
    return label_counts.most_common(1)[0][0]


def get_majority_labels(labels, num_frames, type="center"):
    return get_majority_center_label(labels, num_frames) if type=="center" else get_majority_end_label(labels, num_frames)



def get_dataset_with_transforms(dataset, transforms_dict):
    dataset_copy = copy.deepcopy(dataset)
    transforms = Compose([transform_params['transform_class'](**transform_params['hyperparams']) for transform_params in transforms_dict.values()])
    try:
        dataset_copy.dataset.transforms = transforms
        return dataset_copy
    except AttributeError:
        dataset_copy.transforms = transforms
        dataset_copy.apply_transform = True
        return dataset_copy


def get_dataloader(dataset, dataloader_kwargs):
    return DataLoader(dataset, **dataloader_kwargs)
import argparse
import torch
from tools.model_setup import load_model_with_transforms
from tools.utils import load_mappings_and_initialize_variables
from utils import ConfigParser
import os
import json
import pickle
import numpy as np
from tqdm import tqdm

from utils.classification_model import cnn1d_infer
from utils.dataset_utils import get_majority_labels

# Config
# --------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Infer Action Classification')
parser.add_argument('--pkl_path', type=str, default="../keypoints_train/data/aicactivity/normal/aic_normal_dataset.pkl")
parser.add_argument('--config_path', type=str, default="configs/CNN1D_kp.py")
parser.add_argument('--weight_path', type=str, default="weights/CNN1D_kp.pt")
parser.add_argument('--label_from', type=str, default = "center")
parser.add_argument('--window_size', type=int, default=48)
parser.add_argument('--out_dict_dir', type=str, default="./out_dict")
parser.add_argument('--out_dict_name', type=str, default="result.pkl")
parser.add_argument('--anno_split_num', type=int, default=0)
parser.add_argument('--anno_split_pos', type=int, default=0)

args = parser.parse_args()

pkl_path = args.pkl_path
config_path = args.config_path
weight_path = args.weight_path
label_from = args.label_from
window_size = args.window_size
anno_split_num = args.anno_split_num
anno_split_pos = args.anno_split_pos
output_dict_dir = args.out_dict_dir
output_dict_name = args.out_dict_name

output_dir = 'out_results_dict'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
assert os.path.exists(config_path), f"Config file {config_path} does not exist"
try:
    cfg_parser = ConfigParser(config_path)
except:
    raise Exception(f"Error parsing config file {config_path}. Are you sure it is a valid config file?")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

f = open(cfg_parser.dataset_cfg["mappings_json_file"], "r")
mappings = json.loads(f.read())[cfg_parser.dataset_cfg["dataset_type"]]

label_dict = mappings["labels"]
labels = list(label_dict.values())[1:] # Remove the no_label, which is the first one in the list
num_classes = len(labels)

mapping_labels, _, label_count, actionmap, _, _ = load_mappings_and_initialize_variables(cfg_parser.dataset_cfg["mappings_json_file"], mode = "normal")

# Action classification model with transforms
# --------------------------------------------------------------------------------------------------------------------------------------------
model, transforms = load_model_with_transforms(cfg_parser, num_classes, weight_path, device)
# --------------------------------------------------------------------------------------------------------------------------------------------

with open(pkl_path, 'rb') as file:
    pkl_data = pickle.load(file)

test_split = pkl_data['split']['sub_test']
test_annotations = [annotation for annotation in pkl_data["annotations"] if annotation["frame_dir"] in test_split]

if args.anno_split_num > 0:
    split_size = len(test_annotations) // (args.anno_split_num + 1)
    splits = [test_annotations[i * split_size:(i + 1) * split_size] for i in range(args.anno_split_num)]
    splits.append(test_annotations[args.anno_split_num * split_size:])
    if args.anno_split_pos < 0 or args.anno_split_pos > args.anno_split_num:
        raise ValueError("anno_split_pos must be between 0 and anno_split_num")
    test_annotations = splits[args.anno_split_pos]
else:
    test_annotations = test_annotations

results = {}

i = 0
for ann in tqdm(test_annotations, desc=f"Processing Annotations for split {args.anno_split_pos}/{args.anno_split_num}"):
    i += 1
    results[ann['frame_dir']] = []
    keypoints = ann['keypoint']
    keypoint_scores = ann['keypoint_score']
    labels = ann['labels']
     
    num_frames = keypoints.shape[1]  # The total number of frames in the video
    
    for start_frame in tqdm(range(num_frames - window_size + 1), desc=f"Sliding Windows {args.anno_split_pos}", leave=False):
        if window_size == 48:
            window_keypoints = keypoints[:, start_frame:start_frame + window_size, :, :]
            window_scores = keypoint_scores[:, start_frame:start_frame + window_size, :]
        else:
            window_keypoints = keypoints[:, start_frame:start_frame + window_size:3, :, :]
            window_scores = keypoint_scores[:, start_frame:start_frame + window_size:3, :]
        
        # Add an extra dimension to window_scores and concatenate it with window_keypoints
        window_scores_expanded = np.expand_dims(window_scores, axis=-1)
        window_with_scores = np.concatenate((window_keypoints, window_scores_expanded), axis=-1)[0]

        predicted_action, confidence = cnn1d_infer(model, transforms, device, window_with_scores, return_secondary=False, return_confidence=True)
        window_labels = labels[start_frame:start_frame + window_size] if window_size == 48 else labels[start_frame:start_frame + window_size:3]
        label_this_window = get_majority_labels(window_labels, num_frames=5, type=label_from)
        label_this_window = mapping_labels[label_this_window]
        results[ann['frame_dir']].append([predicted_action, label_this_window, start_frame])

# --------------------------------------------------------------------------------------------------------------------------------------------

os.makedirs(output_dict_dir, exist_ok=True)  # Ensure the directory exists

output_path = os.path.join(output_dict_dir, output_dict_name)
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
import os
import cv2 
import numpy as np
import pandas as pd
import argparse
import pickle
from utils import load_mappings_and_initialize_variables, get_action_dict_from_csv, get_action_from_frame_timestamp

# Argument defaults
 
parser = argparse.ArgumentParser(description='Extract Keypoints')
parser.add_argument('--resolution', type=str, default="1080p", required=False)
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--extract_keypoints_dir', type=str, required=True)
parser.add_argument('--video_file', type=str, required=True)
parser.add_argument('--pickle_file', type=str, default="output_data.pkl")
parser.add_argument('--mappings_json_path', type=str, default='mappings.json')
parser.add_argument('--extract_mode', type=str, default='normal') # normal or wheelchair

args = parser.parse_args()

# Path to the pickle file
pickle_file_path = args.pickle_file


# Load the mappings and initialize variables
MAIN_LABELS, ALL_LABELS, label_count, actionmap, all_ylabelmap, all_label_count = load_mappings_and_initialize_variables(args.mappings_json_path, 
                                                                                                                         args.extract_mode)


# Function to load existing data from pickle file
def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

# Function to save data to pickle file
def save_data_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to '{file_path}'")

# ----------------------------------------------------------------

action_label_map = get_action_dict_from_csv(args.csv_file)
# print(action_label_map)

framecount = 0
skip_frames = 0
count = 0

# Initialize data storage lists
labels_all = []
all_labels_all = []

def add_labels_to_all(labels, all_labels):
    labels_all.append(labels)
    all_labels_all.append(all_labels)


# Variable for video dimensions
video_width, video_height = None, None

vid = cv2.VideoCapture(args.video_file)
length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv2.CAP_PROP_FPS)
vid.release()  # No longer need to keep the video file open

# Initialize label count dictionaries
label_count = {i:0 for i in MAIN_LABELS}
all_label_count = {i:0 for i in ALL_LABELS}

for framecount in range(length):
    current_timestamp = framecount / fps
    action = get_action_from_frame_timestamp(action_label_map, current_timestamp)

    # Check if no person is detected or no label available
    action = action if action else "no_label"
    label = actionmap[action]
    label_count[MAIN_LABELS[label]] += 1

    all_label = all_ylabelmap[action]
    all_label_count[ALL_LABELS[all_label]] += 1
    
    add_labels_to_all(labels=label, all_labels=all_label)

    if (framecount + 1) % 1000 == 0:
        print(f"Processed {framecount + 1}/{length} frames.")
        print("Current label count:", label_count)
        print("Current all label count:", all_label_count)


# Convert lists to numpy arrays
labels_all = np.array(labels_all)
all_labels_all = np.array(all_labels_all)

# Load existing data from pickle file
existing_data = load_existing_data(pickle_file_path)

existing_data[0]["labels"] = labels_all
existing_data[0]["full_labels"] = all_labels_all

if existing_data[0]["keypoint"].dtype == np.float64:
    existing_data[0]["keypoint"] = existing_data[0]["keypoint"].astype(np.float32)
if existing_data[0]["keypoint_score"].dtype == np.float64:
    existing_data[0]["keypoint_score"] = existing_data[0]["keypoint_score"].astype(np.float32)
if existing_data[0]["bbox"].dtype == np.float64:
    existing_data[0]["bbox"] = existing_data[0]["bbox"].astype(np.float32)

# Save updated data to pickle file
save_data_to_pickle(pickle_file_path, existing_data)

print("End")
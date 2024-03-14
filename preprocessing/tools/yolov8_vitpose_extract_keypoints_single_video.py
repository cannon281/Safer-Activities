import os
import sys
from src import VitInference
import cv2 
import numpy as np
import pandas as pd
import argparse
import pickle
from utils import load_mappings_and_initialize_variables, get_action_dict_from_csv, get_action_from_frame_timestamp
from pose_utils import process_pose_results

# ----------------------------------------------------------------

# Arguments and defaults
 
parser = argparse.ArgumentParser(description='Extract Keypoints')
parser.add_argument('--save_out_video', type=bool, default=False, required=False)
parser.add_argument('--out_video_root', type=str, default="./out_video", required=False)
parser.add_argument('--out_action_root', type=str, default="./out_action", required=False)
parser.add_argument('--resolution', type=str, default="1080p", required=False)
parser.add_argument('--vitpose_model_path', type=str, default='models/vitpose-l-multi-coco.pth')
parser.add_argument('--yolo_path', type=str, default='yolov8x.pt')
parser.add_argument('--mappings_json_path', type=str, default='mappings.json')
parser.add_argument('--extract_mode', type=str, default='normal') # normal or wheelchair

parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--extract_keypoints_dir', type=str, required=True)
parser.add_argument('--video_file', type=str, required=True)
parser.add_argument('--pickle_file', type=str, default="output_data.pkl")

# ----------------------------------------------------------------

args = parser.parse_args()

model = VitInference(args.vitpose_model_path,
            det_type="yolov8", tensorrt=False, tracking_method="bytetrack")

# Path to the pickle file
pickle_file_path = args.pickle_file

# Load the mappings and initialize variables
MAIN_LABELS, ALL_LABELS, label_count, actionmap, all_ylabelmap, all_label_count = load_mappings_and_initialize_variables(args.mappings_json_path, 
                                                                                                                         args.extract_mode)

# ----------------------------------------------------------------

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

framecount = 0

# ----------------------------------------------------------------

vid = cv2.VideoCapture(args.video_file)
length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
print( "Length of the video:", length )
fps = vid.get(cv2.CAP_PROP_FPS)

if args.save_out_video:
    os.makedirs(args.out_video_root, exist_ok=True)
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    if args.resolution == "1080p":
        size = (int(1920),
                int(1080))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        os.path.join(args.out_video_root,
                        f'vis_{os.path.basename(args.video_file)}'), fourcc,
        fps, size)

action_label_map = get_action_dict_from_csv(args.csv_file)


# ----------------------------------------------------------------

assert vid.isOpened(), f'Failed to load video file {args.video_file}'

pose_results = []
framecount = 0

print("Extracting from", args.video_file, "with fps", fps, "and resolution", size)

# Initialize data storage lists
keypoints_all = []
keypoint_scores_all = []
labels_all = []
bboxes_all = []

def add_data_to_all(keypoints, keypoint_scores, labels, bboxes):
    keypoints_all.append(keypoints)
    keypoint_scores_all.append(keypoint_scores)
    labels_all.append(labels)
    bboxes_all.append(bboxes)


# Variable for video dimensions
video_width, video_height = None, None

previous_track_id = None

# For displaying the label or text in out_video and out_action
print_font = cv2.FONT_HERSHEY_SIMPLEX
print_font_size = 1.5
print_coord = (100, 100)

# ----------------------------------------------------------------

while True:
    ret,frame = vid.read()
    if ret:
        if args.resolution == "1080p":
            frame = cv2.resize(frame, dsize=(1920, 1080))#, interpolation=cv2.INTER_AREA)

        # Update video dimensions
        if video_width is None or video_height is None:
            video_height, video_width = frame.shape[:2]
            
        # Calculate the reference point (between the center and the bottom of the frame)
        reference_y = int((video_height / 2) + (video_height / 4))  # 3/4 down from the top
        reference_point = (int(video_width / 2), reference_y)

        framecount = framecount + 1
        current_timestamp = framecount/fps

        action = get_action_from_frame_timestamp(action_label_map, current_timestamp)
        
        # Check if no person is detected or no label available
        action = action if action else "no_label"
        label=actionmap[action]
        label_count[MAIN_LABELS[label]]+=1
        
# ----------------------------------------------------------------

        # 384x640 is the input image resolution to yolo
        pts,tids,bboxes,drawn_frame,orig_frame,bbox_scores = model.inference(frame,framecount)

        pose_results = []
        # Reshape data in the same format as pose_results from mmpose
        for tid, bbox, pt, bbox_score in zip(tids, bboxes, pts, bbox_scores):
            pt[:,[0, 1]] = pt[:,[1, 0]]
            pose_dict = {
                "bbox": bbox,   
                "keypoints": pt,
                "track_id": tid,
                "bbox_score":bbox_score,
            }
            pose_results.append(pose_dict)
            
        
# ----------------------------------------------------------------

        drawn_frame, keypoints, keypoint_scores, label, bboxes, previous_track_id = process_pose_results(drawn_frame, 
                                                                                                         pose_results, 
                                                                                                         action, 
                                                                                                         video_width, 
                                                                                                         video_height, 
                                                                                                         label, 
                                                                                                         print_coord, 
                                                                                                         print_font_size, 
                                                                                                         args.out_action_root)
    
        
        add_data_to_all(keypoints, keypoint_scores, label, bboxes)

# ----------------------------------------------------------------

        if framecount%1000==0:
            print(f"Processed {framecount + 1}/{length} frames.")
            print(label_count)
        
        # cv2.imshow("Test", drawn_frame)
            
        cv2.circle(drawn_frame, reference_point, 5, (0, 0, 255), -1)  # Red circle with radius of 5

        if args.save_out_video:
            videoWriter.write(drawn_frame)

        cv2.waitKey(1)
    else:
        break

# ----------------------------------------------------------------


vid.release()
cv2.destroyAllWindows()

if args.save_out_video:
    videoWriter.release()

# Convert lists to numpy arrays
keypoints_all = np.array(keypoints_all)
keypoint_scores_all = np.array(keypoint_scores_all)
labels_all = np.array(labels_all)
bboxes_all = np.array(bboxes_all)

# Set the nan keypoint scores to 0
keypoint_scores_all[np.isnan(keypoint_scores_all)] = 0

# Define the extreme value threshold
extreme_value_threshold = max(video_width, video_height)

# Identify frames with extreme keypoints values
extreme_keypoints_frames = np.any((keypoints_all > extreme_value_threshold) | (keypoints_all < 0) | (keypoints_all==np.nan), axis=(1, 2))

# Count the number of frames with extreme values
num_extreme_frames = np.sum(extreme_keypoints_frames)
print(f"Number of frames with extreme values: {num_extreme_frames}, Setting these to 0.")

# Set keypoints, keypoint scores, and bounding boxes to zero for frames with extreme values
keypoints_all[extreme_keypoints_frames] = 0
keypoint_scores_all[extreme_keypoints_frames] = 0
bboxes_all[extreme_keypoints_frames] = 0

total_frames = keypoints_all.shape[0]
# Reshape to num_frames, num_people, keypoints, coordinates format
keypoints_all = keypoints_all.reshape(keypoints_all.shape[0], 1, 17, 2) 

# Reshape all to num_people, num_frames, values format
keypoints_all = keypoints_all.swapaxes(0,1)
keypoint_scores_all = keypoint_scores_all.swapaxes(0,1)
bboxes_all = bboxes_all.swapaxes(0,1)

# Organize data into a dictionary
data = {
    "frame_dir": os.path.basename(args.video_file),
    "keypoint": keypoints_all,
    "keypoint_score": keypoint_scores_all,
    "labels": labels_all,
    "img_shape": (video_height, video_width),
    "width": video_width,
    "height": video_height,
    "bbox": bboxes_all,
    "total_frames":total_frames,
    "bbox_format": "xywh"
}

print(data.keys())

print(f"Shape of the keypoints: {keypoints_all.shape}")
print(f"Shape of the keypoint_scores: {keypoint_scores_all.shape}")
print(f"Shape of the bboxes: {bboxes_all.shape}")

existing_data = [data]

# Save updated data to pickle file
save_data_to_pickle(pickle_file_path, existing_data)

print("End")
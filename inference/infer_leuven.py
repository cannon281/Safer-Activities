import argparse
import torch
from tools.process_video import infer_leuven_dataset
from tools.model_setup import load_model_with_transforms
from tools.src.vitpose_infer.main import VitInference
from tools.utils import load_mappings_and_initialize_variables
from utils import ConfigParser
import os
import json
import pandas as pd
from pathlib import Path

# Config
# --------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Infer Action Classification')
parser.add_argument('--config_path', type=str, default="configs/CNN1D_kp.py")
parser.add_argument('--weight_path', type=str, default="weights/CNN1D_kp.pt")
parser.add_argument('--save_model_name', type=str, default="cnn1d_leuven")
parser.add_argument('--label_from', type=str, default="center")
args = parser.parse_args()

label_from = args.label_from
config_path = args.config_path
weight_path = args.weight_path
save_model_name = args.save_model_name

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

mapping_labels, _, label_count, actionmap, _, _ = load_mappings_and_initialize_variables(cfg_parser.dataset_cfg["mappings_json_file"], mode = "normal")
labels = list(mapping_labels[1:]) # Remove the no_label, which is the first one in the list
num_classes = len(labels)
# --------------------------------------------------------------------------------------------------------------------------------------------

save_out_video = True
out_video_root = "./out_video/"
os.makedirs(out_video_root, exist_ok=True)

pose_path = 'det_pose_models/vitpose-b-multi-coco.pth'
det_path = 'det_pose_models/yolov8x.pt'
det_pose_model = VitInference(pose_path=pose_path, det_type="yolov8", det_path=det_path, 
                              tensorrt=False, tracking_method="bytetrack")


# Action classification model with transforms
# --------------------------------------------------------------------------------------------------------------------------------------------
model, transforms = load_model_with_transforms(cfg_parser, num_classes, weight_path, device)
# --------------------------------------------------------------------------------------------------------------------------------------------

video_dir = "data/Leuven/Videos"
target_csv_dir = f"./out_leuven_dir/{save_model_name}_label_{label_from}/predictions"
os.makedirs(target_csv_dir, exist_ok=True)
out_csv_suffix = "_pred_gt.csv"
anno_file = "data/Leuven/Data_Description.xlsx"

# Read the Excel file
df = pd.read_excel(anno_file)  # Assuming the first column is 'Scenario'
columns_of_interest = {df.columns[0]:'Scenario', df.columns[8]: 'Start', df.columns[9]: 'Fall', df.columns[10]: 'End'}  # Adjust these indices if necessary
filtered_df = df[list(columns_of_interest.keys())].rename(columns=columns_of_interest)

# Iterate over each video file in the video directory
for f in os.listdir(video_dir):
    filepath = os.path.join(video_dir, f)
    anno_basename = Path(filepath).stem
    
    # Extract scenario number from the video file name
    scenario_number = int(anno_basename.split('_')[0].replace('Fall', ''))
    
    # Fetch the start and end times using the scenario number
    if scenario_number in filtered_df['Scenario'].values:
        matching_row = filtered_df[filtered_df['Scenario'] == scenario_number]
        start_time = matching_row['Start'].iloc[0]
        fall_time = matching_row['Fall'].iloc[0]
        end_time = matching_row['End'].iloc[0]
        start_end = [start_time, end_time]

    prediction_results, max_actions_one_frame = infer_leuven_dataset(
        filepath, start_end, det_pose_model, model, transforms, device,
        mapping_labels=mapping_labels, actionmap=actionmap, 
        mode="1080p", show_video=True, save_out_video=True, 
        out_video_root=f"./out_leuven_dir/{save_model_name}_label_{label_from}/videos/", fall_threshold=0.3, label_from=label_from)

    # Ensure column names match the data structure
    if prediction_results:
        if max_actions_one_frame>0:
            column_names = ['Predicted', 'GT', 'Timestamp', 'Num_detections'] + [f'Action_{i}' for i in range(1, max_actions_one_frame)]
        else:
            column_names = ['Predicted', 'GT', 'Timestamp', 'Num_detections']
        predictions_df = pd.DataFrame(prediction_results, columns=column_names)
    
        # Save the data with modified naming to include the folder
        out_csv_path = os.path.join(target_csv_dir, f"{anno_basename}{out_csv_suffix}")
        predictions_df.to_csv(out_csv_path, index=False)
        print("Saved to ", out_csv_path)
    else:
        print("No prediction results for", filepath)



import argparse
from pathlib import Path
import torch
from tools.process_video import infer_leuven_dataset, process_one_video
from tools.model_setup import load_model_with_transforms
from tools.src.vitpose_infer.main import VitInference
import os
import pandas as pd

import sys
pyskl_path = "/home/work/inference/pyskl"
if pyskl_path not in sys.path:
    sys.path.append(pyskl_path)
import mmcv
from pyskl.apis import init_recognizer

# Config
# --------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Infer Action Classification')
parser.add_argument('--config_path', type=str, default="pyskl/configs/msg3d/safer_activity_xsub/non-wheelchair.py")
parser.add_argument('--weight_path', type=str, default="pyskl/weights/msg3d/non-wheelchair/non-wheelchair-epoch_16.pth")
parser.add_argument('--save_model_name', type=str, default="msg3d_leuven")
parser.add_argument('--label_from', type=str, default="center")
args = parser.parse_args()

config_path = args.config_path
weight_path = args.weight_path
label_from = args.label_from
save_model_name = args.save_model_name

assert os.path.exists(config_path), f"Config file {config_path} does not exist"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



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
config = mmcv.Config.fromfile(config_path)
config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
model = init_recognizer(config, weight_path, device)
# --------------------------------------------------------------------------------------------------------------------------------------------


label_from = "center"
# Action classification model with transforms

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
        filepath, start_end, det_pose_model, model, None, device,
        None, None, infer_pyskl=True,
        mode="1080p", show_video=False, save_out_video=True, 
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



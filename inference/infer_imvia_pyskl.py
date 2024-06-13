import argparse
from pathlib import Path
import torch
from tools.process_video import infer_imvia_dataset, infer_leuven_dataset, process_one_video
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
parser.add_argument('--config_path', type=str, default="pyskl/configs/posec3d/safer_activity_xsub/non-wheelchair.py")
parser.add_argument('--weight_path', type=str, default="pyskl/weights/posec3d/non-wheelchair/non-wheelchair-epoch_44.pth")
parser.add_argument('--save_model_name', type=str, default="posec3d_imvia")
args = parser.parse_args()

config_path = args.config_path
weight_path = args.weight_path
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
# --------------------------------------------------------------------------------------------------------------------------------------------

target_folder = f"./out_imvia_dir/{save_model_name}_conf_0.2_nms_0.7_match_10_label_{label_from}"
os.makedirs(target_folder, exist_ok=True)
base_dataset_dir = "data/ImViA"
target_csv_dir = os.path.join(target_folder,"predictions")
os.makedirs(target_csv_dir, exist_ok=True)
out_csv_suffix = "_pred_gt.csv"

# Iterate over each subdirectory in the base directory
for sub_dir in os.listdir(base_dataset_dir):
    video_dir = os.path.join(base_dataset_dir, sub_dir, "Videos")
    anno_dir = os.path.join(base_dataset_dir, sub_dir, "Annotation_files")

    # Iterate over each video file in the video directory
    for f in os.listdir(video_dir):
        filepath = os.path.join(video_dir, f)
        anno_basename = Path(filepath).stem
        anno_file = os.path.join(anno_dir, anno_basename + ".txt")

        prediction_results, max_actions_one_frame = infer_imvia_dataset(
            filepath, anno_file, det_pose_model, model, None, device,
            mapping_labels=None, actionmap=None, 
            mode="1080p", show_video=False, save_out_video=True, 
            out_video_root=os.path.join(target_folder,f"videos/{sub_dir}"), fall_threshold=0.3, label_from=label_from, infer_pyskl=True)

        # Ensure column names match the data structure
        if prediction_results:
            if max_actions_one_frame>0:
                column_names = ['Predicted', 'GT', 'Frame', 'Num_detections'] + [f'Action_{i}' for i in range(1, max_actions_one_frame)]
            else:
                column_names = ['Predicted', 'GT', 'Frame', 'Num_detections']
            predictions_df = pd.DataFrame(prediction_results, columns=column_names)
        
            # Save the data with modified naming to include the folder
            out_csv_path = os.path.join(target_csv_dir, f"{sub_dir}_{anno_basename}{out_csv_suffix}")
            predictions_df.to_csv(out_csv_path, index=False)
            print("Saved to ", out_csv_path)
        else:
            print("No prediction results for", filepath)



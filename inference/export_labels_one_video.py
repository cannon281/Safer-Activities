import argparse
import torch
from tools.process_video import infer_export_one_video
from tools.model_setup import load_model_with_transforms
from tools.src.vitpose_infer.main import VitInference
from tools.utils import load_mappings_and_initialize_variables
from utils import ConfigParser
import os
import pandas as pd

# Config
# --------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Infer Action Classification')
parser.add_argument('--config_path', type=str, default="configs/CNN1D_kp.py")
args = parser.parse_args()

config_path = args.config_path

assert os.path.exists(config_path), f"Config file {config_path} does not exist"
try:
    cfg_parser = ConfigParser(config_path)
except:
    raise Exception(f"Error parsing config file {config_path}. Are you sure it is a valid config file?")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

mapping_labels, _, label_count, actionmap, _, _ = load_mappings_and_initialize_variables(cfg_parser.dataset_cfg["mappings_json_file"], mode = "normal")
labels = list(mapping_labels[1:]) # Remove the no_label, which is the first one in the list
num_classes = len(labels)
# --------------------------------------------------------------------------------------------------------------------------------------------

save_out_video = True
out_video_root = "./out_video/"
os.makedirs(out_video_root, exist_ok=True)
video_path = "videos/aic_fall.mp4"

pose_path = 'det_pose_models/vitpose-b-multi-coco.pth'
det_path = 'det_pose_models/yolov8x.pt'
det_pose_model = VitInference(pose_path=pose_path, det_type="yolov8", det_path=det_path, 
                              tensorrt=False, tracking_method="bytetrack")


# Action classification model with transforms
# --------------------------------------------------------------------------------------------------------------------------------------------
classifiaction_weight_path= "weights/_KeypointCNN1D_epoch_25.pt" # End Label
# classifiaction_weight_path= "weights/CNN1D_kp.pt" # Center Llabel

model, transforms = load_model_with_transforms(cfg_parser, num_classes, classifiaction_weight_path, device)
# --------------------------------------------------------------------------------------------------------------------------------------------

video_path = "data/aicactivity/normal/Videos/apr_25_2023_fall_p005_d01.mp4"
spreadsheet_path = "data/aicactivity/normal/CSVs/apr_25_2023_fall_p005.csv"
prediction_results, max_actions_one_frame = infer_export_one_video(video_path, spreadsheet_path, det_pose_model, model, transforms, device,
                                            mapping_labels=mapping_labels, actionmap=actionmap, 
                                            mode="1080p", show_video=False, save_out_video=True, 
                                            out_video_root="./out_video/", fall_threshold=0.5, label_from = "end")

target_csv_dir = "./out_csv"
os.makedirs(target_csv_dir, exist_ok=True)
out_csv_suffix = "_pred_gt.csv"

column_names = ['Predicted', 'GT', 'Timestamp', 'Num_detections'] + [f'Action_{i}' for i in range(1, max_actions_one_frame)]
predictions_df = pd.DataFrame(prediction_results, columns=column_names)

# Save the data:
out_csv_path = os.path.join(target_csv_dir, os.path.basename(video_path).split(".")[0] + out_csv_suffix)
predictions_df.to_csv(out_csv_path, index=False)
print("Saved to ", out_csv_path)

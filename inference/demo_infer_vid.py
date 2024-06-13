import argparse
import torch
from tools.process_video import process_one_video
from tools.model_setup import load_model_with_transforms
from tools.src.vitpose_infer.main import VitInference
from utils import ConfigParser
import os
import json
import os

# Config
# --------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Infer Action Classification')
parser.add_argument('--config_path', type=str, default="configs/CNN1D_kp.py")
parser.add_argument('--weight_path', type=str, default="weights/CNN1D_kp.pt")
parser.add_argument('--video_path', type=str, default="videos/new_fall.mp4")
parser.add_argument('--label_from', type=str, default="center")

args = parser.parse_args()
filepath = args.video_path
config_path = args.config_path
weight_path = args.weight_path
label_from = args.label_from

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

process_one_video(filepath, det_pose_model, model, transforms, device, mode="1080p", show_video=True, save_out_video=True, 
                    out_video_root="./out_video/", fall_threshold=0.3, label_from=label_from, viz_kpt=False)

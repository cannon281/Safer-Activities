import argparse
import torch
from tools.process_video import process_one_video
from tools.model_setup import load_model_with_transforms
from tools.src.vitpose_infer.main import VitInference
from utils import ConfigParser
import os
import json
import os

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
parser.add_argument('--label_type', type=str, default="normal")
parser.add_argument('--label_from', type=str, default="center")
parser.add_argument('--filepath', type=str, default="videos/Fall4_Cam3.avi")
args = parser.parse_args()

config_path = args.config_path
weight_path = args.weight_path
label_type = args.label_type
label_from = args.label_from
filepath = args.filepath

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

process_one_video(filepath, det_pose_model, model, None, device, mode="1080p", show_video=True, save_out_video=True, 
                    out_video_root="./out_video/", fall_threshold=0.3, label_from=label_from, viz_kpt=False, infer_pyskl=True, label_type=label_type)

import argparse
import json
import logging
import torch
import torch.nn as nn
from utils import ConfigParser, get_dataset_with_transforms, get_dataloader
from utils.train_utils import get_classification_accuracy, get_segmentation_accuracy, print_and_log
from utils.test_utils import save_confusion_matrix_and_classification_report
import time
import os
import numpy as np

# example usage: python test.py --config_path configs/CNN1D_kp.py --checkpoint_path work_dirs/CNN1D_kp/20240227-013900/_KeypointCNN1D_epoch_40.pt --visible_gpus 0

# --------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Test Action Classification Model')
parser.add_argument('--config_path', type=str, default="configs/CNN1D_kp.py")
parser.add_argument('--visible_gpus', type=str, default="0", required=False)
parser.add_argument('--checkpoint_path', type=str, required=True)
args = parser.parse_args()


# --------------------------------------------------------------------------------------------------------------------------------------------

config_path = args.config_path
visible_gpus = args.visible_gpus
os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

assert os.path.exists(config_path), f"Config file {config_path} does not exist"
try:
    cfg_parser = ConfigParser(config_path)
except:
    raise Exception(f"Error parsing config file {config_path}. Are you sure it is a valid config file?")


# For logging
# --------------------------------------------------------------------------------------------------------------------------------------------
save_logs = cfg_parser.log_cfg['save_logs']
save_ckpt = cfg_parser.train_cfg['save_ckpt']

timestr = time.strftime("%Y%m%d-%H%M%S")
log_base_dir = os.path.join(cfg_parser.log_cfg['log_dir'], 'test_logs')
model_name = cfg_parser.model_cfg['type']
config_name = os.path.splitext(os.path.basename(config_path))[0]

log_instance_dir = os.path.join(log_base_dir,  config_name, f'test_logs_{timestr}')

if save_logs or save_ckpt:
    os.makedirs(log_instance_dir, exist_ok=True)

if save_logs:
    log_filename=f'test_logs.log'
    log_filename = os.path.join(log_instance_dir, log_filename)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Logging to {log_filename}")

print_and_log(f"Loaded config from {config_path}", save_logs)
print_and_log(f"Model: {model_name}", save_logs)
print_and_log(f"Dataset: {cfg_parser.dataset_cfg['dataset_class']}", save_logs)
print_and_log(f"Checkpoint: {args.checkpoint_path}", save_logs)
print_and_log(f"#################################################################################", save_logs)
# --------------------------------------------------------------------------------------------------------------------------------------------


# Device settings
# --------------------------------------------------------------------------------------------------------------------------------------------
multi_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    multi_gpu = cfg_parser.train_cfg['multi_gpu'] and torch.cuda.device_count() > 1
else:
    device = torch.device('cpu')
    multi_gpu = False

print_and_log(f"Using device: {visible_gpus}, multi_gpu: {multi_gpu}", save_logs)
# --------------------------------------------------------------------------------------------------------------------------------------------


# Dataset labels
# --------------------------------------------------------------------------------------------------------------------------------------------
f = open(cfg_parser.dataset_cfg["mappings_json_file"], "r")
mappings = json.loads(f.read())[cfg_parser.dataset_cfg["dataset_type"]]

label_dict = mappings["labels"]
labels = list(label_dict.values())[1:] # Remove the no_label, which is the first one in the list
num_classes = len(labels)

try:
    train_mode = cfg_parser.dataset_cfg['args']['target_type'] # classification or segmentation
except:
    print("Target type not found, using classification as default.")
    train_mode = "classification"
# --------------------------------------------------------------------------------------------------------------------------------------------


# Model settings
# --------------------------------------------------------------------------------------------------------------------------------------------
model_class = cfg_parser.get_model_class_from_config()

model = model_class(num_classes = num_classes, **cfg_parser.model_cfg['args'])
if multi_gpu:
    model = nn.DataParallel(model)
model = model.to(device)

# load model from checkpoint
checkpoint = torch.load(args.checkpoint_path)
try:
    model.load_state_dict(checkpoint)
except:
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
# --------------------------------------------------------------------------------------------------------------------------------------------


# Dataset settings
# --------------------------------------------------------------------------------------------------------------------------------------------
dataset_class = cfg_parser.get_test_dataset_class_from_config()
test_dataset = dataset_class(**cfg_parser.dataset_cfg['args'], split_type = cfg_parser.dataset_cfg["splits"]["test"])
test_dataset = get_dataset_with_transforms(test_dataset, cfg_parser.get_test_transforms())
test_loader = get_dataloader(test_dataset, cfg_parser.get_test_dataloader_settings())
# --------------------------------------------------------------------------------------------------------------------------------------------

if train_mode == "classification":
    accuracy, predicted_labels, true_labels = get_classification_accuracy(test_loader, model, device=next(model.parameters()).device, return_preds=True)
else:
    accuracy, predicted_labels, true_labels = get_segmentation_accuracy(test_loader, model, device=next(model.parameters()).device, return_preds=True)
    

print_and_log(f"Test Accuracy: {accuracy}%", save_logs)
print_and_log(f"#################################################################################", save_logs)

cm_save_path = os.path.join(log_instance_dir,"confusion_matrix_normalized.png")
report_save_path = os.path.join(log_instance_dir,"classification_report.txt")
cm_csv_path = os.path.join(log_instance_dir,"confusion_matrix.csv")

report = save_confusion_matrix_and_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, 
                                                         labels=labels, cm_save_path=cm_save_path, report_save_path=report_save_path,
                                                         cm_csv_path = cm_csv_path)


print_and_log(f"Confusion matrix and classification report saved to {cm_save_path} and {report_save_path}, respecitvely.:", save_logs)


# --------------------------------------------------------------------------------------------------------------------------------------------

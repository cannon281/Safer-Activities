import argparse
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
from utils import ConfigParser, get_dataset_with_transforms, get_dataloader
from utils.train_utils import train_and_validate, print_and_log
import time
import os
import random
import json
import sys


# example usage: python train.py --config_path configs/CNN1D_kp.py --visible_gpus 0,1,2,3

# --------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train Action Classification Model')
parser.add_argument('--config_path', type=str, default="configs/CNN1D_kp.py")
parser.add_argument('--visible_gpus', type=str, default="0", required=False)
args = parser.parse_args()

config_path = args.config_path
visible_gpus = args.visible_gpus
os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

assert os.path.exists(config_path), f"Config file {config_path} does not exist"
try:
    cfg_parser = ConfigParser(config_path)
except:
    raise Exception(f"Error parsing config file {config_path}. Are you sure it is a valid config file?")

save_ckpt = cfg_parser.train_cfg['save_ckpt']


# For logging
# --------------------------------------------------------------------------------------------------------------------------------------------
save_logs = cfg_parser.log_cfg['save_logs']
timestr = time.strftime("%Y%m%d-%H%M%S")
log_base_dir = cfg_parser.log_cfg['log_dir']
model_name = cfg_parser.model_cfg['type']
config_name = os.path.splitext(os.path.basename(config_path))[0]

log_instance_dir = os.path.join(log_base_dir, config_name, timestr)

if save_logs or save_ckpt:
    os.makedirs(log_instance_dir, exist_ok=True)

# Copy config file to log directory
shutil.copy(config_path, os.path.join(log_instance_dir, os.path.basename(config_path)))

if save_logs:
    log_filename='train_logs.log'
    log_filename = os.path.join(log_instance_dir, log_filename)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Logging to {log_filename}")

print_and_log(f"Loaded config from {config_path}", save_logs)
print_and_log(f"Model: {model_name}", save_logs)
print_and_log(f"Dataset: {cfg_parser.dataset_cfg['dataset_class']}", save_logs)
print_and_log(f"Skip Frames: {cfg_parser.dataset_cfg['args']['skip_frames']}", save_logs)
print_and_log(f"#################################################################################", save_logs)
# --------------------------------------------------------------------------------------------------------------------------------------------


# Seed to replicate training
# --------------------------------------------------------------------------------------------------------------------------------------------
def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything(cfg_parser.dataset_cfg["seed"])
# --------------------------------------------------------------------------------------------------------------------------------------------

# Train settings
# --------------------------------------------------------------------------------------------------------------------------------------------
NUM_EPOCHS = cfg_parser.train_cfg['epochs']
N = cfg_parser.train_cfg['print_every']  # Report accuracy every N epochs
SAVE_EVERY = cfg_parser.train_cfg['save_ckpt_every']  # Save model every SAVE_EVERY epochs
try:
    train_mode = cfg_parser.dataset_cfg['args']['target_type'] # classification or segmentation
except:
    print("Target type not found, using classification as default.")
    train_mode = "classification"
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
# --------------------------------------------------------------------------------------------------------------------------------------------


# Model settings
# --------------------------------------------------------------------------------------------------------------------------------------------
model_class = cfg_parser.get_model_class_from_config()

model = model_class(num_classes = num_classes, **cfg_parser.model_cfg['args'])
if multi_gpu:
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = cfg_parser.get_optimizer_from_config(model)
criterion = cfg_parser.get_loss_class_from_config()(ignore_index=-1) # If index is -1, ignore. Meant for the segmentation training
# --------------------------------------------------------------------------------------------------------------------------------------------

# Dataset settings
# --------------------------------------------------------------------------------------------------------------------------------------------
dataset_class = cfg_parser.get_dataset_class_from_config()

train_dataset = dataset_class(**cfg_parser.dataset_cfg['args'], split_type = cfg_parser.dataset_cfg["splits"]["train"])
test_dataset = dataset_class(**cfg_parser.dataset_cfg['args'], split_type = cfg_parser.dataset_cfg["splits"]["test"])


train_dataset = get_dataset_with_transforms(train_dataset, cfg_parser.get_train_transforms())
val_dataset = get_dataset_with_transforms(test_dataset, cfg_parser.get_val_transforms())

train_loader = get_dataloader(train_dataset, cfg_parser.get_train_dataloader_settings())
train_loader.worker_init_fn=worker_init_fn  # for seeding

val_loader = get_dataloader(val_dataset, cfg_parser.get_val_dataloader_settings())
# --------------------------------------------------------------------------------------------------------------------------------------------

# Training
# --------------------------------------------------------------------------------------------------------------------------------------------
train_losses, train_accuracies, val_accuracies = train_and_validate(NUM_EPOCHS, 
                                                                    model, 
                                                                    train_loader, 
                                                                    val_loader, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    print_every=N,
                                                                    save_every=SAVE_EVERY,
                                                                    save_logs = save_logs,
                                                                    log_filename=log_filename,
                                                                    save_ckpt= save_ckpt,
                                                                    ckpt_path = os.path.join(log_instance_dir, f"_{model_name}.pt"),
                                                                    mode=train_mode
                                                                )
# --------------------------------------------------------------------------------------------------------------------------------------------


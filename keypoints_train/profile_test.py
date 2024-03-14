import torch
import torch.nn as nn
from utils import ConfigParser
from torch.profiler import profile, record_function, ProfilerActivity

cfg_parser = ConfigParser("configs/Custom_Attention_Transformer_kp.py")
model_class = cfg_parser.get_model_class_from_config()
dataset_class = cfg_parser.get_dataset_class_from_config()

model_settings = {'d_model': 4, 'nhead': 2, 'num_layers': 2, 'num_classes': 14}
model = model_class(**model_settings).cuda()
inputs = torch.randn(128, 28, 34).cuda()

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
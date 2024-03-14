import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis


class KeypointCNN1D(nn.Module):
    def __init__(self, num_classes, num_frames, motion_info=False):
        super(KeypointCNN1D, self).__init__()
        
        num_in_channels = 34*2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * num_frames, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage:
num_classes = 15 # Example value, change as needed
num_frames = 48 # Example value, change as needed
model = KeypointCNN1D(num_classes=num_classes, num_frames=num_frames, motion_info=True)
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")

# Dummy input to trigger the hooks and compute FLOPs
input = torch.randn(1, 68, 48) # Example input; adjust the size accordingly
output = model(input)

flops = FlopCountAnalysis(model, input)
print(flops.total())
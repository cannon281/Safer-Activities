import torch.nn as nn
import torch.nn.functional as F

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


class KeypointCNN1DTiny(nn.Module):
    def __init__(self, num_classes, num_frames, motion_info=False):
        super(KeypointCNN1DTiny, self).__init__()
        
        num_in_channels = 34*2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * num_frames, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class KeypointCNN1DSegment(nn.Module):
    def __init__(self, num_classes, num_frames, motion_info=False):
        super(KeypointCNN1DSegment, self).__init__()
        
        num_in_channels = 34*2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # No activation function here, assuming you'll use CrossEntropyLoss which includes Softmax
        return x


class EnhancedCNN1D(nn.Module):
    def __init__(self, num_classes, num_frames):
        super(EnhancedCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(34, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256 * num_frames, 1024)  # Adjusted for increased feature map depth
        self.dropout1 = nn.Dropout(0.5)  # Dropout for regularization before the first FC layer
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

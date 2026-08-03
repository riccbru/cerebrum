import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc1(x))
        return self.fc2(x)

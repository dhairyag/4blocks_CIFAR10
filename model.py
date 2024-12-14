import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  padding=padding, groups=in_channels, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define all channel sizes as class variables
        self.input_channels = 3
        self.layer1_channels = 8
        self.layer2_channels = 16
        self.layer3_channels = 32
        self.layer4_channels = 48
        self.layer5_channels = 64
        self.num_classes = 10
        
        # Receptive Field (RF) is shown in the comments
        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, self.layer1_channels, kernel_size=3, padding=1), # RF=3   
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer1_channels, self.layer2_channels, kernel_size=3, padding=1), # RF=5
            nn.BatchNorm2d(self.layer2_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer2_channels, self.layer2_channels, kernel_size=3, padding=1), # RF=7
            nn.BatchNorm2d(self.layer2_channels),
            nn.ReLU(),
        )
        # Conv Block 2 (with Depthwise Separable Conv)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(self.layer2_channels, self.layer3_channels, stride=2), # RF=9
            nn.BatchNorm2d(self.layer3_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer3_channels, self.layer3_channels, kernel_size=3, padding=1), # RF=14
            nn.BatchNorm2d(self.layer3_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer3_channels, self.layer3_channels, kernel_size=3, padding=1), # RF=13
            nn.BatchNorm2d(self.layer3_channels),
            nn.ReLU(),
        )        
        # Conv Block 3 (with Dilated Conv)
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer3_channels, self.layer4_channels, kernel_size=3, padding=2, dilation=2), # RF=21
            nn.BatchNorm2d(self.layer4_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer4_channels, self.layer4_channels, kernel_size=3, padding=1), # RF=29
            nn.BatchNorm2d(self.layer4_channels),
            nn.ReLU(),
        )

        # Conv Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer4_channels, self.layer5_channels, kernel_size=3, padding=1, stride=2), # RF=37
            nn.BatchNorm2d(self.layer5_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer5_channels, self.layer5_channels, kernel_size=3, padding=1), # RF=45 
            nn.BatchNorm2d(self.layer5_channels),
            nn.ReLU(),
            nn.Conv2d(self.layer5_channels, self.layer5_channels, kernel_size=1),  # RF=45
        )
        # Output Block
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.layer5_channels, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, self.layer5_channels)
        x = self.fc(x)
        return x 
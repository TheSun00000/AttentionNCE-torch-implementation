import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, 1)
        out = self.projection_head(feature)
        return F.normalize(out, dim=-1)


""" Another version of the model that, instead of implementing ResNet from scratch, 
utilizes the Torchvision implementation and then modifies it. Both implementations are equivalent, 
with the Torchvision-based version being  faster to train,
 as Torchvision's ResNet is  optimized for performance. """


class Model_torchvision(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_torchvision, self).__init__()

        self.encoder = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Customize conv1
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.encoder.append(module)
        # Encoder
        self.encoder = nn.Sequential(*self.encoder)
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),  # ResNet18's output dimension is 512
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim, bias=True)
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder
        feature = torch.flatten(x, start_dim=1)  # Flatten features
        projections = self.projection_head(feature)  # Pass through the projection head
        projections=F.normalize(projections, dim=-1)
        return  projections
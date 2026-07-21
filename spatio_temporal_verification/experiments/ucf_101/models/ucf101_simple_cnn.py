import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class UCF101SimpleCNN(nn.Module):
    """
    A simple 3D CNN model for UCF-101 action recognition as defined in UCF-101 notebook.
    
    Architecture:
    - 3 convolutional layers with BatchNorm3d and MaxPool3d
    - Global average pooling
    - Single fully connected layer classifier with Dropout
    """
    def __init__(
        self, 
        num_frames: int = 16,
        in_channels: int = 3,
        img_x: int = 342, 
        img_y: int = 256,
        num_classes: int = 101,
        dropout_rate: float = 0.5
    ):
        super(UCF101SimpleCNN, self).__init__()
        
        self.num_frames = num_frames
        self.input_shape = (img_x, img_y)
        self.in_channels = in_channels
        self.img_x = img_x
        self.img_y = img_y
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.name = "ucf101_simple_cnn"
        
        # Conv layer 1
        self.conv1 = nn.Conv3d(self.in_channels, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(0, 0, 0))
        self.layer1 = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Conv layer 2
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.layer2 = nn.Sequential(
            self.conv2,
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Conv layer 3
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.layer3 = nn.Sequential(
            self.conv3,
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, channels, frames, height, width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> 'UCF101SimpleCNN':
        model = cls(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model

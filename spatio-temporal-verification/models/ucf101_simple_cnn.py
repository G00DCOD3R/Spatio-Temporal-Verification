import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class UCF101SimpleCNN(nn.Module):
    """
    A simple 3D CNN model for UCF-101 action recognition.
    
    Architecture:
    - Only 3 convolutional layers (separated)
    - Global average pooling to reduce parameters
    - Single fully connected layer for classification
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
        
        # Model configuration
        self.num_frames = num_frames
        self.input_shape = (img_y, img_x)  # Height, width format
        self.in_channels = in_channels
        self.img_x = img_x
        self.img_y = img_y
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.name = "ucf101_simple_cnn"
        
        # First layer - convolutional part only
        self.conv1 = nn.Conv3d(
            in_channels=self.in_channels, 
            out_channels=16, 
            kernel_size=(3, 5, 5), 
            stride=(1, 2, 2), 
            padding=(1, 2, 2)  # Added proper padding
        )
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Second layer - convolutional part only
        self.conv2 = nn.Conv3d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Modified to preserve time dimension
        
        # Third layer - convolutional part only
        self.conv3 = nn.Conv3d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3, 3), 
            stride=(1, 1, 1), 
            padding=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Calculate output size after all conv and pooling layers
        t, h, w = self._calculate_output_shape()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_output_shape(self):
        """Calculate the output shape after all conv and pooling layers"""
        # Start with input dimensions
        t, h, w = self.num_frames, self.img_y, self.img_x
        
        # Conv1: kernel=(3,5,5), stride=(1,2,2), padding=(1,2,2)
        t = self._conv_output_size(t, 3, 1, 1)
        h = self._conv_output_size(h, 5, 2, 2)
        w = self._conv_output_size(w, 5, 2, 2)
        
        # Pool1: kernel=(1,2,2), stride=(1,2,2)
        t = self._conv_output_size(t, 1, 1, 0)
        h = self._conv_output_size(h, 2, 2, 0)
        w = self._conv_output_size(w, 2, 2, 0)
        
        # Conv2: kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)
        t = self._conv_output_size(t, 3, 1, 1)
        h = self._conv_output_size(h, 3, 1, 1)
        w = self._conv_output_size(w, 3, 1, 1)
        
        # Pool2: kernel=(1,2,2), stride=(1,2,2)
        t = self._conv_output_size(t, 1, 1, 0)
        h = self._conv_output_size(h, 2, 2, 0)
        w = self._conv_output_size(w, 2, 2, 0)
        
        # Conv3: kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)
        t = self._conv_output_size(t, 3, 1, 1)
        h = self._conv_output_size(h, 3, 1, 1)
        w = self._conv_output_size(w, 3, 1, 1)
        
        # Pool3: kernel=(2,2,2), stride=(2,2,2)
        t = self._conv_output_size(t, 2, 2, 0)
        h = self._conv_output_size(h, 2, 2, 0)
        w = self._conv_output_size(w, 2, 2, 0)
        
        return t, h, w
    
    def _conv_output_size(self, input_size, kernel_size, stride, padding, dilation=1):
        """Calculate output size for a dimension after convolution/pooling"""
        return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    
    def forward(self, x):
        # Input shape: (batch_size, channels, frames, height, width)
        
        # Layer 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Layer 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Layer 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
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
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Load a pretrained model from checkpoint"""
        model = cls(**kwargs)
        device = kwargs.get('device', None)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if device:
            model = model.to(device)
        return model 
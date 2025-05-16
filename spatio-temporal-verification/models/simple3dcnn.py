import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class Simple3DCNN(nn.Module):
    """
    Base implementation of a simple 3D CNN model for video classification.
    
    This model consists of three convolutional layers followed by batch normalization
    and ReLU activations, and two fully connected layers.
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        conv_channels: List[int] = [2, 4, 8],
        kernel_sizes: List[int] = [2, 2, 2],
        paddings: List[int] = [0, 1, 1],
        fc_hidden: int = 128,
        num_frames: int = 10, 
        input_shape: Tuple[int, int] = (28, 28), 
        num_classes: int = 10, 
        dropout_rate: float = 0.5
    ) -> None:
        """
        Initialize the Simple3DCNN model.
        
        Args:
            in_channels: Number of input channels
            conv_channels: List of output channels for each conv layer
            kernel_sizes: List of kernel sizes for conv layers
            paddings: List of padding values for conv layers
            fc_hidden: Number of hidden units in the fully connected layer
            num_frames: Number of frames in each video sequence
            input_shape: Tuple of (height, width) for each frame
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(Simple3DCNN, self).__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Define individual convolutional layers
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=conv_channels[0], 
                              kernel_size=kernel_sizes[0], stride=1, padding=paddings[0])
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(conv_channels[0])
        
        self.conv2 = nn.Conv3d(in_channels=conv_channels[0], out_channels=conv_channels[1], 
                              kernel_size=kernel_sizes[1], stride=1, padding=paddings[1])
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(conv_channels[1])
        
        self.conv3 = nn.Conv3d(in_channels=conv_channels[1], out_channels=conv_channels[2], 
                              kernel_size=kernel_sizes[2], stride=1, padding=paddings[2])
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm3d(conv_channels[2])
        
        # Calculate flattened size mathematically instead of using a dummy tensor
        t, h, w = self.num_frames, self.input_shape[0], self.input_shape[1]
        
        # Apply formula for each conv layer: output_size = floor((input_size + 2*padding - kernel_size) / stride + 1)
        # Conv1
        t = self._conv_output_size(t, kernel_sizes[0], 1, paddings[0])
        h = self._conv_output_size(h, kernel_sizes[0], 1, paddings[0])
        w = self._conv_output_size(w, kernel_sizes[0], 1, paddings[0])
        
        # Conv2
        t = self._conv_output_size(t, kernel_sizes[1], 1, paddings[1])
        h = self._conv_output_size(h, kernel_sizes[1], 1, paddings[1])
        w = self._conv_output_size(w, kernel_sizes[1], 1, paddings[1])
        
        # Conv3
        t = self._conv_output_size(t, kernel_sizes[2], 1, paddings[2])
        h = self._conv_output_size(h, kernel_sizes[2], 1, paddings[2])
        w = self._conv_output_size(w, kernel_sizes[2], 1, paddings[2])
        
        # Calculate final flattened size
        self.flattened_size = conv_channels[2] * t * h * w
        # print(f"Calculated flattened size: {self.flattened_size}")
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, fc_hidden)
        self.relu_fc = nn.ReLU()
        self.bn_fc = nn.BatchNorm1d(fc_hidden)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(fc_hidden, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _conv_output_size(self, input_size, kernel_size, stride, padding, dilation=1):
        """Calculate output size for a dimension after convolution"""
        return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    
    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers"""
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire network"""
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Forward through fully connected layers
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.bn_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> 'Simple3DCNN':
        """Load a pretrained model from checkpoint"""
        model = cls(**kwargs)
        device = kwargs.get('device', None)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if device:
            model = model.to(device)
        return model 
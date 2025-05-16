import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

class Verifiable3DCNN(nn.Module):
    """
    3D CNN model designed aimed at scaling well with IBP verification.
    
    - Aggressive downsampling with strided convolutions and pooling
    - Global average pooling instead of flattening
    - Lightweight design with fewer parameters
    - Batch normalization for stability
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 8,
        spatial_reduction: int = 4,
        temporal_reduction: int = 2,
        num_classes: int = 200,
        num_frames: int = 30,
        input_shape: Tuple[int, int] = (256, 342),
        dropout_rate: float = 0.3
    ) -> None:
        """
        Initialize the IBPEfficientCNN model.
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            base_channels: Base number of channels (will be doubled in deeper layers)
            spatial_reduction: Factor to reduce spatial dimensions in first conv (higher = more aggressive)
            temporal_reduction: Factor to reduce temporal dimension in first conv
            num_classes: Number of output classes (default: 200 for TinyImageNet)
            num_frames: Number of frames in the input video
            input_shape: Input dimensions (H, W) before batch and channel dimensions
            dropout_rate: Dropout probability
        """
        super(Verifiable3DCNN, self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.dropout_rate = dropout_rate
        
        # Calculate initial stride to immediately reduce spatial dimensions
        # This is crucial for memory efficiency with large inputs
        t_stride = temporal_reduction
        hw_stride = spatial_reduction
        
        # Feature extraction block 1: aggressive downsampling
        self.block1 = nn.Sequential(
            # Initial strided convolution for aggressive downsampling
            nn.Conv3d(in_channels, base_channels, kernel_size=3, 
                     stride=(t_stride, hw_stride, hw_stride), padding=0),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Feature extraction block 2
        self.block2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Feature extraction block 3
        self.block3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Global pooling (eliminates need for huge FC layer)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels*4, num_classes)
        )
        
        # Calculate output dimensions for verification
        self._calculate_output_dimensions()
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_output_dimensions(self):
        """Calculate and print the output dimensions after each block for debugging"""
        t, h, w = self.num_frames, self.input_shape[0], self.input_shape[1]
        c = self.in_channels
        
        # Track dimensions through the network (useful for debugging)
        # Block 1: Strided conv + MaxPool
        t = math.floor((t + 2*1 - 3) / self.block1[0].stride[0] + 1)
        h = math.floor((h + 2*1 - 3) / self.block1[0].stride[1] + 1) 
        w = math.floor((w + 2*1 - 3) / self.block1[0].stride[2] + 1)
        c = self.block1[0].out_channels
        
        # MaxPool
        t = math.floor((t - 2) / 2 + 1)
        h = math.floor((h - 2) / 2 + 1)
        w = math.floor((w - 2) / 2 + 1)
        
        print(f"After block1: ({c}, {t}, {h}, {w})")
        
        # Block 2: Conv + MaxPool
        t = math.floor((t - 2) / 2 + 1)
        h = math.floor((h - 2) / 2 + 1)
        w = math.floor((w - 2) / 2 + 1)
        c = self.block2[0].out_channels
        
        print(f"After block2: ({c}, {t}, {h}, {w})")
        
        # Block 3: Conv + MaxPool
        t = math.floor((t - 2) / 2 + 1)
        h = math.floor((h - 2) / 2 + 1)
        w = math.floor((w - 2) / 2 + 1)
        c = self.block3[0].out_channels
        
        print(f"After block3: ({c}, {t}, {h}, {w})")
        
        # Global pooling will produce (c, 1, 1, 1)
        print(f"After global pooling: ({c}, 1, 1, 1)")
        print(f"Final features: {c}")
        
        # Store for potential verification use
        self.final_feature_dim = c
    
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global pooling (B, C, 1, 1, 1)
        x = self.global_pool(x)
        
        # Flatten to (B, C)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
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
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> 'Verifiable3DCNN':
        """Load a pretrained model from checkpoint"""
        model = cls(**kwargs)
        device = kwargs.get('device', None)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if device:
            model = model.to(device)
        return model 
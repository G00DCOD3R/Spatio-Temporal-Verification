import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from models.simple3dcnn import Simple3DCNN

class MNISTSimple3DCNN(Simple3DCNN):
    """
    MNIST-specific implementation of Simple3DCNN for video classification.
    
    This extends the base Simple3DCNN model with specific parameters for MNIST 
    video classification tasks.
    """
    def __init__(
        self, 
        num_frames: int = 10, 
        input_shape: tuple = (28, 28), 
        num_classes: int = 10, 
        dropout_rate: float = 0.5
    ) -> None:
        """
        Initialize the MNIST-specific Simple3DCNN model.
        
        Args:
            num_frames: Number of frames in each video sequence (default: 10)
            input_shape: Tuple of (height, width) for each frame (default: (28, 28))
            num_classes: Number of output classes (default: 10)
            dropout_rate: Dropout probability (default: 0.5)
        """
        # MNIST-specific parameters
        conv_channels = [2, 4, 8]  # Channels for each conv layer
        kernel_sizes = [2, 2, 2]   # Kernel sizes for each conv layer
        paddings = [0, 1, 1]       # Paddings for each conv layer
        fc_hidden = 128            # Hidden units in fully connected layer
        
        # Initialize the base model with MNIST-specific parameters
        super(MNISTSimple3DCNN, self).__init__(
            in_channels=1,  # MNIST videos have 1 channel
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            paddings=paddings,
            fc_hidden=fc_hidden,
            num_frames=num_frames,
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> 'MNISTSimple3DCNN':
        """Load a pretrained model from checkpoint"""
        model = cls(**kwargs)
        device = kwargs.get('device', None)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if device:
            model = model.to(device)
        return model
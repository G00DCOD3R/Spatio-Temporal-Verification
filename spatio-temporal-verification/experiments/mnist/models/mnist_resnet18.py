import torch
import torch.nn as nn
from typing import Type
from models.resnet18 import VideoResNet, BasicBlock, Conv3DSimple

class MNISTStem(nn.Sequential):
    """Custom stem for MNIST video sequences"""
    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(1, 64, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

class MNISTResNet18(VideoResNet):
    """ResNet18 model specifically designed for MNIST video sequences"""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(
            block=BasicBlock,
            conv_makers=[Conv3DSimple] * 4,
            layers=[2, 2, 2, 2],
            stem=MNISTStem,
            num_classes=num_classes
        )
        
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, num_classes: int = 10, device: str = None) -> 'MNISTResNet18':
        """Load a pretrained model from checkpoint"""
        model = cls(num_classes=num_classes)
        if device:
            model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model

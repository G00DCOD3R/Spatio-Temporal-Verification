import torch
import torch.nn as nn
import torch.nn.functional as F

class MedMNISTSimple3DCNN(nn.Module):
    """
    Simple 3D CNN model for MedMNIST 3D (SynapseMNIST3D) classification as defined in 
    Simple3DCNN-MedMNISTSynapse3D-Verification-ClosedForm-Lipschitz notebook.
    """
    def __init__(self, in_channels=1, num_classes=5, input_frames=64, resize_shape=(64, 64)):
        super(MedMNISTSimple3DCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_frames = input_frames
        self.resize_shape = resize_shape
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=2, kernel_size=2, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=2, out_channels=4, kernel_size=2, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=2, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Calculate flattened size dynamically via dummy forward pass
        with torch.no_grad():
            dummy_input = torch.rand(1, in_channels, input_frames, *resize_shape)
            dummy_output = self._forward_conv_layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
            
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv_layers(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

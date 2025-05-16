import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=50, in_channels=1):
        super(CNN3D, self).__init__()
        
        self.layers = []
        self.batch_norms = []        
        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.layers.append("Conv3d_2_0")
        
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.layers.append("BatchNorm3d_32")
        self.batch_norms.append(self.bn1)
        
        self.layers.append("ReLU")
        
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.layers.append("Conv3d_2_0")
        
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.layers.append("BatchNorm3d_48")
        self.batch_norms.append(self.bn2)
        
        self.relu = nn.ReLU(inplace=True)
        self.layers.append("ReLU")
        
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        
        # Add adaptive pooling layer to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        self.layers.append("Flatten")
        # Fixed size output from adaptive pooling: ch2 * 2 * 2 * 2
        fc_input_size = self.ch2 * 2 * 2 * 2
        
        self.fc1 = nn.Linear(fc_input_size, self.fc_hidden1)
        self.layers.append("Linear")
        self.layers.append("ReLU")
        
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.layers.append("Linear")
        self.layers.append("ReLU")
        
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes
        self.layers.append("Linear")
        
    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # Use adaptive pooling to get fixed size output regardless of input dimensions
        x = self.adaptive_pool(x)
        
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        
        return x 
    

class UdacitySteeringAngle_3DCNN(nn.Module):
    def __init__(self, num_frames=30, img_x=90, img_y=120, fc_hidden1=256, fc_hidden2=128, dropout=0.2, num_classes=3, in_channels=3):
        super(UdacitySteeringAngle_3DCNN, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_x = img_x
        self.img_y = img_y
        
        # Define the 3D CNN model
        self.cnn3d = CNN3D(t_dim=num_frames, img_x=img_x, img_y=img_y, fc_hidden1=fc_hidden1, 
                          fc_hidden2=fc_hidden2, num_classes=num_classes, in_channels=in_channels, drop_p=dropout)
        
    def forward(self, x):
        return self.cnn3d(x) 
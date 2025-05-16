import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Default image dimensions
img_x = 90
img_y = 120

# Dataset to convert Udacity images into sequence of frames
class UdacitySteeringAngleDataset(Dataset):
    """Custom Dataset for loading images and steering angles for Udacity self-driving car dataset."""
    
    def __init__(self, image_dir, labels_file, transform=None, train=True, test_size=0.2, random_state=42, resize_shape=(img_x, img_y), num_classes=10, num_frames=30):
        """
        Args:
            image_dir (str): Directory containing all the images.
            labels_file (str): Path to CSV file with image filenames and steering angles.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): training or testing split
            test_size (float): Proportion of the dataset to be used as test set.
            random_state (int): Random seed for reproducibility.
        """
        self.image_dir = image_dir
        self.labels_file = labels_file
        
        self.train = train
        self.img_x = resize_shape[0]
        self.img_y = resize_shape[1]
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.thresholds = np.linspace(-1.0, 1.0, num_classes + 1)
        
        
        # Define the transform to be applied to the images
        # If transform is not provided, use default transformations
        if transform is not None:
            self.transform = transform
        else:
            train_transform = transforms.Compose([
                transforms.Resize((self.img_x, self.img_y)),  # Typical size for Udacity dataset
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add some color variation
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small random shifts
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            test_transform = transforms.Compose([
                transforms.Resize((self.img_x, self.img_y)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if train:
                # Define transforms for steering angle prediction task
                self.transform = train_transform
            else:
                self.transform = test_transform
                
        
        # Read the CSV file with image filenames and steering angles
        self.data_frame = pd.read_csv(labels_file)
        
        # Get image paths and steering angles
        self.image_files = [os.path.join(image_dir, f"{filename}.jpg") for filename in self.data_frame['frame_id'].values]
        self.steering_angles = self.data_frame['steering_angle'].values.astype(np.float32)
        
        # Split the dataset into train and test sets
        train_files, test_files, train_angles, test_angles = train_test_split(
            self.image_files, self.steering_angles, test_size=test_size, random_state=random_state
        )
        
        # Select the appropriate data split
        if train:
            self.image_files = train_files
            self.steering_angles = train_angles
        else:  # 'test' split
            self.image_files = test_files
            self.steering_angles = test_angles
    
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)
    
    
    def __getitem__(self, idx):
        
        start_idx = idx
        if (start_idx + self.num_frames  >= len(self.image_files)):
            # If the index is too close to the end, we need to backtrack
            start_idx = len(self.image_files) - self.num_frames - 1
        
        
        # Load images
        images = []
        angle_labels = []
        for i in range(start_idx, start_idx+self.num_frames):
            img_path = self.image_files[idx]
            steering_angle = self.steering_angles[idx]
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            images.append(img)
            # Get steering angle and convert to class
            angle_labels.append(self._get_steering_angle_class_(steering_angle))
        
        # Stack images into tensor
        image_sequence = torch.stack(images, dim=1) # Shape: (channels, depth, W, H)
        
        
        steering_class = max(set(angle_labels), key=angle_labels.count)
        
        return image_sequence, steering_class


    def _get_steering_angle_class_(self, angle):
        """
        Convert continuous steering angles to discrete classes.
        
        Classes example (10 classes):
        - Extreme Left:  [-1.0 to -0.8]   -> 0
        - Hard Left:     [-0.8 to -0.6]   -> 1
        - Medium Left:   [-0.6 to -0.4]   -> 2
        - Soft Left:     [-0.4 to -0.2]   -> 3
        - Slight Left:   [-0.2 to 0.0]    -> 4
        - Slight Right:  [0.0 to 0.2]     -> 5
        - Soft Right:    [0.2 to 0.4]     -> 6
        - Medium Right:  [0.4 to 0.6]     -> 7
        - Hard Right:    [0.6 to 0.8]     -> 8
        - Extreme Right: [0.8 to 1.0]     -> 9
        
        Classes example (3 classes):
        - Left: [-1.0 to -0.333] -> 0
        - Straight:[-0.333 to 0.3333] -> 1
        - Right: [-0.333 to 1]   -> 2
        """
        angle = np.clip(angle, -1.0, 1.0)  # Ensure angle is within bounds
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= angle <= self.thresholds[i + 1]:
                return i
        return self.num_classes - 1  # For edge case of angle = 1.0 
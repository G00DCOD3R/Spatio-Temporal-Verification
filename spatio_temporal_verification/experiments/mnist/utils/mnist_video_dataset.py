import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

input_frames = 10  # Number of frames in each video sequence
resize_shape = (10, 10)  # Target image resolution

class MNISTVideoDataset(Dataset):
    def __init__(self, root, train=True, transform=None, frames=input_frames, resize_shape=resize_shape):
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.frames = frames
        self.resize_shape = resize_shape
        self.rotation = itertools.cycle([0, 5, 10, 15, 20, 25, -25, -20, -15, -10])
        self.translate = itertools.cycle([(0,0), (0,1), (1,2), (2,3), (3,4), (4,5), (-4,-5), (-3,-4), (-3,-2), (-2,-1)])
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: TF.affine(
                img,
                angle=next(self.rotation),
                translate=next(self.translate),
                scale=1.0,
                shear=0
            )),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.mnist) // self.frames

    def __getitem__(self, idx):
        start_idx = idx * self.frames
        video_frames = []
        label = self.mnist[start_idx][1]

        for i in range(self.frames):
            # img, _ = self.mnist[start_idx + i]
            img, _ = self.mnist[start_idx]
            img = self.transform(img)
            video_frames.append(img.squeeze(0).numpy())
            
        video_tensor = torch.tensor(np.stack(video_frames, axis=0), dtype=torch.float32).unsqueeze(0)
        return video_tensor, label 
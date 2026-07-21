import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Transform3D:
    """3D voxel transform as defined in MedMNIST 3D experiments notebook."""
    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        return voxel.astype(np.float32)

class MedMNIST3DDataset(Dataset):
    """
    Dataset wrapper for SynapseMNIST3D / MedMNIST 3D data.
    Loads NPZ file if present or provides synthetic fallback for dry-run testing.
    """
    def __init__(self, npz_file=None, split="test", transform=None, num_samples=50):
        self.split = split
        self.transform = transform
        
        if npz_file and os.path.exists(npz_file):
            data = np.load(npz_file)
            self.imgs = data[f"{split}_images"]
            self.labels = data[f"{split}_labels"]
        else:
            # Fallback synthetic dataset: (num_samples, 64, 64, 64) 3D volumes
            self.imgs = np.random.rand(num_samples, 64, 64, 64).astype(np.float32)
            self.labels = np.random.randint(0, 5, size=(num_samples, 1))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Ensure tensor format: (C, D, H, W)
        if isinstance(img, np.ndarray):
            tensor_img = torch.from_numpy(img)
        else:
            tensor_img = img
            
        if tensor_img.dim() == 3:
            tensor_img = tensor_img.unsqueeze(0)
            
        return tensor_img, int(label[0] if hasattr(label, '__len__') else label)

import os
import numpy as np
import torch
import torch.nn.functional as F
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
    Loads NPZ volumes, scales uint8 intensities to [0, 1], and downsamples
    to 32 cubed using area interpolation, matching the training pipeline.
    """
    def __init__(self, npz_file=None, split="test", transform=None, num_samples=50):
        self.split = split
        self.transform = transform
        
        if not npz_file or not os.path.isfile(npz_file):
            raise FileNotFoundError(
                f"MedMNIST dataset not found: {npz_file!r}. "
                "Pass --dataset_path data/medmnist/synapsemnist3d_64.npz."
            )
        with np.load(npz_file) as data:
            self.imgs = data[f"{split}_images"]
            self.labels = data[f"{split}_labels"]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]

        # Official NPZ volumes store uint8 intensities; floating-point inputs
        # are expected to already use the training range [0, 1].
        is_uint8 = img.dtype == np.uint8
        img = img.astype(np.float32)
        if is_uint8:
            img = img / 255.0
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Ensure tensor format: (C, D, H, W)
        if isinstance(img, np.ndarray):
            tensor_img = torch.from_numpy(img)
        else:
            tensor_img = img
            
        if tensor_img.dim() == 3:
            tensor_img = tensor_img.unsqueeze(0)

        tensor_img = tensor_img.to(dtype=torch.float32)
        if tensor_img.shape[-3:] != (32, 32, 32):
            tensor_img = F.interpolate(
                tensor_img.unsqueeze(0), size=(32, 32, 32), mode="area"
            ).squeeze(0)
            
        return tensor_img, int(label[0] if hasattr(label, '__len__') else label)

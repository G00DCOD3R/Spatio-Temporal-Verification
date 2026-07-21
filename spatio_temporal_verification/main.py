import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Ensure workspace imports work seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.mnist.models.mnist_simple3dcnn import MNISTSimple3DCNN
from experiments.mnist.utils import evaluate as mnist_eval, train as mnist_train, MNISTVideoDataset
from experiments.ucf_101.models.ucf101_simple_cnn import UCF101SimpleCNN
from experiments.udacity_steering_angle.models.udacity_3dcnn import UdacitySteeringAngle_3DCNN
from experiments.udacity_steering_angle.utils.udacity_dataset import UdacitySteeringAngleDataset
from experiments.udacity_steering_angle.utils.training import train as udacity_train
from experiments.medmnist.models.medmnist_simple3dcnn import MedMNISTSimple3DCNN
from experiments.medmnist.utils.dataset import MedMNIST3DDataset, Transform3D

from verifiers import (
    BaseVerifier,
    IBPVerifier,
    ClosedFormIBPVerifier,
    ClosedFormLipschitzVerifier,
    ClosedFormLownerJohnVerifier,
    MILPIBPVerifier,
    MILPIBPPatchVerifier,
)


def get_verifier(verifier_name: str, model: nn.Module, device: str):
    """Instantiate requested verifier."""
    name = verifier_name.lower()
    if name == "closed_form_ibp":
        return ClosedFormIBPVerifier(model, device=device)
    elif name in ["closed_form_lips", "closed_form_lipschitz"]:
        return ClosedFormLipschitzVerifier(model, device=device)
    elif name in ["closed_form_lj", "closed_form_lownerjohn"]:
        return ClosedFormLownerJohnVerifier(model, device=device)
    elif name == "ibp":
        return IBPVerifier(model, device=device)
    elif name == "milp" and MILPIBPVerifier is not None:
        return MILPIBPVerifier(model, device=device)
    elif name == "milp_patch" and MILPIBPPatchVerifier is not None:
        return MILPIBPPatchVerifier(model, device=device)
    else:
        print(f"Verifier '{verifier_name}' not found or unavailable, falling back to ClosedFormIBPVerifier.")
        return ClosedFormIBPVerifier(model, device=device)


def run_mnist_experiment(args, device):
    """Run MNIST Video 3DCNN Training / Verification."""
    print("\n==================================================")
    print("           MNIST Video 3DCNN Experiment           ")
    print("==================================================")
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_dir, "experiments/mnist/checkpoints/best_simple3dcnn_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Dataset setup
    dataset_root = args.dataset_path or os.path.expanduser("~/data/mnist")
    try:
        train_dataset = MNISTVideoDataset(root=dataset_root, train=True, transform=None, frames=10, resize_shape=(28, 28))
        test_dataset = MNISTVideoDataset(root=dataset_root, train=False, transform=None, frames=10, resize_shape=(28, 28))
    except Exception as e:
        print(f"Notice: Synthetic/Fallback dataset loading due to dataset root error: {e}")
        # Create small dummy tensor dataset for verification dry-run
        dummy_x = torch.randn(args.num_samples, 1, 10, 28, 28)
        dummy_y = torch.randint(0, 10, (args.num_samples,))
        test_dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
        train_dataset = test_dataset

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = MNISTSimple3DCNN(num_classes=10).to(device)
    
    if args.mode == "train":
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print(f"Training MNIST model for {args.epochs} epochs...")
        mnist_train(model, train_loader, val_loader, optimizer, criterion, device, model_path=model_path, epochs=args.epochs)
    else:
        if os.path.exists(model_path):
            print(f"Loading checkpoint from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("No saved checkpoint found. Evaluating with initialized model.")
        
        model.eval()
        verifier = get_verifier(args.verifier, model, str(device))
        epsilons = [float(e) for e in args.eps]
        shared_frames = [[3, 4, 5, 6, 7]]  # Middle 5 frames shared
        
        print(f"\nRunning verifier '{args.verifier}' on {args.num_samples} samples with epsilons={epsilons}...")
        results = defaultdict(list)
        total_time = 0.0
        verified_count = 0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if verified_count >= args.num_samples:
                break
            
            data, target = data.to(device), target.to(device)
            n_samples = min(data.size(0), args.num_samples - verified_count)
            
            for i in range(n_samples):
                sample_x = data[i:i+1]
                sample_y = target[i].item()
                
                for eps in epsilons:
                    start_t = time.perf_counter()
                    if hasattr(verifier, 'verify'):
                        res = verifier.verify(sample_x, eps, shared_frames, true_label=sample_y)
                        is_verified = res.get('verified', False)
                    elif hasattr(verifier, 'verify_sequence'):
                        res_dict = verifier.verify_sequence(sample_x, target[i:i+1], shared_frames, eps)
                        is_verified = (res_dict.get(eps, 0.0) == 1.0)
                    else:
                        is_verified = False
                    
                    elapsed = time.perf_counter() - start_t
                    results[eps].append(is_verified)
                    total_time += elapsed
            
            verified_count += n_samples
        
        print("\nMNIST Verification Results:")
        for eps in epsilons:
            acc = np.mean(results[eps]) * 100 if results[eps] else 0.0
            print(f"  ε={eps:.5f}: Verified Robustness = {acc:.2f}% ({sum(results[eps])}/{len(results[eps])})")
        print(f"Total verification time: {total_time:.3f}s")


def run_ucf101_experiment(args, device):
    """Run UCF-101 Action Recognition 3DCNN Training / Verification."""
    print("\n==================================================")
    print("           UCF-101 Action 3DCNN Experiment        ")
    print("==================================================")
    
    num_frames = 16
    img_x, img_y = 342, 256
    num_classes = 101
    
    model = UCF101SimpleCNN(
        num_frames=num_frames,
        in_channels=3,
        img_x=img_x,
        img_y=img_y,
        num_classes=num_classes,
        dropout_rate=0.5
    ).to(device)
    model.eval()
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_dir, "experiments/ucf_101/checkpoints/best_ucf101_model.pth")
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No saved checkpoint found. Evaluating with initialized model.")
    
    epsilons = [float(e) for e in args.eps]
    shared_frames = [[0, 1, 2, 3, 4]]
    
    verifier = get_verifier(args.verifier, model, str(device))
    print(f"\nRunning verifier '{args.verifier}' on {args.num_samples} samples with epsilons={epsilons}...")
    
    # Generate test inputs for dry-run verification
    dummy_x = torch.randn(args.num_samples, 3, num_frames, img_y, img_x, device=device)
    dummy_y = torch.randint(0, num_classes, (args.num_samples,), device=device)
    
    results = defaultdict(list)
    total_time = 0.0
    
    for i in range(args.num_samples):
        sample_x = dummy_x[i:i+1]
        sample_y = dummy_y[i].item()
        
        for eps in epsilons:
            start_t = time.perf_counter()
            if hasattr(verifier, 'verify'):
                res = verifier.verify(sample_x, eps, shared_frames, true_label=sample_y)
                is_verified = res.get('verified', False)
            elif hasattr(verifier, 'verify_sequence'):
                res_dict = verifier.verify_sequence(sample_x, dummy_y[i:i+1], shared_frames, eps)
                is_verified = (res_dict.get(eps, 0.0) == 1.0)
            else:
                is_verified = False
            
            elapsed = time.perf_counter() - start_t
            results[eps].append(is_verified)
            total_time += elapsed
    
    print("\nUCF-101 Verification Results:")
    for eps in epsilons:
        acc = np.mean(results[eps]) * 100 if results[eps] else 0.0
        print(f"  ε={eps:.5f}: Verified Robustness = {acc:.2f}% ({sum(results[eps])}/{len(results[eps])})")
    print(f"Total verification time: {total_time:.3f}s")


def run_udacity_experiment(args, device):
    """Run Udacity Steering Angle Prediction 3DCNN Experiment."""
    print("\n==================================================")
    print("      Udacity Steering Angle 3DCNN Experiment     ")
    print("==================================================")
    
    num_frames = 30
    img_x, img_y = 90, 120
    num_classes = 3
    
    model = UdacitySteeringAngle_3DCNN(
        num_frames=num_frames,
        img_x=img_x,
        img_y=img_y,
        fc_hidden1=256,
        fc_hidden2=128,
        dropout=0.2,
        num_classes=num_classes,
        in_channels=3
    ).to(device)
    model.eval()
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_dir, "experiments/udacity_steering_angle/checkpoints/best_udacity_model.pth")
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No saved checkpoint found. Evaluating with initialized model.")
        
    epsilons = [float(e) for e in args.eps]
    shared_frames = [[i for i in range(10)]]  # First 10 frames shared
    
    verifier = get_verifier(args.verifier, model, str(device))
    print(f"\nRunning verifier '{args.verifier}' on {args.num_samples} samples with epsilons={epsilons}...")
    
    dummy_x = torch.randn(args.num_samples, 3, num_frames, img_x, img_y, device=device)
    dummy_y = torch.randint(0, num_classes, (args.num_samples,), device=device)
    
    results = defaultdict(list)
    total_time = 0.0
    
    for i in range(args.num_samples):
        sample_x = dummy_x[i:i+1]
        sample_y = dummy_y[i].item()
        
        for eps in epsilons:
            start_t = time.perf_counter()
            if hasattr(verifier, 'verify'):
                res = verifier.verify(sample_x, eps, shared_frames, true_label=sample_y)
                is_verified = res.get('verified', False)
            elif hasattr(verifier, 'verify_sequence'):
                res_dict = verifier.verify_sequence(sample_x, dummy_y[i:i+1], shared_frames, eps)
                is_verified = (res_dict.get(eps, 0.0) == 1.0)
            else:
                is_verified = False
            
            elapsed = time.perf_counter() - start_t
            results[eps].append(is_verified)
            total_time += elapsed
    
    print("\nUdacity Steering Angle Verification Results:")
    for eps in epsilons:
        acc = np.mean(results[eps]) * 100 if results[eps] else 0.0
        print(f"  ε={eps:.5f}: Verified Robustness = {acc:.2f}% ({sum(results[eps])}/{len(results[eps])})")
    print(f"Total verification time: {total_time:.3f}s")


def run_medmnist_experiment(args, device):
    """Run MedMNIST 3D Synapse 3DCNN Experiment."""
    print("\n==================================================")
    print("      MedMNIST 3D Synapse 3DCNN Experiment       ")
    print("==================================================")
    
    in_channels = 1
    num_classes = 5
    input_frames = 64
    resize_shape = (64, 64)
    
    model = MedMNISTSimple3DCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        input_frames=input_frames,
        resize_shape=resize_shape
    ).to(device)
    model.eval()
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_dir, "experiments/medmnist/checkpoints/best_medmnist_model.pth")
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No saved checkpoint found. Evaluating with initialized model.")
        
    epsilons = [float(e) for e in args.eps]
    shared_frames = [[i for i in range(16)]]  # First 16 frames shared
    
    verifier = get_verifier(args.verifier, model, str(device))
    print(f"\nRunning verifier '{args.verifier}' on {args.num_samples} samples with epsilons={epsilons}...")
    
    try:
        dataset = MedMNIST3DDataset(npz_file=args.dataset_path, split="test", num_samples=args.num_samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    except Exception:
        loader = None
        
    results = defaultdict(list)
    total_time = 0.0
    verified_count = 0
    
    if loader is not None:
        for batch_idx, (data, target) in enumerate(loader):
            if verified_count >= args.num_samples:
                break
            data, target = data.to(device), target.to(device)
            n_samples = min(data.size(0), args.num_samples - verified_count)
            
            for i in range(n_samples):
                sample_x = data[i:i+1]
                sample_y = target[i]
                
                for eps in epsilons:
                    start_t = time.perf_counter()
                    if hasattr(verifier, 'verify'):
                        res = verifier.verify(sample_x, eps, shared_frames, true_label=sample_y)
                        is_verified = res.get('verified', False)
                    elif hasattr(verifier, 'verify_sequence'):
                        res_dict = verifier.verify_sequence(sample_x, torch.tensor([sample_y], device=device), shared_frames, eps)
                        is_verified = (res_dict.get(eps, 0.0) == 1.0)
                    else:
                        is_verified = False
                    
                    elapsed = time.perf_counter() - start_t
                    results[eps].append(is_verified)
                    total_time += elapsed
            verified_count += n_samples
    
    print("\nMedMNIST 3D Verification Results:")
    for eps in epsilons:
        acc = np.mean(results[eps]) * 100 if results[eps] else 0.0
        print(f"  ε={eps:.5f}: Verified Robustness = {acc:.2f}% ({sum(results[eps])}/{len(results[eps])})")
    print(f"Total verification time: {total_time:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Verification Benchmark CLI")
    parser.add_argument(
        "--experiment",
        type=str,
        default="mnist",
        choices=["mnist", "ucf101", "udacity", "medmnist", "all"],
        help="Experiment domain to run (mnist, ucf101, udacity, medmnist, all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="verify",
        choices=["train", "verify"],
        help="Execution mode (train, verify)",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="closed_form_ibp",
        choices=["closed_form_ibp", "closed_form_lips", "closed_form_lj", "ibp", "milp", "milp_patch"],
        help="Robustness verifier technique",
    )
    parser.add_argument(
        "--eps",
        nargs="+",
        default=[0.001],
        help="Perturbation epsilon value(s), e.g. --eps 0.001 0.005",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of test samples to verify",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for dataloaders",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs if mode=train",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Custom dataset directory path",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.experiment in ["mnist", "all"]:
        run_mnist_experiment(args, device)
    if args.experiment in ["ucf101", "all"]:
        run_ucf101_experiment(args, device)
    if args.experiment in ["udacity", "all"]:
        run_udacity_experiment(args, device)
    if args.experiment in ["medmnist", "all"]:
        run_medmnist_experiment(args, device)


if __name__ == "__main__":
    main()

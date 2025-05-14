import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from experiments.mnist.models.mnist_resnet18 import MNISTResNet18
from experiments.mnist.models.mnist_simple3dcnn import MNISTSimple3DCNN, Simple3DCNN
from experiments.mnist.utils import evaluate, train, MNISTVideoDataset
from verifiers.ibp_verifier import IBPVerifier
from verifiers.milp_ibp_patch_verifier import MILPIBPPatchVerifier
from verifiers.milp_ibp_verifier import MILPIBPVerifier


def verify_robustness(
    model, test_loader, device, num_test_samples=50, frame_groups=None, epsilons=None
):
    """
    Verify robustness using both pure IBP and hybrid MILP-IBP

    Args:
        model: The model to verify
        test_loader: DataLoader for test set
        device: Device to run verification on
        num_test_samples: Number of test samples to verify (default: 50)
    """
    print(f"\nVerifying robustness on {num_test_samples} test samples...")

    ibp_verifier = IBPVerifier(model, device=device)
    milp_verifier = MILPIBPVerifier(model, device=device)
    milp_patch_verifier = MILPIBPPatchVerifier(model, device=device)

    # Bounding box spec
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir,"experiments/mnist/sample_bounding_boxes.json"), "r") as f:
        bb_spec = json.load(f)

    # Store results and runtimes
    ibp_times = defaultdict(list)
    hybrid_times = defaultdict(list)
    hybrid_patch_times = defaultdict(list)
    ibp_accuracies = defaultdict(list)
    hybrid_accuracies = defaultdict(list)
    hybrid_patch_accuracies = defaultdict(list)
    
    # Calculate total number of verifications
    total_verifications = num_test_samples * len(frame_groups) * len(epsilons)
    print(f"Total verifications to perform: {total_verifications}")

    # Create progress bar
    pbar = tqdm(total=total_verifications * 2)  # *2 for both verifiers

    start_time = time.perf_counter()
    verified_samples = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        if verified_samples >= num_test_samples:
            break

        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)  # Use the actual batch size from data
        samples_to_verify = min(batch_size, num_test_samples - verified_samples)
        data = data[:samples_to_verify]
        target = target[:samples_to_verify]

        for group in frame_groups:
            group_desc = "grouped frames" if len(group) > 1 else "individual frames"
            print(f"\nVerifying group: {group_desc}")

            # Pure IBP verification
            print("\nRunning pure IBP verification...")

            for eps in epsilons:
                start_eps = time.perf_counter()
                ibp_result = ibp_verifier.verify_sequence(data, target, group, eps)
                end_eps = time.perf_counter()
                eps_time = end_eps - start_eps
                ibp_times[eps].append(eps_time)
                ibp_accuracies[eps].append(ibp_result[eps])

                # Update progress
                pbar.update(samples_to_verify)

                # Calculate and display ETA
                elapsed_time = time.perf_counter() - start_time
                verifications_done = len(ibp_times[eps]) * samples_to_verify
                time_per_verification = elapsed_time / verifications_done
                remaining_verifications = total_verifications * 2 - verifications_done
                eta_seconds = remaining_verifications * time_per_verification
                eta = datetime.now() + timedelta(seconds=int(eta_seconds))

                print(f"ε={eps}: Acc={ibp_result[eps]*100:.2f}% "
                      f"Time={eps_time/samples_to_verify:.3f}s/sample "
                      f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

            # Hybrid MILP-IBP verification
            print("\nRunning hybrid MILP-IBP verification...")

            for eps in epsilons:
                start_eps = time.perf_counter()
                hybrid_result = milp_verifier.verify_sequence(data, target, group, eps)
                end_eps = time.perf_counter()
                eps_time = end_eps - start_eps
                hybrid_times[eps].append(eps_time)
                hybrid_accuracies[eps].append(hybrid_result[eps])
                pbar.update(samples_to_verify)

                # Calculate and display ETA
                elapsed_time = time.perf_counter() - start_time
                verifications_done = len(hybrid_times[eps]) * samples_to_verify
                time_per_verification = elapsed_time / verifications_done
                remaining_verifications = total_verifications * 2 - verifications_done
                eta_seconds = remaining_verifications * time_per_verification
                eta = datetime.now() + timedelta(seconds=int(eta_seconds))

                print(f"ε={eps:.4f}: Acc={hybrid_result[eps]*100:.2f}% "
                      f"Time={eps_time/samples_to_verify:.3f}s/sample "
                      f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

            # Hybrid MILP-IBP Patch verification
            print("\nRunning hybrid MILP-IBP Patch verification...")

            for eps in epsilons:
                start_eps = time.perf_counter()
                hybrid_result = milp_patch_verifier.verify_sequence(
                    data, target, bb_spec, eps
                )
                end_eps = time.perf_counter()
                eps_time = end_eps - start_eps
                hybrid_patch_times[eps].append(eps_time)
                hybrid_patch_accuracies[eps].append(hybrid_result[eps])
                pbar.update(samples_to_verify)

                # Calculate and display ETA
                elapsed_time = time.perf_counter() - start_time
                verifications_done = len(hybrid_patch_times[eps]) * samples_to_verify
                time_per_verification = elapsed_time / verifications_done
                remaining_verifications = total_verifications * 2 - verifications_done
                eta_seconds = remaining_verifications * time_per_verification
                eta = datetime.now() + timedelta(seconds=int(eta_seconds))

                print(
                    f"ε={eps:.4f}: Acc={hybrid_result[eps]*100:.2f}% "
                    f"Time={eps_time/samples_to_verify:.3f}s/sample "
                    f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        verified_samples += samples_to_verify

    pbar.close()
    total_time = time.perf_counter() - start_time

    # Print final statistics
    print("\nVerification Complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Samples verified: {verified_samples}")

    print("\nPure IBP Statistics:")
    for eps in sorted(ibp_times.keys()):
        avg_time = np.mean(ibp_times[eps])
        avg_acc = np.mean(ibp_accuracies[eps]) * 100
        print(f"ε={eps}: Avg Time={avg_time:.3f}s Avg Acc={avg_acc:.2f}%")

    print("\nHybrid MILP-IBP Statistics:")
    for eps in sorted(hybrid_times.keys()):
        avg_time = np.mean(hybrid_times[eps])
        avg_acc = np.mean(hybrid_accuracies[eps]) * 100
        print(f"ε={eps}: Avg Time={avg_time:.3f}s Avg Acc={avg_acc:.2f}%")
    
    print("\nHybrid MILP-IBP Patch Statistics:")
    for eps in sorted(hybrid_patch_times.keys()):
        avg_time = np.mean(hybrid_patch_times[eps])
        avg_acc = np.mean(hybrid_patch_accuracies[eps]) * 100
        print(f"ε={eps}: Avg Time={avg_time:.3f}s Avg Acc={avg_acc:.2f}%")

    # Calculate average time per verification
    if verifications_done > 0:
        time_per_verification = elapsed_time / verifications_done
        print(f"\nAverage time per verification: {time_per_verification:.2f} seconds")
    else:
        print("\nNo verifications were completed successfully")

    # Extrapolate time for full test set
    samples_per_hour = verified_samples / (total_time / 3600)
    full_test_hours = 1000 / samples_per_hour  # Assuming ~1000 sequences in test set

    print("\nExtrapolated Time for Full Test Set:")
    print(f"Estimated time for all test sequences: {full_test_hours:.1f} hours")
    print(f"Verification throughput: {samples_per_hour:.1f} samples/hour")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    
    # (New code using MNISTVideoDataset:)
    train_dataset = MNISTVideoDataset(root="~/data/mnist", train=True, transform=None, frames=10, resize_shape=(28, 28))
    test_dataset = MNISTVideoDataset(root="~/data/mnist", train=False, transform=None, frames=10, resize_shape=(28, 28))

    # (Split train into train and val (80-20) using torch.utils.data.random_split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Set up model checkpoint path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    # Create and train ResNet18 model
    # model = MNISTResNet18(num_classes=10)
    # model = model.to(device)
    # model_path = os.path.join(cur_dir, "experiments/mnist/checkpoints/best_resnet18_model.pth")
    
    # Create and train Simple3DCNN model    
    model = MNISTSimple3DCNN(num_classes=10)
    model = model.to(device)
    model_path = os.path.join(cur_dir, "experiments/mnist/checkpoints/best_simple3dcnn_model.pth")
    
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()

    # print("Training model...")
    # train(model, train_loader, val_loader, optimizer, criterion, device, model_path=model_path, epochs=10)

    # Load best model for verification
    print("Loading best model...")
    model = MNISTSimple3DCNN(num_classes=10)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Verify robustness
    print("\nVerifying robustness...")
    epsilons = [0.0001]

    # Define frame groups to test
    frame_groups = [
        [[5]],  # All frames together
    ]
    verify_robustness(
        model,
        test_loader,
        device,
        num_test_samples=10,
        frame_groups=frame_groups,
        epsilons=epsilons,
    )


if __name__ == "__main__":
    print("Start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main()
    print("End time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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

# Import Udacity steering angle related modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.udacity_steering_angle.models.udacity_3dcnn import UdacitySteeringAngle_3DCNN
from experiments.udacity_steering_angle.utils.udacity_dataset import UdacitySteeringAngleDataset
from experiments.udacity_steering_angle.utils.training import train as udacity_train, evaluate as udacity_evaluate
from experiments.udacity_steering_angle.utils.udacity_verification_utils import get_bounding_box_info_udacity


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
    
    # Add option for Udacity steering angle model training or verification
    if os.environ.get("TRAIN_UDACITY", "0") == "1":
        train_udacity_model(device)
    elif os.environ.get("VERIFY_UDACITY", "0") == "1":
        udacity_verify_model(device)
    else:
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
        epsilons = [0.00001]

        # Define frame groups to test
        frame_groups = [
            [[3,4,5,6,7]],  # Middle 5 frames together
        ]
        verify_robustness(
            model,
            test_loader,
            device,
            num_test_samples=1000,
            frame_groups=frame_groups,
            epsilons=epsilons,
        )


def train_udacity_model(device):
    """
    Train the Udacity steering angle prediction model
    """
    print("Setting up Udacity steering angle model training...")
    
    # Configuration parameters
    DATASET_PATH = os.environ.get("UDACITY_DATASET_PATH", "./data/udacity")
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    num_frames = int(os.environ.get("NUM_FRAMES", "30"))
    num_classes = int(os.environ.get("NUM_CLASSES", "3"))
    img_x = int(os.environ.get("IMG_X", "90"))
    img_y = int(os.environ.get("IMG_Y", "120"))
    resize_shape = (img_x, img_y)
    epochs = int(os.environ.get("EPOCHS", "50"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.001"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
    dropout = float(os.environ.get("DROPOUT", "0.2"))
    patience = int(os.environ.get("PATIENCE", "10"))
    clip_grad = float(os.environ.get("CLIP_GRAD", "1.0"))
    
    # Create datasets as specified in the example
    train_dataset = UdacitySteeringAngleDataset(
        image_dir=f"{DATASET_PATH}/center", 
        labels_file=f"{DATASET_PATH}/final_example.csv", 
        train=True, 
        resize_shape=resize_shape, 
        num_classes=num_classes, 
        num_frames=num_frames
    )
    
    val_dataset = UdacitySteeringAngleDataset(
        image_dir=f"{DATASET_PATH}/center", 
        labels_file=f"{DATASET_PATH}/final_example.csv", 
        train=False, 
        resize_shape=resize_shape, 
        num_classes=num_classes, 
        num_frames=num_frames
    )
    
    train_dataset_at_eval = UdacitySteeringAngleDataset(
        image_dir=f"{DATASET_PATH}/center", 
        labels_file=f"{DATASET_PATH}/final_example.csv", 
        train=True, 
        resize_shape=resize_shape, 
        num_classes=num_classes
    )
    
    # Set the transform of the training eval to be the same as the validation dataset
    train_dataset_at_eval.transform = val_dataset.transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    train_loader_at_eval = DataLoader(
        train_dataset_at_eval, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Create model
    model = UdacitySteeringAngle_3DCNN(
        num_frames=num_frames,
        img_x=img_x,
        img_y=img_y,
        fc_hidden1=256,
        fc_hidden2=128,
        dropout=dropout,
        num_classes=num_classes,
        in_channels=3  # RGB images
    )
    
    model = model.to(device)
    
    # Set up model checkpoint path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(cur_dir, "experiments/udacity_steering_angle/checkpoints"), exist_ok=True)
    model_path = os.path.join(cur_dir, "experiments/udacity_steering_angle/checkpoints", "best_udacity_model.pth")
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Train model
    print("Training Udacity steering angle model...")
    train_results = udacity_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_path=model_path,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=patience,
        clip_grad=clip_grad
    )
    
    # Evaluate model on validation set
    print("Evaluating model on validation set...")
    val_loss, val_acc, val_preds, val_targets = udacity_evaluate(
        model=model,
        test_loader=val_loader,
        criterion=criterion,
        device=device
    )
    
    print(f"Validation accuracy: {val_acc:.2f}%")
    
    # Evaluate model on training set (with validation transforms)
    print("Evaluating model on training set (with validation transforms)...")
    train_loss, train_acc, train_preds, train_targets = udacity_evaluate(
        model=model,
        test_loader=train_loader_at_eval,
        criterion=criterion,
        device=device
    )
    
    print(f"Training accuracy (at eval): {train_acc:.2f}%")
    
    # Save final model
    torch.save(
        model.state_dict(), 
        os.path.join(cur_dir, "experiments/udacity_steering_angle/checkpoints", "final_udacity_model.pth")
    )
    print(f"Model saved to {model_path}")


def udacity_verify_model(device):
    """
    Verify robustness of the Udacity steering angle prediction model
    """
    print("Setting up Udacity steering angle model verification...")
    
    # Configuration parameters
    DATASET_PATH = os.environ.get("UDACITY_DATASET_PATH", "./data/udacity")
    BOUNDING_BOX_PATH = os.environ.get("BOUNDING_BOX_PATH", f"{DATASET_PATH}/bounding_boxes")
    MODEL_PATH = os.environ.get("MODEL_PATH", "")
    
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    num_frames = int(os.environ.get("NUM_FRAMES", "30"))
    num_classes = int(os.environ.get("NUM_CLASSES", "3"))
    img_x = int(os.environ.get("IMG_X", "90"))
    img_y = int(os.environ.get("IMG_Y", "120"))
    resize_shape = (img_x, img_y)
    
    # Verification parameters
    epsilons_str = os.environ.get("EPSILONS", "0.00001,0.0001,0.001")
    epsilons = [float(eps) for eps in epsilons_str.split(",")]
    
    frames_str = os.environ.get("FRAMES", "")
    if frames_str:
        frames = [int(f) for f in frames_str.split(",")]
        frame_groups = [frames]
    else:
        frame_groups = [[i for i in range(num_frames)]]  # Default: perturb all frames
    
    num_batches = int(os.environ.get("NUM_BATCHES", "1"))
    
    # Create datasets
    val_dataset = UdacitySteeringAngleDataset(
        image_dir=f"{DATASET_PATH}/center", 
        labels_file=f"{DATASET_PATH}/final_example.csv", 
        train=False, 
        resize_shape=resize_shape, 
        num_classes=num_classes, 
        num_frames=num_frames
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Load model
    model = UdacitySteeringAngle_3DCNN(
        num_frames=num_frames,
        img_x=img_x,
        img_y=img_y,
        fc_hidden1=256,
        fc_hidden2=128,
        dropout=0.2,
        num_classes=num_classes,
        in_channels=3
    )
    
    # Load model weights if provided
    if MODEL_PATH:
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        # Load from default location if exists
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(cur_dir, "experiments/udacity_steering_angle/checkpoints", "best_udacity_model.pth")
        if os.path.exists(default_path):
            print(f"Loading model from default path: {default_path}")
            model.load_state_dict(torch.load(default_path, map_location=device))
        else:
            print("WARNING: No model weights provided. Using random weights.")
    
    model = model.to(device)
    model.eval()
    
    # Initialize verifiers
    ibp_verifier = IBPVerifier(model, device=device)
    milp_ibp_verifier = MILPIBPVerifier(model, device=device)
    milp_ibp_patch_verifier = MILPIBPPatchVerifier(model, device=device)
    
    # Results storage
    ibp_accuracies = {eps: [] for eps in epsilons}
    milp_ibp_accuracies = {eps: [] for eps in epsilons}
    milp_ibp_patch_accuracies = {eps: [] for eps in epsilons}
    
    # Timing
    ibp_times = {eps: [] for eps in epsilons}
    milp_ibp_times = {eps: [] for eps in epsilons}
    milp_ibp_patch_times = {eps: [] for eps in epsilons}
    
    # Determine which verification methods to run
    run_ibp = os.environ.get("RUN_IBP", "1") == "1"
    run_milp_ibp = os.environ.get("RUN_MILP_IBP", "1") == "1"
    run_milp_ibp_patch = os.environ.get("RUN_MILP_IBP_PATCH", "1") == "1"
    
    # Run verification
    num_samples_verified = 0
    for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Verifying")):
        if batch_idx >= num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        # Print batch information
        print(f"\nBatch {batch_idx+1}/{min(num_batches, len(val_loader))}")
        print(f"Input shape: {data.shape}, Target shape: {target.shape}")
        
        # IBP Verification
        if run_ibp:
            print("\nRunning IBP verification...")
            for eps in epsilons:
                start_time = time.time()
                ibp_result = ibp_verifier.verify_sequence(data, target, frame_groups, eps)
                end_time = time.time()
                
                ibp_times[eps].append(end_time - start_time)
                ibp_accuracies[eps].append(ibp_result[eps])
                
                print(f"IBP (ε={eps}): Accuracy={ibp_result[eps]*100:.2f}%, Time={ibp_times[eps][-1]:.2f}s")
        
        # MILP-IBP Verification
        if run_milp_ibp:
            print("\nRunning MILP-IBP verification...")
            for eps in epsilons:
                start_time = time.time()
                milp_ibp_result = milp_ibp_verifier.verify_sequence(data, target, frame_groups, eps)
                end_time = time.time()
                
                milp_ibp_times[eps].append(end_time - start_time)
                milp_ibp_accuracies[eps].append(milp_ibp_result[eps])
                
                print(f"MILP-IBP (ε={eps}): Accuracy={milp_ibp_result[eps]*100:.2f}%, Time={milp_ibp_times[eps][-1]:.2f}s")
        
        # MILP-IBP-Patch Verification
        if run_milp_ibp_patch:
            print("\nRunning MILP-IBP-Patch verification...")
            for eps in epsilons:
                patch_accuracies = []
                patch_times = []
                
                for i in range(len(data)):
                    # Get filename from dataset
                    filename = val_loader.dataset.data_frame.loc[batch_idx*batch_size + i, 'frame_id']
                    
                    # Load bounding box information
                    try:
                        bounding_box_info = get_bounding_box_info_udacity(
                            BOUNDING_BOX_PATH, f"{filename}_tracked.json"
                        )
                        
                        start_time = time.time()
                        patch_result = milp_ibp_patch_verifier.verify_sequence(
                            data[i:i+1], target[i:i+1], bounding_box_info, eps
                        )
                        end_time = time.time()
                        
                        patch_times.append(end_time - start_time)
                        patch_accuracies.append(patch_result[eps])
                        
                        print(f"Sample {i+1}: MILP-IBP-Patch (ε={eps}): Accuracy={patch_result[eps]*100:.2f}%, Time={patch_times[-1]:.2f}s")
                    except FileNotFoundError:
                        print(f"Warning: Bounding box file not found for {filename}")
                        continue
                    except Exception as e:
                        print(f"Error during patch verification for sample {i+1}: {e}")
                        continue
                
                if patch_accuracies:
                    avg_accuracy = sum(patch_accuracies) / len(patch_accuracies)
                    avg_time = sum(patch_times) / len(patch_times)
                    milp_ibp_patch_accuracies[eps].append(avg_accuracy)
                    milp_ibp_patch_times[eps].append(avg_time)
                    
                    print(f"MILP-IBP-Patch (ε={eps}): Avg Accuracy={avg_accuracy*100:.2f}%, Avg Time={avg_time:.2f}s")
        
        num_samples_verified += len(data)
    
    # Print final results
    print("\n===== Final Results =====")
    
    if run_ibp:
        print("\nIBP Verification Results:")
        for eps in epsilons:
            avg_acc = sum(ibp_accuracies[eps]) / len(ibp_accuracies[eps]) if ibp_accuracies[eps] else 0
            avg_time = sum(ibp_times[eps]) / len(ibp_times[eps]) if ibp_times[eps] else 0
            print(f"ε={eps}: Avg Accuracy={avg_acc*100:.2f}%, Avg Time={avg_time:.2f}s")
    
    if run_milp_ibp:
        print("\nMILP-IBP Verification Results:")
        for eps in epsilons:
            avg_acc = sum(milp_ibp_accuracies[eps]) / len(milp_ibp_accuracies[eps]) if milp_ibp_accuracies[eps] else 0
            avg_time = sum(milp_ibp_times[eps]) / len(milp_ibp_times[eps]) if milp_ibp_times[eps] else 0
            print(f"ε={eps}: Avg Accuracy={avg_acc*100:.2f}%, Avg Time={avg_time:.2f}s")
    
    if run_milp_ibp_patch:
        print("\nMILP-IBP-Patch Verification Results:")
        for eps in epsilons:
            avg_acc = sum(milp_ibp_patch_accuracies[eps]) / len(milp_ibp_patch_accuracies[eps]) if milp_ibp_patch_accuracies[eps] else 0
            avg_time = sum(milp_ibp_patch_times[eps]) / len(milp_ibp_patch_times[eps]) if milp_ibp_patch_times[eps] else 0
            print(f"ε={eps}: Avg Accuracy={avg_acc*100:.2f}%, Avg Time={avg_time:.2f}s")
    
    print(f"\nTotal samples verified: {num_samples_verified}")


if __name__ == "__main__":
    print("Start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main()
    print("End time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

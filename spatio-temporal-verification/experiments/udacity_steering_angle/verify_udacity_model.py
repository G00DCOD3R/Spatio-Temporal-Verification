import os
import sys
import argparse
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# Import necessary modules
from experiments.udacity_steering_angle.models.udacity_3dcnn import UdacitySteeringAngle_3DCNN
from experiments.udacity_steering_angle.utils.udacity_dataset import UdacitySteeringAngleDataset
from experiments.udacity_steering_angle.utils.udacity_verification_utils import get_bounding_box_info_udacity
from verifiers.ibp_verifier import IBPVerifier
from verifiers.milp_ibp_verifier import MILPIBPVerifier
from verifiers.milp_ibp_patch_verifier import MILPIBPPatchVerifier
from torch.utils.data import DataLoader


def verify_udacity_model_robustness(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration parameters
    DATASET_PATH = args.dataset_path
    BOUNDING_BOX_PATH = args.bounding_box_path
    batch_size = args.batch_size
    num_frames = args.num_frames
    num_classes = args.num_classes
    img_x = args.img_x
    img_y = args.img_y
    resize_shape = (img_x, img_y)
    
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
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model = model.to(device)
    model.eval()
    
    # Initialize verifiers
    ibp_verifier = IBPVerifier(model, device=device)
    milp_ibp_verifier = MILPIBPVerifier(model, device=device)
    milp_ibp_patch_verifier = MILPIBPPatchVerifier(model, device=device)
    
    # Verification parameters
    epsilons = args.epsilons
    
    # Define frames to perturb
    if args.frames:
        frame_groups = [args.frames]
    else:
        frame_groups = [[i for i in range(num_frames)]]  # Default: perturb all frames
    
    # Results storage
    ibp_accuracies = {eps: [] for eps in epsilons}
    milp_ibp_accuracies = {eps: [] for eps in epsilons}
    milp_ibp_patch_accuracies = {eps: [] for eps in epsilons}
    
    # Timing
    ibp_times = {eps: [] for eps in epsilons}
    milp_ibp_times = {eps: [] for eps in epsilons}
    milp_ibp_patch_times = {eps: [] for eps in epsilons}
    
    # Run verification
    num_samples_verified = 0
    for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Verifying")):
        if batch_idx >= args.num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        # Print batch information
        print(f"\nBatch {batch_idx+1}/{min(args.num_batches, len(val_loader))}")
        print(f"Input shape: {data.shape}, Target shape: {target.shape}")
        
        # IBP Verification
        if args.run_ibp:
            print("\nRunning IBP verification...")
            for eps in epsilons:
                start_time = time.time()
                ibp_result = ibp_verifier.verify_sequence(data, target, frame_groups, eps)
                end_time = time.time()
                
                ibp_times[eps].append(end_time - start_time)
                ibp_accuracies[eps].append(ibp_result[eps])
                
                print(f"IBP (ε={eps}): Accuracy={ibp_result[eps]*100:.2f}%, Time={ibp_times[eps][-1]:.2f}s")
        
        # MILP-IBP Verification
        if args.run_milp_ibp:
            print("\nRunning MILP-IBP verification...")
            for eps in epsilons:
                start_time = time.time()
                milp_ibp_result = milp_ibp_verifier.verify_sequence(data, target, frame_groups, eps)
                end_time = time.time()
                
                milp_ibp_times[eps].append(end_time - start_time)
                milp_ibp_accuracies[eps].append(milp_ibp_result[eps])
                
                print(f"MILP-IBP (ε={eps}): Accuracy={milp_ibp_result[eps]*100:.2f}%, Time={milp_ibp_times[eps][-1]:.2f}s")
        
        # MILP-IBP-Patch Verification
        if args.run_milp_ibp_patch:
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
    
    if args.run_ibp:
        print("\nIBP Verification Results:")
        for eps in epsilons:
            avg_acc = sum(ibp_accuracies[eps]) / len(ibp_accuracies[eps]) if ibp_accuracies[eps] else 0
            avg_time = sum(ibp_times[eps]) / len(ibp_times[eps]) if ibp_times[eps] else 0
            print(f"ε={eps}: Avg Accuracy={avg_acc*100:.2f}%, Avg Time={avg_time:.2f}s")
    
    if args.run_milp_ibp:
        print("\nMILP-IBP Verification Results:")
        for eps in epsilons:
            avg_acc = sum(milp_ibp_accuracies[eps]) / len(milp_ibp_accuracies[eps]) if milp_ibp_accuracies[eps] else 0
            avg_time = sum(milp_ibp_times[eps]) / len(milp_ibp_times[eps]) if milp_ibp_times[eps] else 0
            print(f"ε={eps}: Avg Accuracy={avg_acc*100:.2f}%, Avg Time={avg_time:.2f}s")
    
    if args.run_milp_ibp_patch:
        print("\nMILP-IBP-Patch Verification Results:")
        for eps in epsilons:
            avg_acc = sum(milp_ibp_patch_accuracies[eps]) / len(milp_ibp_patch_accuracies[eps]) if milp_ibp_patch_accuracies[eps] else 0
            avg_time = sum(milp_ibp_patch_times[eps]) / len(milp_ibp_patch_times[eps]) if milp_ibp_patch_times[eps] else 0
            print(f"ε={eps}: Avg Accuracy={avg_acc*100:.2f}%, Avg Time={avg_time:.2f}s")
    
    print(f"\nTotal samples verified: {num_samples_verified}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify robustness of the Udacity steering angle model")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="./data/udacity", help="Path to the Udacity dataset")
    parser.add_argument("--bounding_box_path", type=str, default="./data/udacity/bounding_boxes", help="Path to bounding box JSON files")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--img_x", type=int, default=90, help="Image width")
    parser.add_argument("--img_y", type=int, default=120, help="Image height")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames in each sequence")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of steering angle classes")
    
    # Verification parameters
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.00001, 0.0001, 0.001, 0.01, 0.1], help="Perturbation sizes to test")
    parser.add_argument("--frames", type=int, nargs="+", help="Frame indices to perturb (default: all frames)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for verification")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to verify")
    
    # Verification methods
    parser.add_argument("--run_ibp", action="store_true", help="Run IBP verification")
    parser.add_argument("--run_milp_ibp", action="store_true", help="Run MILP-IBP verification")
    parser.add_argument("--run_milp_ibp_patch", action="store_true", help="Run MILP-IBP-Patch verification")
    
    args = parser.parse_args()
    
    # If no verification method is specified, run all
    if not (args.run_ibp or args.run_milp_ibp or args.run_milp_ibp_patch):
        args.run_ibp = True
        args.run_milp_ibp = True
        args.run_milp_ibp_patch = True
    
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    verify_udacity_model_robustness(args)
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
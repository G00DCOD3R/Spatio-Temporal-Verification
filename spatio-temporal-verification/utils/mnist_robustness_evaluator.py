import os
import time
import argparse
from typing import Dict, List, Tuple, Optional
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import model and dataset
from experiments.mnist.models.mnist_simple3dcnn import MNISTSimple3DCNN
from experiments.mnist.utils.mnist_video_dataset import MNISTVideoDataset

# Import verifiers
from verifiers.ibp_verifier import IBPVerifier
from verifiers.milp_ibp_verifier import MILPIBPVerifier
from verifiers.milp_ibp_patch_verifier import MILPIBPPatchVerifier


def create_frame_groups(max_frames: int, k: int) -> List[List[int]]:
    """
    Create frame groups for IBP and S-IBP verification.
    
    Args:
        max_frames: Total number of frames in the sequence
        k: Number of frames to perturb
        
    Returns:
        List of lists, where each sublist contains indices of frames to perturb together
    """
    # Select the first k frames to perturb
    return [list(range(k))]


def create_bounding_box_info(max_frames: int, k: int, frame_height: int, frame_width: int) -> Dict:
    """
    Create bounding box info for S-IBP Patch verification.
    
    Args:
        max_frames: Total number of frames in the sequence
        k: Number of frames to perturb
        frame_height: Height of each frame
        frame_width: Width of each frame
        
    Returns:
        Dictionary with bounding box information
    """
    # Create a simple patch that covers the entire frame for the first k frames
    bounding_box_info = {"0": {"bounding_box": []}}
    
    for i in range(k):
        # Create frame number (e.g., "frame000001.jpg")
        frame_name = f"frame{(i+1):06d}.jpg"
        
        # Add normalized coordinates for full frame coverage
        bounding_box_info["0"]["bounding_box"].append({
            "image": frame_name,
            "xyxyn": [0, 0, 1, 1]  # Full frame [x1, y1, x2, y2] normalized coordinates
        })
    
    return bounding_box_info


def evaluate_verifiers(
    model,
    test_dataset,
    num_examples: int = 1000,
    epsilons: List[float] = [0.1, 0.01, 0.001, 0.0001, 0.00001],
    max_frames: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate robustness using IBP, S-IBP, and S-IBP Patch verifiers.
    
    Args:
        model: Trained model to verify
        test_dataset: MNIST video dataset for testing
        num_examples: Number of examples to test (default: 1000)
        epsilons: List of epsilon values to test
        max_frames: Maximum number of frames to perturb
        device: Device to run verification on
        
    Returns:
        Dictionary containing verification results
    """
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Create subset of test dataset
    subset_indices = list(range(min(num_examples, len(test_dataset))))
    subset_dataset = Subset(test_dataset, subset_indices)
    test_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)
    
    # Initialize verifiers
    ibp_verifier = IBPVerifier(model, device=device)
    milp_ibp_verifier = MILPIBPVerifier(model, device=device)
    milp_ibp_patch_verifier = MILPIBPPatchVerifier(model, device=device)
    
    # Dictionary to store results
    results = {
        "IBP": {str(epsilon): {str(k): 0 for k in range(1, max_frames + 1)} for epsilon in epsilons},
        "S-IBP": {str(epsilon): {str(k): 0 for k in range(1, max_frames + 1)} for epsilon in epsilons},
        "S-IBP-Patch": {str(epsilon): {str(k): 0 for k in range(1, max_frames + 1)} for epsilon in epsilons}
    }
    
    # For each epsilon and k value
    for epsilon in epsilons:
        print(f"\nTesting with epsilon = {epsilon}")
        epsilon_str = str(epsilon)  # Convert to string for consistent dictionary access
        
        for k in range(1, max_frames + 1):
            print(f"  Testing with {k} perturbed frames")
            k_str = str(k)  # Convert to string for consistent dictionary access
            
            # Create frame groups for IBP and S-IBP
            frame_groups = create_frame_groups(max_frames, k)
            
            # Track number of verified examples for each verifier
            ibp_verified, milp_ibp_verified, milp_ibp_patch_verified = 0, 0, 0
            
            # Iterate through test examples
            for i, (x, y) in enumerate(tqdm(test_loader, desc=f"Verifying (ε={epsilon}, k={k})")):
                x, y = x.to(device), y.to(device)
                
                # Verify with IBP
                ibp_result = ibp_verifier.verify_sequence(x, y, frame_groups, epsilon)
                ibp_robust = ibp_result[epsilon] == 1.0
                if ibp_robust:
                    ibp_verified += 1
                
                # Verify with S-IBP
                milp_ibp_result = milp_ibp_verifier.verify_sequence(x, y, frame_groups, epsilon)
                milp_ibp_robust = milp_ibp_result[epsilon] == 1.0
                if milp_ibp_robust:
                    milp_ibp_verified += 1
                
                # Create bounding box info for S-IBP Patch
                frame_height, frame_width = x.shape[3], x.shape[4]
                bounding_box_info = create_bounding_box_info(max_frames, k, frame_height, frame_width)
                
                # Verify with S-IBP Patch
                milp_ibp_patch_result = milp_ibp_patch_verifier.verify_sequence(x, y, bounding_box_info, epsilon)
                milp_ibp_patch_robust = milp_ibp_patch_result[epsilon] == 1.0
                if milp_ibp_patch_robust:
                    milp_ibp_patch_verified += 1
            
            # Store results using string keys
            results["IBP"][epsilon_str][k_str] = ibp_verified / len(subset_dataset)
            results["S-IBP"][epsilon_str][k_str] = milp_ibp_verified / len(subset_dataset)
            results["S-IBP-Patch"][epsilon_str][k_str] = milp_ibp_patch_verified / len(subset_dataset)
            
            # Print results for current configuration
            print(f"    IBP: {ibp_verified}/{len(subset_dataset)} = {ibp_verified/len(subset_dataset):.4f}")
            print(f"    S-IBP: {milp_ibp_verified}/{len(subset_dataset)} = {milp_ibp_verified/len(subset_dataset):.4f}")
            print(f"    S-IBP-Patch: {milp_ibp_patch_verified}/{len(subset_dataset)} = {milp_ibp_patch_verified/len(subset_dataset):.4f}")
    
    return results


def plot_verification_results(
    results: Dict,
    output_dir: str = "./plots",
    max_k: int = 20,  # Limit x-axis to make the plot more readable
    selected_epsilons: Optional[List[str]] = None,
    selected_methods: Optional[List[str]] = None
):
    """
    Plot verification results with patch sizes on the x-axis.
    
    Args:
        results: Dictionary containing verification results
        output_dir: Directory to save output plots
        max_k: Maximum value of k to plot
        selected_epsilons: List of epsilon values to include, or None for all
        selected_methods: List of verification methods to include, or None for all
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available methods, epsilons, and k values
    methods = list(results.keys()) if selected_methods is None else selected_methods
    
    # Make sure all selected methods exist in the results
    for method in methods:
        if method not in results:
            print(f"Warning: Method '{method}' not found in results. Available methods: {list(results.keys())}")
    
    # Only keep methods that exist in the results
    methods = [method for method in methods if method in results]
    
    # Get all epsilon values across all methods
    all_epsilons = set()
    for method in methods:
        all_epsilons.update(results[method].keys())
    
    # Sort epsilon values numerically
    # Convert epsilon strings to floats for sorting
    epsilon_values = []
    for eps in all_epsilons:
        try:
            epsilon_values.append(float(eps))
        except ValueError:
            # If it's a string like "1e-05"
            epsilon_values.append(float(eps))
    
    epsilon_values.sort()
    
    # Create a mapping from float values to their corresponding string keys in the results
    epsilon_key_map = {}
    for method in methods:
        for eps_key in results[method].keys():
            try:
                epsilon_key_map[float(eps_key)] = eps_key
            except ValueError:
                # Handle any conversion errors
                pass
    
    # Use the actual keys from the dictionary
    epsilons = [epsilon_key_map.get(eps, str(eps)) for eps in epsilon_values]
    
    # Filter epsilons if specified
    if selected_epsilons is not None:
        epsilons = [eps for eps in epsilons if eps in selected_epsilons]
    
    # Ensure we have valid methods and epsilons before proceeding
    if not methods or not epsilons:
        print("No valid methods or epsilon values found. Cannot generate plots.")
        return
    
    # Get k values (assuming same for all methods and epsilons)
    # Check if the expected key exists
    first_method = methods[0]
    first_epsilon = epsilons[0]
    
    if first_epsilon not in results[first_method]:
        # Try to find the closest match
        print(f"Warning: Epsilon key '{first_epsilon}' not found in results. Available keys: {list(results[first_method].keys())}")
        # Try to use any available key
        first_epsilon = list(results[first_method].keys())[0]
        print(f"Using '{first_epsilon}' instead.")
    
    k_values = [int(k) for k in results[first_method][first_epsilon].keys()]
    k_values.sort()
    
    # Limit k values if max_k is specified
    k_values = [k for k in k_values if k <= max_k]
    
    # Set seaborn style for better visualizations
    sns.set_style("whitegrid")
    sns.set_context('poster', rc={"lines.linewidth": 2.5})
    
    # Define colors for epsilon values
    # colors = plt.cm.Dark2(np.linspace(0, 1, len(epsilons)))
    colors = plt.cm.Accent(np.linspace(0, 1, len(epsilons)))
    
    # Define line styles for different methods
    line_styles = {
        "IBP": "-",
        "S-IBP": "--",
        "S-IBP-Patch": "-."
    }
    
    # Define markers for different methods
    markers = {
        "IBP": "o",
        "S-IBP": "s",
        "S-IBP-Patch": "^"
    }
    
    # Create a mapping for display names in the legend
    method_display_names = {
        "IBP": "IBP",
        "S-IBP": "STBP",
        "S-IBP-Patch": "STBP-Patch"
    }
    
    # Create plot with k on the x-axis
    plt.figure(figsize=(12, 8))
    
    # Plot data for each method and epsilon
    for i, method in enumerate(methods):
        for j, eps in enumerate(epsilons):
            # Extract accuracy values for this method, epsilon and k values
            try:
                accuracies = [float(results[method][eps][str(k)]) for k in k_values]
                
                # Create label
                label = f"{method_display_names.get(method, method)}, ε={eps}"
                
                # Plot line with distinct style and color
                plt.plot(
                    k_values, 
                    accuracies, 
                    linestyle=line_styles.get(method, "-"),
                    marker=markers.get(method, "o"),
                    markersize=10,  # Increased marker size
                    markeredgecolor='black',  # Add black edge to markers
                    markeredgewidth=1.0,
                    linewidth=2.5,  # Ensure line width is consistent
                    # color=colors[j],
                    alpha=0.8,
                    label=label
                )
            except KeyError as e:
                print(f"Warning: Could not plot {method_display_names.get(method, method)} with ε={eps}. KeyError: {e}")
                continue
    
    # Set plot attributes
    plt.xlabel("Patch Size (k)", fontsize=14)
    plt.ylabel("Robust Accuracy", fontsize=14)
    plt.title("Robust Accuracy vs. Patch Size", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Set x-axis ticks to be integers
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Create legend with smaller font and place outside the plot area on the right
    plt.legend(fontsize=14, loc="center left", bbox_to_anchor=(1.02, 0.5))
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.78, 1])
    
    # Save plot
    output_file = os.path.join(output_dir, "verification_results_by_k.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    
    # Create separate plots for each epsilon value
    for eps in epsilons:
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            # Extract accuracy values for this method, epsilon and k values
            try:
                accuracies = [float(results[method][eps][str(k)]) for k in k_values]
                
                # Plot line
                plt.plot(
                    k_values, 
                    accuracies, 
                    linestyle=line_styles.get(method, "-"),
                    marker=markers.get(method, "o"),
                    markersize=10,  # Increased marker size
                    markeredgecolor='black',  # Add black edge to markers
                    markeredgewidth=1.0,
                    linewidth=2.5,  # Ensure line width is consistent
                    label=method_display_names.get(method, method)
                )
            except KeyError as e:
                print(f"Warning: Could not plot {method_display_names.get(method, method)} with ε={eps} in epsilon-specific plot. KeyError: {e}")
                continue

        # Set plot attributes
        plt.xlabel("Patch Size (k)", fontsize=12)
        plt.ylabel("Robust Accuracy", fontsize=12)
        plt.title(f"Robust Accuracy vs. Patch Size (ε={eps})", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Place legend in the upper right, but avoid overlapping the plot
        plt.legend(loc="best", fontsize=14, framealpha=0.9)
        
        # Set x-axis ticks to be integers
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        eps_str = str(eps).replace(".", "_")
        output_file = os.path.join(output_dir, f"verification_results_epsilon_{eps_str}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    
    # Create separate plots for each method
    for method in methods:
        plt.figure(figsize=(10, 6))
        
        for j, eps in enumerate(epsilons):
            # Extract accuracy values for this method, epsilon and k values
            try:
                accuracies = [float(results[method][eps][str(k)]) for k in k_values]
                
                # Plot line
                plt.plot(
                    k_values, 
                    accuracies, 
                    linestyle="-",
                    marker="o",
                    markersize=10,  # Increased marker size
                    markeredgecolor='black',  # Add black edge to markers
                    markeredgewidth=1.0,
                    linewidth=2.5,  # Ensure line width is consistent
                    color=colors[j],
                    label=f"ε={eps}"
                )
            except KeyError as e:
                print(f"Warning: Could not plot ε={eps} for {method_display_names.get(method, method)} in method-specific plot. KeyError: {e}")
                continue
        
        # Set plot attributes
        plt.xlabel("Patch Size (k)", fontsize=12)
        plt.ylabel("Robust Accuracy", fontsize=12)
        plt.title(f"Robust Accuracy vs. Patch Size ({method_display_names.get(method, method)})", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Place legend in the upper right, but avoid overlapping the plot
        if len(epsilons) > 5:
            # If we have many epsilon values, place legend outside to the right
            plt.legend(fontsize=14, loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            # For fewer epsilon values, try to place it in an empty area
            plt.legend(loc="best", fontsize=14, framealpha=0.9)
            plt.tight_layout()
        
        # Set x-axis ticks to be integers
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Save plot
        output_file = os.path.join(output_dir, f"verification_results_{method}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        
    # Now create a single comprehensive plot with epsilon on the x-axis for all methods and k values
    plt.figure(figsize=(12.5, 8))  # Increase figure width to make room for legends
    
    # Define colors for k values - use a colormap that provides good distinction
    k_colors = plt.cm.tab10(np.linspace(0, 1, len(k_values)))
    
    # Plot data for each method and k value
    for i, method in enumerate(methods):
        # Define different line styles and markers for each method
        line_style = line_styles.get(method, "-")
        marker = markers.get(method, "o")
        
        for j, k in enumerate(k_values):
            try:
                # Convert epsilons to float for proper ordering on x-axis
                eps_float = [float(eps) for eps in epsilons]
                eps_float, sorted_indices = zip(*sorted(zip(eps_float, range(len(eps_float)))))
                
                # Get corresponding accuracies
                accuracies = []
                for eps_idx in sorted_indices:
                    eps = epsilons[eps_idx]
                    accuracies.append(float(results[method][eps][str(k)]))
                
                # Create label
                method_label = method_display_names.get(method, method)
                label = f"k={k}"
                
                # Plot line with distinct style based on method and color based on k
                plt.plot(
                    eps_float, 
                    accuracies, 
                    linestyle=line_style,
                    marker=marker,
                    markersize=10,  # Increased marker size
                    markeredgecolor='black',  # Add black edge to markers
                    markeredgewidth=1.0,
                    linewidth=2.5,  # Ensure line width is consistent
                    color=k_colors[j],
                    label=f"{method_label}, {label}" if method == methods[0] else "_nolegend_"
                )
            except KeyError as e:
                print(f"Warning: Could not plot {method_display_names.get(method, method)} with k={k}. KeyError: {e}")
                continue
    
    # Set plot attributes
    plt.xlabel("Epsilon (ε) - Perturbation Magnitude", fontsize=14)
    plt.ylabel("Robust Accuracy", fontsize=14)
    plt.title("Robust Accuracy vs. Perturbation Magnitude", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Use log scale for x-axis since epsilon values can be very small
    plt.xscale('log')
    
    # Create a single combined legend with both method and k value information
    # First collect legend handles and labels for the k values
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Create custom handles for the methods
    method_handles = []
    method_labels = []
    for method in methods:
        line = plt.Line2D([0], [0], color='black', 
                          linestyle=line_styles.get(method, "-"),
                          marker=markers.get(method, "o"),
                          markersize=10,
                          markeredgecolor='black',
                          markeredgewidth=1.0,
                          linewidth=3.0)
        method_handles.append(line)
        method_labels.append(method_display_names.get(method, method))
    
    # Position the legend outside the plot on the right
    # plt.figlegend(handles + method_handles, 
    #               labels + method_labels,
    #               loc='center left', 
    #               bbox_to_anchor=(1.02, 0.5),
    #               fontsize=10,
    #               framealpha=1.0,
    #               title="Legend")
    plt.figlegend(handles + method_handles, 
                  labels + method_labels,
                  loc='upper right', 
                  bbox_to_anchor=(0.98, 0.94),
                  fontsize=14,
                  framealpha=1.0)
                #   title="Legend")
    
    # Adjust layout with specific dimensions to make room for the legend
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.80, 1])
    
    # Save plot with extra width for the legend
    output_file = os.path.join(output_dir, "verification_by_epsilon.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate robustness of verifiers on MNIST')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to test')
    parser.add_argument('--max_frames', type=int, default=10, help='Maximum number of frames to perturb')
    parser.add_argument('--model_path', type=str, default='experiments/mnist/checkpoints/best_simple3dcnn_model.pth', 
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--max_k', type=int, default=20, help='Maximum k value to include in plots')
    parser.add_argument('--plot_only', action='store_true', help='Only generate plots from existing results file')
    parser.add_argument('--results_file', type=str, default=None, 
                        help='Path to JSON results file (for --plot_only)')
    args = parser.parse_args()
    
    if args.plot_only and args.results_file:
        # Only generate plots from existing results file
        print(f"Loading results from {args.results_file} for plotting...")
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        # Generate plots
        plot_verification_results(
            results=results,
            output_dir=args.output_dir,
            max_k=args.max_k
        )
    else:
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load MNIST video dataset
        test_dataset = MNISTVideoDataset(
            root="~/data/mnist", 
            train=False, 
            transform=None, 
            frames=10, 
            resize_shape=(28, 28)
        )
        
        # Load model
        model = MNISTSimple3DCNN(num_classes=10)
        model_path = args.model_path
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Define epsilon values
        epsilons = [0.001, 0.0001, 0.00001]
        
        # Ensure epsilons are properly formatted - convert to strings for results file
        epsilons_formatted = [float(eps) for eps in epsilons]  # Normalize as floats first
        
        # Evaluate verifiers
        start_time = time.time()
        results = evaluate_verifiers(
            model=model,
            test_dataset=test_dataset,
            num_examples=args.num_examples,
            epsilons=epsilons_formatted,
            max_frames=args.max_frames,
            device=device
        )
        end_time = time.time()
        
        print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
        
        # Save results as JSON
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, 'verification_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")
        
        # Generate plots
        plot_verification_results(
            results=results,
            output_dir=args.output_dir,
            max_k=args.max_k
        )
        print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main() 
    
# Usage: 
# cd spatio-temporal-verification && python utils/mnist_robustness_evaluator.py --num_examples 1000 --max_frames 10 --model_path experiments/mnist/checkpoints/best_simple3dcnn_model.pth --output_dir experiments/mnist/results
# python utils/mnist_robustness_evaluator.py --num_examples 100 --max_frames 10 --model_path experiments/mnist/checkpoints/best_simple3dcnn_model.pth --output_dir experiments/mnist/results

# Usage: to generate plots from existing results file
# python utils/mnist_robustness_evaluator.py --plot_only --results_file experiments/mnist/results/verification_results.json --output_dir experiments/mnist/results


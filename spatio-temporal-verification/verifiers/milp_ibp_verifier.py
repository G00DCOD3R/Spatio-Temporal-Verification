import copy
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gurobipy import GRB
from models.resnet18 import VideoResNet
from models.simple3dcnn import Simple3DCNN
from .ibp_verifier import IBPVerifier

class MILPIBPVerifier(IBPVerifier):
    def __init__(self, model: nn.Module, device: str = "cuda"):
        super().__init__(model)
        self.device = device
        
        # Set Gurobi parameters for better performance
        self.gurobi_params = {
            "Method": 1,  # Dual simplex
            "Presolve": 2,  # Aggressive presolve
            "NumericFocus": 3,  # Highest numerical accuracy
            "TimeLimit": 60,  # 60 second timeout
            "OutputFlag": 0,  # Suppress output
            "Threads": 1,  # Single thread per model
            "FeasibilityTol": 1e-6,  # Tighter feasibility tolerance
            "OptimalityTol": 1e-6,  # Tighter optimality tolerance
        }

    def generate_perturbation_constraints(
        self,
        input_tensor: torch.Tensor,
        frames: List[List[int]],
        epsilons: List[float],
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> Tuple[torch.Tensor, Dict, Dict, Dict]:
        """
        Generates constraints on the perturbation tensor based on input frames and epsilon bounds.

        Args:
            input_tensor: The base single input tensor (C, D, H, W)
            frames: Each sublist specifies a group of frames with shared perturbation constraints
            epsilons: Perturbation bounds for each frame group in `frames`
            kernel_size: Kernel size of the convolutional layer
            stride: Stride of the convolutional layer
            padding: Padding applied to the convolutional layer

        Returns:
            Tuple containing:
            - perturbation_tensor: The tensor where perturbations will be applied
            - bounds_dict: Bounds on the perturbation values
            - equality_dict: Fixed values for frames not in specified groups
            - shared_dict: Dictionary for shared perturbation constraints
        """
        C, D, H, W = input_tensor.shape  # Channels, Depth, Height, Width

        # Initialize the perturbation tensor to zeros
        perturbation_tensor = torch.zeros_like(input_tensor)

        bounds_dict = {}
        equality_dict = {}
        shared_dict = {}

        # Maximum valid index in the flattened tensor to avoid index overflow
        max_index = C * D * H * W

        # Iterate over each frame group to set constraints within receptive fields
        for group_index, frame_group in enumerate(frames):
            epsilon = epsilons[group_index]

            # Iterate over each channel
            for c in range(C):
                # Use the first frame in the group as the reference for bounds
                first_frame = frame_group[0]
                for h in range(H):  # Iterate over height of frame
                    for w in range(W):  # Iterate over width of frame
                        # Calculate the flat index for the first frame in the group
                        first_frame_index = (
                            (c * D * H * W) + (first_frame * H * W) + (h * W) + w
                        )

                        # Ensure first_frame_index is within bounds
                        if first_frame_index < max_index:
                            # Set bounds for the first frame in the group
                            bounds_dict[first_frame_index] = {
                                "lower": -epsilon,
                                "upper": epsilon,
                            }

                        # For other frames in the group, set as shared with the first frame
                        for f in frame_group[1:]:  # Skip the first frame
                            # Calculate the flat index for other frames in the group
                            flat_index = (c * D * H * W) + (f * H * W) + (h * W) + w

                            # Check if indices are within bounds to prevent index errors
                            if flat_index < max_index and first_frame_index < max_index:
                                shared_dict[flat_index] = first_frame_index

        # Populate equality_dict with strict values for frames not in specified groups
        frames_set = set(
            f for group in frames for f in group
        )  # All frames in specified groups

        for c in range(C):  # Iterate over each channel
            for d in range(D):  # Iterate over each depth (frame)
                if d not in frames_set:  # Frames not in specified groups
                    for h in range(H):  # Iterate over each height position
                        for w in range(W):  # Iterate over each width position
                            # Calculate flat index for the current position
                            flat_index = (c * D * H * W) + (d * H * W) + (h * W) + w
                            # Ensure the index is valid and not in other dictionaries
                            if (
                                flat_index < max_index
                                and flat_index not in bounds_dict
                                and flat_index not in shared_dict
                            ):
                                # Set equality constraints in equality_dict, keeping perturbation strictly zero
                                equality_dict[flat_index] = 0.0

        return perturbation_tensor, bounds_dict, equality_dict, shared_dict

    def compute_3D_conv_bounds(
        self,
        input_tensor: torch.Tensor,
        perturbation_tensor: torch.Tensor,
        bounds_dict: Dict,
        equality_dict: Dict,
        shared_dict: Dict,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        conv_layer: nn.Module = None,
    ) -> List[List[Tuple[float, float]]]:
        """
        Efficiently computes the bounds on the output of the first 3D convolutional layer given perturbation constraints.

        Args:
            input_tensor: The base input tensor
            perturbation_tensor: The perturbation tensor with applied constraints
            bounds_dict: Bounds on the perturbation values
            equality_dict: Fixed values for frames not in specified groups
            shared_dict: Dictionary for shared perturbation constraints
            kernel_size: Kernel size of the convolutional layer
            stride: Stride of the convolutional layer
            padding: Padding applied to the convolutional layer
            conv_layer: The first convolutional layer

        Returns:
            List of computed lower and upper bounds for each output element of the first convolutional layer
        """
        # Extract first convolutional layer weights and biases
        weights = conv_layer.weight.cpu().detach().numpy()
        biases = conv_layer.bias.cpu().detach().numpy()
        out_channels, in_channels, kernel_d, kernel_h, kernel_w = weights.shape

        # Flatten input and perturbation tensors
        base_input = input_tensor.flatten().cpu().detach().numpy()
        C, D, H, W = input_tensor.shape

        # Identify which input indices have perturbations (i.e., not fixed)
        perturbed_indices = set(bounds_dict.keys())

        # Initialize Gurobi model with parameters
        m = gp.Model("efficient_conv_perturb")
        for param, value in self.gurobi_params.items():
            m.setParam(param, value)

        # Create perturbation variables only for perturbed inputs
        perturb_vars = {}
        for idx in perturbed_indices:
            lb = bounds_dict[idx].get("lower", -GRB.INFINITY)
            ub = bounds_dict[idx].get("upper", GRB.INFINITY)
            perturb_vars[idx] = m.addVar(lb=lb, ub=ub, name=f"perturb_{idx}")

        # Apply shared constraints: tie perturbation variables if necessary
        for idx1, idx2 in shared_dict.items():
            if idx1 in perturb_vars and idx2 in perturb_vars:
                m.addConstr(
                    perturb_vars[idx1] == perturb_vars[idx2],
                    name=f"shared_{idx1}_{idx2}",
                )

        m.update()

        # Prepare output bounds storage
        output_bounds = []

        # Compute output dimensions after convolution
        D_out = (D + 2 * padding - kernel_d) // stride + 1
        H_out = (H + 2 * padding - kernel_h) // stride + 1
        W_out = (W + 2 * padding - kernel_w) // stride + 1

        # Loop over output channels
        for out_channel in range(out_channels):
            channel_bounds = []

            # Loop over each output spatial location
            for d_out in range(D_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        const_term = 0.0
                        perturb_expr = gp.LinExpr()

                        # For each receptive field contributing to this output
                        for in_channel in range(in_channels):
                            for kd in range(kernel_d):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        input_d = d_out * stride + kd - padding
                                        input_h = h_out * stride + kh - padding
                                        input_w = w_out * stride + kw - padding

                                        if (
                                            0 <= input_d < D
                                            and 0 <= input_h < H
                                            and 0 <= input_w < W
                                        ):
                                            flat_index = (
                                                in_channel * D * H * W
                                                + input_d * H * W
                                                + input_h * W
                                                + input_w
                                            )
                                            weight = weights[
                                                out_channel, in_channel, kd, kh, kw
                                            ]

                                            if flat_index in perturb_vars:
                                                perturb_expr += (
                                                    weight * perturb_vars[flat_index]
                                                )
                                            else:
                                                # Fixed input, contribute directly
                                                const_term += (
                                                    weight * base_input[flat_index]
                                                )

                        # Add bias
                        const_term += biases[out_channel]

                        # Solve for lower bound
                        m.setObjective(perturb_expr + const_term, GRB.MINIMIZE)
                        try:
                            m.optimize()
                            if m.status == GRB.OPTIMAL:
                                lower_bound = m.ObjVal
                            else:
                                print(f"Warning: Lower bound optimization failed with status {m.status}")
                                # Use IBP bounds as fallback
                                lower_bound = const_term - abs(perturb_expr.getCoeff().sum() * max(abs(bounds_dict[idx]["lower"]) for idx in perturb_vars))
                        except Exception as e:
                            print(f"Error computing lower bound: {str(e)}")
                            lower_bound = float("-inf")

                        # Solve for upper bound
                        m.setObjective(perturb_expr + const_term, GRB.MAXIMIZE)
                        try:
                            m.optimize()
                            if m.status == GRB.OPTIMAL:
                                upper_bound = m.ObjVal
                            else:
                                print(f"Warning: Upper bound optimization failed with status {m.status}")
                                # Use IBP bounds as fallback
                                upper_bound = const_term + abs(perturb_expr.getCoeff().sum() * max(abs(bounds_dict[idx]["upper"]) for idx in perturb_vars))
                        except Exception as e:
                            print(f"Error computing upper bound: {str(e)}")
                            upper_bound = float("inf")

                        # Sanity check
                        if upper_bound < lower_bound:
                            print(f"Warning: Upper bound ({upper_bound}) < Lower bound ({lower_bound})")
                            # Use IBP bounds as fallback
                            lower_bound = const_term - abs(perturb_expr.getCoeff().sum() * max(abs(bounds_dict[idx]["lower"]) for idx in perturb_vars))
                            upper_bound = const_term + abs(perturb_expr.getCoeff().sum() * max(abs(bounds_dict[idx]["upper"]) for idx in perturb_vars))

                        channel_bounds.append((lower_bound, upper_bound))

            output_bounds.append(channel_bounds)

        return output_bounds

    def format_bounds_for_next_layer(
        self, input_tensor: torch.Tensor, output_bounds: List[List[Tuple[float, float]]], conv_layer: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Formats the computed bounds from a convolutional layer to be used as input bounds for the next layer.

        Args:
            input_tensor: The input batch tensor with shape (batch_size, channels, depth, height, width)
            output_bounds: Bounds computed by `compute_3D_conv_bounds`

        Returns:
            Tuple of (lower_bounds, upper_bounds) tensors formatted for the next layer input
        """
        # Get first input sample in the batch and compute output shape after applying the first conv layer
        single_input = input_tensor  # Shape (C, D, H, W)
        conv_output = conv_layer(single_input)
        single_output_shape = conv_output.shape  # Exclude batch dimension (C, D, H, W)

        out_channels = single_output_shape[0]
        output_shape = (
            single_output_shape  # Output shape of the conv layer for individual inputs
        )

        # Initialize tensors for lower and upper bounds with computed shape
        lower_bounds = torch.empty(output_shape)
        upper_bounds = torch.empty(output_shape)

        # Flatten the output bounds list for easy indexing
        bounds_flat = [
            bound for channel_bounds in output_bounds for bound in channel_bounds
        ]

        # Reshape bounds into tensors of the correct shape for the next layer
        index = 0
        for c in range(out_channels):
            for d in range(output_shape[1]):  # Depth dimension
                for h in range(output_shape[2]):  # Height dimension
                    for w in range(output_shape[3]):  # Width dimension
                        lower, upper = bounds_flat[index]
                        if lower is not None:
                            lower_bounds[c, d, h, w] = lower
                        else:
                            print("err")
                            lower_bounds[c, d, h, w] = float(
                                "-inf"
                            )  # We can replace with IBP bounds later if we want
                        if upper is not None:
                            upper_bounds[c, d, h, w] = upper
                        else:
                            upper_bounds[c, d, h, w] = float("inf")
                        index += 1

        lower_bounds = lower_bounds.to(self.device)
        upper_bounds = upper_bounds.to(self.device)
        return lower_bounds, upper_bounds

    def selective_interval_bound_propagation_resnet(
        self,
        input_tensor: torch.Tensor,
        epsilons: List[float],
        sets_of_frames: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective Interval Bound propagation with LP applied to the first layer of the network

        Args:
            input_tensor: Input tensor
            epsilons: List of epsilon values for perturbation
            sets_of_frames: List of lists each corresponding to one set of shared perturbed frames

        Returns:
            Tuple of (lower_bounds, upper_bounds) for the entire batch
        """
        lower_bounds, upper_bounds = None, None

        perturbation_tensor, bounds_dict, equality_dict, shared_dict = (
            self.generate_perturbation_constraints(
                input_tensor[0],
                sets_of_frames,
                epsilons,
                kernel_size=2,
                stride=1,
                padding=0,
            )
        )

        for i in range(len(input_tensor)):
            # Compute bounds for the first layer using compute_3D_conv_bounds
            output_bounds = self.compute_3D_conv_bounds(
                input_tensor[i],
                perturbation_tensor,
                bounds_dict,
                equality_dict,
                shared_dict,
                kernel_size=2,
                stride=1,
                padding=0,
                conv_layer=self.model.stem[0],
            )

            # Format bounds for the next layer
            lower_bounds_first, upper_bounds_first = self.format_bounds_for_next_layer(
                input_tensor[i], output_bounds, self.model.stem[0]
            )
            
            # Apply naive IBP to the first layer and use the tightest bounds
            # Add the epsilon perturbations to the frames
            lower_bounds_ibp, upper_bounds_ibp = self._compute_bounds(input_tensor[i], epsilons[0], sets_of_frames)
            lower_bounds_ibp, upper_bounds_ibp = self._propagate_conv3d(lower_bounds_ibp, upper_bounds_ibp, self.model.stem[0])
            lower_bounds_first = torch.min(lower_bounds_first, lower_bounds_ibp)
            upper_bounds_first = torch.min(upper_bounds_first, upper_bounds_ibp)
            
            lower_bounds_first = lower_bounds_first.unsqueeze(0)
            upper_bounds_first = upper_bounds_first.unsqueeze(0)

            # Perform IBP for the remaining of the stem
            lower_bounds_first, upper_bounds_first = self._propagate_batchnorm3d(
                lower_bounds_first, upper_bounds_first, self.model.stem[1]
            )
            lower_bounds_first, upper_bounds_first = self._propagate_relu(
                lower_bounds_first, upper_bounds_first
            )

            lb, ub = self.propagate_bounds_resnet(
                input_tensor[i].unsqueeze(0),
                lower_input=lower_bounds_first,
                upper_input=upper_bounds_first,
                skip_stem=True,
            )

            if lower_bounds is None and upper_bounds is None:
                lower_bounds = lb
                upper_bounds = ub
            else:
                lower_bounds = torch.cat((lower_bounds, lb), dim=0)
                upper_bounds = torch.cat((upper_bounds, ub), dim=0)

        return lower_bounds, upper_bounds
    

    def selective_interval_bound_propagation_3dcnn(
        self,
        input_tensor: torch.Tensor,
        epsilons: List[float],
        sets_of_frames: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective Interval Bound propagation with LP applied to the first layer of the network

        Args:
            input_tensor: Input tensor
            epsilons: List of epsilon values for perturbation
            sets_of_frames: List of lists each corresponding to one set of shared perturbed frames

        Returns:
            Tuple of (lower_bounds, upper_bounds) for the entire batch
        """
        lower_bounds, upper_bounds = None, None

        perturbation_tensor, bounds_dict, equality_dict, shared_dict = (
            self.generate_perturbation_constraints(
                input_tensor[0],
                sets_of_frames,
                epsilons,
                kernel_size=2,
                stride=1,
                padding=0,
            )
        )

        for i in range(len(input_tensor)):
            # Compute bounds for the first layer using compute_3D_conv_bounds
            output_bounds = self.compute_3D_conv_bounds(
                input_tensor[i],
                perturbation_tensor,
                bounds_dict,
                equality_dict,
                shared_dict,
                kernel_size=2,
                stride=1,
                padding=0,
                conv_layer=self.model.conv1,
            )

            # Format bounds for the next layer
            lower_bounds_first, upper_bounds_first = self.format_bounds_for_next_layer(
                input_tensor[i], output_bounds, self.model.conv1
            )
            
            
            lower_bounds_first = lower_bounds_first.unsqueeze(0)
            upper_bounds_first = upper_bounds_first.unsqueeze(0)

            # Perform IBP for the remaining layers                
            lb, ub = self.propagate_bounds_simple3dcnn(
                input_tensor[i].unsqueeze(0),
                lower_input=lower_bounds_first,
                upper_input=upper_bounds_first,
                skip_first_conv=True,
            )

            if lower_bounds is None and upper_bounds is None:
                lower_bounds = lb
                upper_bounds = ub
            else:
                lower_bounds = torch.cat((lower_bounds, lb), dim=0)
                upper_bounds = torch.cat((upper_bounds, ub), dim=0)

        return lower_bounds, upper_bounds
    
    
    def verify_sequence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        frame_group: List[List[int]],
        epsilon: float,
    ) -> Dict[float, float]:
        """
        Verify robustness of the model for given frame groups and epsilon

        Args:
            x: Input tensor of shape (batch_size, channels, num_frames, height, width)
            y: Ground truth labels
            frame_group: List of lists containing frame indices to perturb together
            epsilon: Perturbation size

        Returns:
            Dictionary containing verification results for each epsilon
        """
        results = {}
        verified_correct = 0
        total = x.size(0)

        print(f"\nMILP-IBP Verification with epsilon={epsilon}")
        print(f"Input shape: {x.shape}")
        print(f"Number of samples: {total}")

        for i in range(total):
            print(f"\nVerifying sample {i+1}/{total}")
            input_tensor = x[i:i+1]
            y_i = y[i:i+1]
        
            with torch.no_grad():
                if isinstance(self.model, VideoResNet):
                    lower_bounds_milp, upper_bounds_milp = self.selective_interval_bound_propagation_resnet(
                        input_tensor, [epsilon], frame_group
                    )
                elif isinstance(self.model, Simple3DCNN):
                    lower_bounds_milp, upper_bounds_milp = self.selective_interval_bound_propagation_3dcnn(
                        input_tensor, [epsilon], frame_group
                    )
            
            # Print bound statistics
            print(f"Lower bounds range: [{lower_bounds_milp.min().item():.4f}, {lower_bounds_milp.max().item():.4f}]")
            print(f"Upper bounds range: [{upper_bounds_milp.min().item():.4f}, {upper_bounds_milp.max().item():.4f}]")
            print(f"Bound width: {(upper_bounds_milp - lower_bounds_milp).mean().item():.4f}")

            # Use the same robustness check as IBP
            margin = lower_bounds_milp[0, y_i] - torch.max(
                torch.cat([upper_bounds_milp[0, :y_i], upper_bounds_milp[0, y_i + 1 :]], dim=0)
            )

            if margin.item() > 0:
                verified_correct += 1
                print(f"Sample {i+1} is robust (margin: {margin.item():.4f})")
            else:
                print(f"Sample {i+1} is not robust (margin: {margin.item():.4f})")

        results[epsilon] = verified_correct / total
        return results

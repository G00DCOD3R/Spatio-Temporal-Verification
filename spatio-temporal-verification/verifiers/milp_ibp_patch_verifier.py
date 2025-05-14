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


class MILPIBPPatchVerifier(IBPVerifier):
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        epsilons: Optional[List[float]] = None,
        milp_timeout: int = 60,
        parallel_jobs: int = 4,
    ):
        super().__init__(model)
        self.device = device
        self.epsilons = epsilons if epsilons is not None else [0.1]
        self.milp_timeout = milp_timeout
        self.parallel_jobs = parallel_jobs

        # State variables for tracking verification progress
        self.verified_count = 0
        self.total_count = 0
        self.start_time = None
        self.last_update_time = None
        self.verified_correct = 0
        self.total = 0

        # Set Gurobi parameters for better performance
        self.gurobi_params = {
            "Method": 1,  # Dual simplex
            "Presolve": 2,  # Aggressive presolve
            "NumericFocus": 3,  # Highest numerical accuracy
            "TimeLimit": milp_timeout,
            "OutputFlag": 0,  # Suppress output
            "Threads": 1,  # Single thread per model
        }

    def generate_perturbation_constraints_patch(
        self,
        input_tensor: torch.Tensor,
        frame_patch_locations: Dict[int, List[Tuple[int, int, int, int]]],
        epsilon: float,
    ) -> Tuple[torch.Tensor, Dict, Dict, Dict]:
        """
        Generate perturbation constraints for a 3D CNN under patch constraints.

        Args:
            input_tensor: The base input tensor (C, D, H, W)
            frame_patch_locations: Dictionary mapping frame indices to lists of (x, y, w, h) tuples
            epsilon: Maximum allowed perturbation

        Returns:
            Tuple containing:
            - perturbation_tensor: The tensor where perturbations will be applied
            - bounds_dict: Bounds on the perturbation values
            - equality_dict: Fixed values for frames not in specified groups
            - shared_dict: Dictionary for shared perturbation constraints
        """
        C, D, H, W = input_tensor.shape
        bounds_dict = {}
        shared_dict = {}
        patch_shared_indices = {}

        # Iterate through each frame and its list of bounding boxes
        for frame_idx, boxes in frame_patch_locations.items():
            for x, y, w, h in boxes:
                for c in range(C):
                    for i in range(w):
                        for j in range(h):
                            h_idx = x + i
                            w_idx = y + j
                            if 0 <= frame_idx < D and 0 <= h_idx < H and 0 <= w_idx < W:
                                flat_index = (
                                    c * D * H * W
                                    + frame_idx * H * W
                                    + h_idx * W
                                    + w_idx
                                )
                                bounds_dict[flat_index] = {
                                    "lower": -epsilon,
                                    "upper": epsilon,
                                }

                                key = (c, i, j)
                                if key in patch_shared_indices:
                                    shared_dict[flat_index] = patch_shared_indices[key]
                                else:
                                    patch_shared_indices[key] = flat_index

        perturbation_tensor = torch.zeros_like(input_tensor)
        equality_dict = {}  # Not needed for patch version

        return perturbation_tensor, bounds_dict, equality_dict, shared_dict

    def compute_3D_conv_bounds_patch(
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
        Computes the bounds on the output of the first 3D convolutional layer given perturbation constraints.

        Args:
            input_tensor: The base input tensor
            perturbation_tensor: The perturbation tensor (not used)
            bounds_dict: Bounds on the perturbation values
            equality_dict: Not used (for compatibility)
            shared_dict: Dictionary for shared perturbation constraints
            kernel_size: Kernel size of the convolutional layer
            stride: Stride of the convolutional layer
            padding: Padding applied to the convolutional layer
            conv_layer: The first convolutional layer to compute bounds for

        Returns:
            List of computed lower and upper bounds for each output element of the first convolutional layer
        """

        weights = conv_layer.weight.cpu().detach().numpy()
        biases = conv_layer.bias.cpu().detach().numpy()
        out_channels, in_channels, kernel_d, kernel_h, kernel_w = weights.shape

        base_input = input_tensor.flatten().cpu().detach().numpy()
        C, D, H, W = input_tensor.shape

        perturbed_indices = set(bounds_dict.keys())

        m = gp.Model("conv_bounds_patch")
        m.setParam("OutputFlag", 0)

        perturb_vars = {}
        for idx in perturbed_indices:
            lb = bounds_dict[idx].get("lower", -GRB.INFINITY)
            ub = bounds_dict[idx].get("upper", GRB.INFINITY)
            perturb_vars[idx] = m.addVar(lb=lb, ub=ub, name=f"perturb_{idx}")

        for idx1, idx2 in shared_dict.items():
            if idx1 in perturb_vars and idx2 in perturb_vars:
                m.addConstr(
                    perturb_vars[idx1] == perturb_vars[idx2],
                    name=f"shared_{idx1}_{idx2}",
                )

        m.update()

        output_bounds = []

        D_out = (D + 2 * padding - kernel_d) // stride + 1
        H_out = (H + 2 * padding - kernel_h) // stride + 1
        W_out = (W + 2 * padding - kernel_w) // stride + 1

        for out_channel in range(out_channels):
            channel_bounds = []

            for d_out in range(D_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        const_term = 0.0
                        perturb_expr = gp.LinExpr()

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
                                                const_term += (
                                                    weight * base_input[flat_index]
                                                )

                        const_term += biases[out_channel]

                        if perturb_expr.size() == 0:
                            lower_bound = upper_bound = const_term
                        else:
                            m.setObjective(perturb_expr + const_term, GRB.MINIMIZE)
                            m.optimize()
                            lower_bound = (
                                m.ObjVal if m.status == GRB.OPTIMAL else const_term
                            )

                            m.setObjective(perturb_expr + const_term, GRB.MAXIMIZE)
                            m.optimize()
                            upper_bound = (
                                m.ObjVal if m.status == GRB.OPTIMAL else const_term
                            )

                        channel_bounds.append((float(lower_bound), float(upper_bound)))

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

    def selective_interval_bound_propagation_patch_resnet(
        self, input_tensor: torch.Tensor, bounding_box_info: Dict, epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective Interval Bound propagation with LP applied to the first layer of the network for patch-based verification

        Args:
            input_tensor: Input tensor
            bounding_box_info: Dictionary containing patch information
            epsilon: Epsilon perturbation value

        Returns:
            Tuple of (lower_bounds, upper_bounds) for the entire batch
        """
        lower_bounds, upper_bounds = None, None

        frame_patch_locations = self.process_bounding_boxes(
            bounding_box_info, input_tensor.shape[3], input_tensor.shape[4]
        )

        for i in range(len(input_tensor)):
            # Generate perturbation constraints
            perturbation_tensor, bounds_dict, equality_dict, shared_dict = (
                self.generate_perturbation_constraints_patch(
                    input_tensor[i], frame_patch_locations, epsilon
                )
            )

            # Get bounds for the first layer using MILP
            output_bounds = self.compute_3D_conv_bounds_patch(
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
            lower_bounds_ibp, upper_bounds_ibp = self._compute_bounds(input_tensor[i], epsilon, [list(range(input_tensor[i].shape[1]))] ) # all the frames
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

            # Propagate bounds through remaining layers
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
    
    def selective_interval_bound_propagation_patch_3dcnn(
        self, input_tensor: torch.Tensor, bounding_box_info: Dict, epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective Interval Bound propagation with LP applied to the first layer of the network for patch-based verification

        Args:
            input_tensor: Input tensor
            bounding_box_info: Dictionary containing patch information
            epsilon: Epsilon perturbation value

        Returns:
            Tuple of (lower_bounds, upper_bounds) for the entire batch
        """
        lower_bounds, upper_bounds = None, None

        frame_patch_locations = self.process_bounding_boxes(
            bounding_box_info, input_tensor.shape[3], input_tensor.shape[4]
        )

        for i in range(len(input_tensor)):
            # Generate perturbation constraints
            perturbation_tensor, bounds_dict, equality_dict, shared_dict = (
                self.generate_perturbation_constraints_patch(
                    input_tensor[i], frame_patch_locations, epsilon
                )
            )

            # Get bounds for the first layer using MILP
            output_bounds = self.compute_3D_conv_bounds_patch(
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
    

    def process_bounding_boxes(
        self, bounding_box_info: Dict, H: int, W: int
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Process bounding box information into frame_patch_locations format.

        Args:
            bounding_box_info: Dictionary containing bounding box information for each object
            H: Height of the video frames
            W: Width of the video frames

        Returns:
            Dictionary with frame indices as keys and lists of (x1, y1, x2, y2) tuples as values
        """
        frame_patch_locations = {}

        # Process each object's bounding boxes
        for obj_id, obj_data in bounding_box_info.items():
            for frame_data in obj_data["bounding_box"]:
                # Extract frame number from image name (e.g., "frame000001.jpg" -> 1)
                frame_num = int(frame_data["image"].split("frame")[1].split(".")[0]) - 1

                # Get normalized bounding box coordinates
                x1, y1, x2, y2 = frame_data["xyxyn"]

                # Convert normalized coordinates to pixel coordinates
                x1_px = int(x1 * W)
                y1_px = int(y1 * H)
                x2_px = int(x2 * W)
                y2_px = int(y2 * H)

                # Store the coordinates in frame_patch_locations as a list
                if frame_num not in frame_patch_locations:
                    frame_patch_locations[frame_num] = []
                frame_patch_locations[frame_num].append((x1_px, y1_px, x2_px, y2_px))

        return frame_patch_locations

    def verify_sequence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bounding_box_info: Dict,
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

        print(f"\nMILP-IBP Patch Verification with epsilon={epsilon}")
        print(f"Input shape: {x.shape}")
        print(f"Number of samples: {total}")

        for i in range(total):
            print(f"\nVerifying sample {i+1}/{total}")
            input_tensor = x[i:i+1]
            y_i = y[i:i+1]
        
            with torch.no_grad():
                if isinstance(self.model, VideoResNet):
                    lower_bounds_milp, upper_bounds_milp = self.selective_interval_bound_propagation_patch_resnet(
                        input_tensor, bounding_box_info, epsilon
                    )
                elif isinstance(self.model, Simple3DCNN):
                    lower_bounds_milp, upper_bounds_milp = self.selective_interval_bound_propagation_patch_3dcnn(
                        input_tensor, bounding_box_info, epsilon
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

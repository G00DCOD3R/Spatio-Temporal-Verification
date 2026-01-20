"""
Closed-Form IBP Verifier for Spatio-Temporal Video Classification

This module implements efficient neural network robustness verification that combines:
1. GPU-accelerated closed-form solution for first Conv3D layer bounds (EXACT)
2. Interval Bound Propagation (IBP) for subsequent layers - operates in L∞ norm space

============================================================================
THEORETICAL FOUNDATION
============================================================================

For the first convolutional layer with structured perturbations, we derive an
ANALYTICAL (closed-form) solution that eliminates the need for LP optimization.

Problem Setup:
    Given input x ∈ ℝ^{C×D×H×W} and perturbation δ satisfying:
    - δ_i ∈ [-ε, ε] for bounded indices (L∞ constraint)
    - δ_i = δ_j for (i,j) in shared frame groups
    - δ_i = 0 for fixed indices

    First layer output: z = Conv3D(x + δ) = W * (x + δ) + b

Key Insight (Lemma - Analytical Optimum):
    For an LP of the form:
        max/min  c^T δ
        s.t.     δ_i ∈ [-ε, ε]
                 δ_i = δ_j for shared pairs (i,j)

    The optimal solution is:
        δ*_g = ε × sign(Σ_{i∈g} c_i)  for each independent group g

============================================================================
SOUNDNESS
============================================================================

Both stages operate entirely in L∞ (coordinate-wise) space:
- First layer: Exact bounds for L∞ perturbations with shared constraints
- Subsequent layers: IBP propagates coordinate-wise intervals

There is NO norm mismatch. The bounds are sound and complete for the first layer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_verifier import BaseVerifier


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LayerBounds:
    """
    Container for layer activation bounds with utility properties.
    
    Attributes:
        lower: Lower bounds tensor
        upper: Upper bounds tensor
    """
    lower: torch.Tensor
    upper: torch.Tensor
    
    @property
    def center(self) -> torch.Tensor:
        """Compute center of bounds: (L + U) / 2"""
        return (self.lower + self.upper) / 2
    
    @property
    def radius(self) -> torch.Tensor:
        """Compute half-width of bounds: (U - L) / 2"""
        return (self.upper - self.lower) / 2
    
    @property
    def width(self) -> torch.Tensor:
        """Compute total width of bounds: U - L"""
        return self.upper - self.lower
    
    def to(self, device: str) -> 'LayerBounds':
        """Move bounds to specified device."""
        return LayerBounds(
            lower=self.lower.to(device),
            upper=self.upper.to(device)
        )


@dataclass
class PerturbationSpec:
    """
    Specification for structured spatio-temporal perturbations.
    
    Models the constraint set T(x) = {x + δ : δ satisfies constraints}
    
    Attributes:
        epsilon: Perturbation budget (ℓ∞ bound)
        shared_frames: List of frame groups with shared perturbations
                       e.g., [[0,1,2,3,4]] means all 5 frames share same δ per spatial location
        input_shape: Shape of input tensor (C, D, H, W)
    """
    epsilon: float
    shared_frames: List[List[int]]
    input_shape: Tuple[int, int, int, int]
    
    @property
    def all_frames_shared(self) -> bool:
        """Check if all frames share the same perturbation."""
        if len(self.shared_frames) != 1:
            return False
        return len(self.shared_frames[0]) == self.input_shape[1]
    
    @property
    def n_independent_spatial(self) -> int:
        """Number of independent spatial perturbation variables."""
        C, D, H, W = self.input_shape
        if self.all_frames_shared:
            return C * H * W  # One δ per spatial location (shared across time)
        else:
            # Count independent variables
            n_groups = len(self.shared_frames)
            return C * H * W * n_groups


# =============================================================================
# Closed-Form IBP Verifier
# =============================================================================

class ClosedFormIBPVerifier(BaseVerifier):
    """
    Verifier using closed-form first layer bounds + IBP propagation.
    
    This verifier combines:
    1. Analytical (closed-form) bounds on first Conv3D layer (EXACT)
    2. Interval Bound Propagation through subsequent layers (SOUND)
    
    Both stages operate in L∞ norm space, ensuring consistency.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize the ClosedFormIBPVerifier.
        
        Args:
            model: Neural network model to verify
            device: Computation device ('cuda' or 'cpu')
        """
        super().__init__(model)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = model.to(self.device)
    
    # =========================================================================
    # Closed-Form First Layer Bounds
    # =========================================================================
    
    def closed_form_conv3d_bounds(
        self,
        input_tensor: torch.Tensor,
        conv_layer: nn.Conv3d,
        epsilon: float,
        shared_frames: List[List[int]],
    ) -> LayerBounds:
        """
        Compute EXACT bounds on first Conv3D layer output using closed-form analytical solution.
        
        For Conv3D with shared temporal perturbations:
            max z = z_clean + ε × Σ |W_eff|
            min z = z_clean - ε × Σ |W_eff|
        
        Where W_eff aggregates weights across temporal dimension for shared frames.
        
        Args:
            input_tensor: Input tensor of shape (C, D, H, W) or (B, C, D, H, W)
            conv_layer: First Conv3d layer
            epsilon: Perturbation budget
            shared_frames: Frame groups with shared perturbations
            
        Returns:
            LayerBounds containing lower and upper bounds on Conv3D output
        """
        # Handle input dimensions
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        B, C, D, H, W = input_tensor.shape
        input_tensor = input_tensor.to(self.device)
        
        # Get layer parameters
        weights = conv_layer.weight.to(self.device)  # (out_c, in_c, kd, kh, kw)
        bias = conv_layer.bias.to(self.device) if conv_layer.bias is not None else None
        
        out_c, in_c, kd, kh, kw = weights.shape
        stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
        padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
        
        # Compute clean output: z(x)
        z_clean = F.conv3d(input_tensor, weights, bias, stride=stride, padding=padding)
        
        # Compute output spatial dimensions
        out_D = (D + 2 * padding - kd) // stride + 1
        out_H = (H + 2 * padding - kh) // stride + 1
        out_W = (W + 2 * padding - kw) // stride + 1
        
        # Check if all frames share the same perturbation
        all_shared = (len(shared_frames) == 1 and len(shared_frames[0]) == D)
        
        if all_shared:
            # OPTIMIZED PATH: All frames share same perturbation
            # Sum weights over temporal kernel dimension
            w_temporal_sum = weights.sum(dim=2)  # (out_c, in_c, kh, kw)
            
            # Create perturbation indicator
            spatial_pert = torch.ones(B, in_c, H, W, device=self.device) * epsilon
            
            # Compute radius via 2D convolution with absolute weights
            radius_2d = F.conv2d(
                spatial_pert,
                w_temporal_sum.abs(),
                bias=None,
                stride=stride,
                padding=padding
            )  # Shape: (B, out_c, out_H, out_W)
            
            # Expand to match output shape
            radius = radius_2d.unsqueeze(2).expand(-1, -1, out_D, -1, -1)
        else:
            # GENERAL PATH: Handle arbitrary shared frame configurations
            radius = self._compute_general_analytical_radius(
                input_tensor.shape,
                weights,
                epsilon,
                shared_frames,
                conv_layer,
            )
        
        # Compute final bounds
        lower_bounds = z_clean - radius
        upper_bounds = z_clean + radius
        
        return LayerBounds(lower=lower_bounds, upper=upper_bounds)
    
    def _compute_general_analytical_radius(
        self,
        input_shape: Tuple[int, ...],
        weights: torch.Tensor,
        epsilon: float,
        shared_frames: List[List[int]],
        conv_layer: nn.Conv3d,
    ) -> torch.Tensor:
        """
        Compute analytical radius for general (non-all-shared) frame configurations.
        """
        B, C, D, H, W = input_shape
        out_c, in_c, kd, kh, kw = weights.shape
        stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
        padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
        
        out_D = (D + 2 * padding - kd) // stride + 1
        out_H = (H + 2 * padding - kh) // stride + 1
        out_W = (W + 2 * padding - kw) // stride + 1
        
        # Initialize radius tensor
        radius = torch.zeros(B, out_c, out_D, out_H, out_W, device=self.device)
        
        # For each shared group, compute its contribution to the radius
        for g_idx, group in enumerate(shared_frames):
            group_set = set(group)
            
            # For each output temporal position
            for od in range(out_D):
                # Build mask for which kernel depth positions contribute to this group
                kd_mask = torch.zeros(kd, device=self.device)
                for kd_idx in range(kd):
                    input_d = od * stride + kd_idx - padding
                    if 0 <= input_d < D and input_d in group_set:
                        kd_mask[kd_idx] = 1.0
                
                if kd_mask.sum() == 0:
                    continue
                
                # Compute effective weight by summing masked kernel positions
                w_masked = weights * kd_mask.view(1, 1, kd, 1, 1)
                w_eff = w_masked.sum(dim=2)  # (out_c, in_c, kh, kw)
                
                # Compute radius contribution via 2D convolution
                spatial_pert = torch.ones(B, in_c, H, W, device=self.device) * epsilon
                radius_2d = F.conv2d(
                    spatial_pert,
                    w_eff.abs(),
                    bias=None,
                    stride=stride,
                    padding=padding
                )
                
                # Add to radius at this output depth
                radius[:, :, od, :, :] += radius_2d
        
        return radius
    
    def closed_form_conv3d_bounds_patch(
        self,
        input_tensor: torch.Tensor,
        conv_layer: nn.Conv3d,
        epsilon: float,
        frame_patch_locations: Dict[int, Tuple[int, int, int, int]],
    ) -> LayerBounds:
        """
        Compute EXACT bounds on first Conv3D layer output for adversarial patch constraints.
        
        Args:
            input_tensor: Input tensor of shape (C, D, H, W) or (B, C, D, H, W)
            conv_layer: First Conv3d layer
            epsilon: Perturbation budget (L∞ norm)
            frame_patch_locations: Dictionary mapping frame index to patch bounding boxes
                                   Each box is (x1, y1, x2, y2) in pixel coordinates
            
        Returns:
            LayerBounds containing lower and upper bounds on Conv3D output
        """
        # Handle input dimensions
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        B, C, D, H, W = input_tensor.shape
        input_tensor = input_tensor.to(self.device)
        
        # Get layer parameters
        weights = conv_layer.weight.to(self.device)
        bias = conv_layer.bias.to(self.device) if conv_layer.bias is not None else None
        
        out_c, in_c, kd, kh, kw = weights.shape
        
        # Handle stride and padding
        if isinstance(conv_layer.stride, int):
            stride = (conv_layer.stride, conv_layer.stride, conv_layer.stride)
        else:
            stride = conv_layer.stride
            
        if isinstance(conv_layer.padding, int):
            padding = (conv_layer.padding, conv_layer.padding, conv_layer.padding)
        else:
            padding = conv_layer.padding
        
        # Compute clean output
        z_clean = F.conv3d(input_tensor, weights, bias, stride=stride, padding=padding)
        
        # Create patch mask
        mask = torch.zeros(D, H, W, device=self.device)
        
        for frame_idx, (x1, y1, x2, y2) in frame_patch_locations.items():
            if 0 <= frame_idx < D:
                y1_clamped = max(0, min(int(y1), H))
                y2_clamped = max(0, min(int(y2), H))
                x1_clamped = max(0, min(int(x1), W))
                x2_clamped = max(0, min(int(x2), W))
                mask[frame_idx, y1_clamped:y2_clamped, x1_clamped:x2_clamped] = 1.0
        
        # Compute radius via 3D convolution with absolute weights
        mask_5d = mask.unsqueeze(0).unsqueeze(0)
        mask_expanded = mask_5d.expand(1, in_c, D, H, W) * epsilon
        
        W_abs = weights.abs()
        
        radius = F.conv3d(
            mask_expanded,
            W_abs,
            bias=None,
            stride=stride,
            padding=padding
        )
        
        radius = radius.expand(B, -1, -1, -1, -1)
        
        lower_bounds = z_clean - radius
        upper_bounds = z_clean + radius
        
        return LayerBounds(lower=lower_bounds, upper=upper_bounds)
    
    # =========================================================================
    # IBP Propagation
    # =========================================================================
    
    def compute_spectral_norm_conv3d(self, conv_layer: nn.Conv3d) -> float:
        """Compute spectral norm of a Conv3d layer (diagnostic only)."""
        weight = conv_layer.weight.detach()
        out_c, in_c, kd, kh, kw = weight.shape
        weight_2d = weight.reshape(out_c, -1)
        
        try:
            singular_values = torch.linalg.svdvals(weight_2d)
            return singular_values[0].item()
        except Exception:
            return torch.norm(weight_2d, p=2).item()
    
    def compute_spectral_norm_linear(self, linear_layer: nn.Linear) -> float:
        """Compute spectral norm of a Linear layer (diagnostic only)."""
        weight = linear_layer.weight.detach()
        
        try:
            singular_values = torch.linalg.svdvals(weight)
            return singular_values[0].item()
        except Exception:
            return torch.norm(weight, p=2).item()
    
    def ibp_propagate_bounds(
        self,
        first_layer_bounds: LayerBounds,
        model: nn.Module,
        start_from_layer: int = 1,
    ) -> Tuple[LayerBounds, float]:
        """
        Propagate bounds through subsequent layers using Interval Bound Propagation (IBP).
        
        IBP propagates coordinate-wise interval bounds through neural network layers,
        operating entirely in L∞ (coordinate-wise) space.
        
        Args:
            first_layer_bounds: Bounds from analytical first layer computation
            model: Neural network model
            start_from_layer: Layer index to start propagation (0 = first conv)
            
        Returns:
            (final_bounds, cumulative_spectral_norm)
        """
        lower = first_layer_bounds.lower
        upper = first_layer_bounds.upper
        
        if lower.dim() == 4:
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
        
        lower = lower.to(self.device)
        upper = upper.to(self.device)
        
        # Center-radius representation for IBP
        center = (lower + upper) / 2
        radius = (upper - lower) / 2
        
        # Get layers
        layers = list(model.children())
        num_layers = len(layers)
        
        # Cumulative spectral norm (diagnostic only)
        cumulative_spectral_norm = 1.0
        
        for i, layer in enumerate(layers):
            if i < start_from_layer:
                if i == start_from_layer - 1:
                    lower_out = F.relu(center - radius)
                    upper_out = F.relu(center + radius)
                    center = (lower_out + upper_out) / 2
                    radius = (upper_out - lower_out) / 2
                continue
            
            if isinstance(layer, nn.Conv3d):
                spectral_norm = self.compute_spectral_norm_conv3d(layer)
                cumulative_spectral_norm *= spectral_norm
                
                center = F.conv3d(
                    center, layer.weight, bias=layer.bias,
                    stride=layer.stride, padding=layer.padding
                )
                
                W_abs = layer.weight.abs()
                radius = F.conv3d(
                    radius, W_abs, bias=None,
                    stride=layer.stride, padding=layer.padding
                )
                
            elif isinstance(layer, nn.Linear):
                if center.dim() > 2:
                    center = center.view(center.size(0), -1)
                    radius = radius.view(radius.size(0), -1)
                
                spectral_norm = self.compute_spectral_norm_linear(layer)
                cumulative_spectral_norm *= spectral_norm
                
                center = F.linear(center, layer.weight, bias=layer.bias)
                
                W_abs = layer.weight.abs()
                radius = F.linear(radius, W_abs, bias=None)
            
            elif isinstance(layer, nn.BatchNorm3d) or isinstance(layer, nn.BatchNorm1d):
                center, radius = self._ibp_propagate_batchnorm(center, radius, layer)
            
            elif isinstance(layer, nn.Dropout):
                # Dropout is identity during evaluation
                pass
            
            # Apply ReLU (except on final layer)
            if i != num_layers - 1 and isinstance(layer, (nn.Conv3d, nn.Linear)):
                cumulative_spectral_norm *= 1.0  # ReLU is 1-Lipschitz
                
                lower_tmp = center - radius
                upper_tmp = center + radius
                
                lower_out = F.relu(lower_tmp)
                upper_out = F.relu(upper_tmp)
                
                center = (lower_out + upper_out) / 2
                radius = (upper_out - lower_out) / 2
        
        final_lower = center - radius
        final_upper = center + radius
        
        return LayerBounds(lower=final_lower, upper=final_upper), cumulative_spectral_norm
    
    def _ibp_propagate_batchnorm(
        self,
        center: torch.Tensor,
        radius: torch.Tensor,
        bn_layer: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through batch normalization layer."""
        running_mean = bn_layer.running_mean
        running_var = bn_layer.running_var
        weight = bn_layer.weight
        bias = bn_layer.bias
        eps = bn_layer.eps
        
        # Compute scale factor
        std = torch.sqrt(running_var + eps)
        scale = weight / std
        
        # Handle different tensor dimensions
        if center.dim() == 5:  # BatchNorm3d: (B, C, D, H, W)
            scale = scale.view(1, -1, 1, 1, 1)
            running_mean = running_mean.view(1, -1, 1, 1, 1)
            bias = bias.view(1, -1, 1, 1, 1)
        elif center.dim() == 2:  # BatchNorm1d: (B, C)
            scale = scale.view(1, -1)
            running_mean = running_mean.view(1, -1)
            bias = bias.view(1, -1)
        
        # Propagate center
        center_out = scale * (center - running_mean) + bias
        
        # Propagate radius (use absolute scale)
        radius_out = radius * scale.abs()
        
        return center_out, radius_out
    
    # =========================================================================
    # Main Verification Interface
    # =========================================================================
    
    def verify(
        self,
        input_tensor: torch.Tensor,
        epsilon: float,
        shared_frames: List[List[int]],
        true_label: Optional[int] = None,
    ) -> Dict:
        """
        Complete verification pipeline: Analytical GPU bounds + IBP propagation.
        
        Args:
            input_tensor: Input tensor (C, D, H, W) or (B, C, D, H, W)
            epsilon: Perturbation budget (L∞ norm)
            shared_frames: Frame groups with shared perturbations
            true_label: True class label (optional, for robustness checking)
            
        Returns:
            Dictionary containing:
                - lower_bounds: Lower bounds on output logits
                - upper_bounds: Upper bounds on output logits
                - runtime: Verification time in seconds
                - spectral_norm: Cumulative spectral norm (diagnostic)
                - verified: Whether robustness is verified (if true_label provided)
        """
        start_time = time.time()
        
        # Ensure input has correct shape
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Get first conv layer
        first_conv = self._get_first_conv_layer()
        
        # Step 1: Analytical bounds on first Conv3D layer (EXACT)
        first_layer_bounds = self.closed_form_conv3d_bounds(
            input_tensor,
            first_conv,
            epsilon,
            shared_frames,
        )
        
        # Step 2: IBP propagation through remaining layers (SOUND)
        final_bounds, spectral_norm = self.ibp_propagate_bounds(
            first_layer_bounds,
            self.model,
            start_from_layer=1,
        )
        
        runtime = time.time() - start_time
        
        result = {
            'lower_bounds': final_bounds.lower,
            'upper_bounds': final_bounds.upper,
            'runtime': runtime,
            'spectral_norm': spectral_norm,
        }
        
        if true_label is not None:
            result['verified'] = self.check_robustness(
                final_bounds.lower, final_bounds.upper, true_label
            )
        
        return result
    
    def verify_patch(
        self,
        input_tensor: torch.Tensor,
        epsilon: float,
        frame_patch_locations: Dict[int, Tuple[int, int, int, int]],
        true_label: Optional[int] = None,
    ) -> Dict:
        """
        Verification for adversarial patch perturbations.
        
        Args:
            input_tensor: Input tensor (C, D, H, W) or (B, C, D, H, W)
            epsilon: Perturbation budget (L∞ norm)
            frame_patch_locations: Dictionary mapping frame index to patch bounding boxes
            true_label: True class label (optional)
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        first_conv = self._get_first_conv_layer()
        
        # Step 1: Analytical bounds on first Conv3D layer with patch constraints
        first_layer_bounds = self.closed_form_conv3d_bounds_patch(
            input_tensor,
            first_conv,
            epsilon,
            frame_patch_locations,
        )
        
        # Step 2: IBP propagation through remaining layers
        final_bounds, spectral_norm = self.ibp_propagate_bounds(
            first_layer_bounds,
            self.model,
            start_from_layer=1,
        )
        
        runtime = time.time() - start_time
        
        result = {
            'lower_bounds': final_bounds.lower,
            'upper_bounds': final_bounds.upper,
            'runtime': runtime,
            'spectral_norm': spectral_norm,
        }
        
        if true_label is not None:
            result['verified'] = self.check_robustness(
                final_bounds.lower, final_bounds.upper, true_label
            )
        
        return result
    
    def _get_first_conv_layer(self) -> nn.Conv3d:
        """Get the first Conv3d layer from the model."""
        for module in self.model.modules():
            if isinstance(module, nn.Conv3d):
                return module
        raise ValueError("Model does not have a Conv3d layer")
    
    # =========================================================================
    # Robustness Checking
    # =========================================================================
    
    def check_robustness(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        true_label: int,
    ) -> bool:
        """
        Check if the network is verified robust for a given input.
        
        Robustness criterion:
            lower[true_label] > max(upper[j]) for all j ≠ true_label
        
        Args:
            lower_bounds: Lower bounds on output logits
            upper_bounds: Upper bounds on output logits
            true_label: True class label
            
        Returns:
            True if robustness is verified, False otherwise
        """
        lower = lower_bounds.squeeze()
        upper = upper_bounds.squeeze()
        
        num_classes = len(lower)
        
        for j in range(num_classes):
            if j != true_label:
                if lower[true_label] <= upper[j]:
                    return False
        return True
    
    def check_robustness_batch(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch robustness checking.
        
        Args:
            lower_bounds: Lower bounds on output logits (B, num_classes)
            upper_bounds: Upper bounds on output logits (B, num_classes)
            true_labels: True class labels (B,)
            
        Returns:
            Boolean tensor of shape (B,) indicating verified robustness
        """
        B, num_classes = lower_bounds.shape
        
        # Get lower bound for true class
        lower_true = lower_bounds.gather(1, true_labels.unsqueeze(1)).squeeze(1)
        
        # Create mask to exclude true class
        mask = torch.ones(B, num_classes, device=lower_bounds.device, dtype=torch.bool)
        mask.scatter_(1, true_labels.unsqueeze(1), False)
        
        # Get max upper bound for other classes
        upper_other = upper_bounds.masked_fill(~mask, float('-inf'))
        max_upper_other = upper_other.max(dim=1).values
        
        # Check robustness
        verified = lower_true > max_upper_other
        
        return verified

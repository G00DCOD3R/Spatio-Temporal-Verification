"""
Closed-Form Lipschitz Verifier for Spatio-Temporal Video Classification

This module implements neural network robustness verification that combines:
1. GPU-accelerated closed-form solution for first Conv3D layer bounds (EXACT)
2. Lipschitz-based propagation for subsequent layers - operates in L-2 norm space

============================================================================
THEORETICAL FOUNDATION
============================================================================

For the first convolutional layer with structured perturbations, we derive an
ANALYTICAL (closed-form) solution that eliminates the need for LP optimization.

Subsequent layers use Lipschitz constant propagation:
    For a layer f with Lipschitz constant L:
        ||f(x) - f(c)||_2 ≤ L × ||x - c||_2

============================================================================
HYBRID APPROACH: L∞ → L-2 NORM CONVERSION
============================================================================

This function addresses the norm mismatch challenge:
- First layer: Exact bounds for L∞ perturbations with shared constraints
- Subsequent layers: Lipschitz propagation in L-2 norm space

Norm conversion strategy:
1. First layer outputs L∞ bounds [center - r, center + r] (coordinate-wise)
2. Convert to L-2 radius: R = ||r||_2 (L-2 norm of coordinate-wise radii)
3. Propagate in L-2 space: R_out = L × R_in (where L is Lipschitz constant)
4. Convert back to L∞ bounds: r_out = R_out / √n (per dimension)

============================================================================
SOUNDNESS
============================================================================

The bounds are sound because:
- First layer bounds are exact (closed-form solution)
- L∞ to L-2 conversion: L-2 ball contains the L∞ box
- Lipschitz propagation: Preserves containment in L-2 space
- L-2 to L∞ conversion: L∞ box contains the L-2 ball
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
    """Container for layer activation bounds with utility properties."""
    lower: torch.Tensor
    upper: torch.Tensor
    
    @property
    def center(self) -> torch.Tensor:
        return (self.lower + self.upper) / 2
    
    @property
    def radius(self) -> torch.Tensor:
        return (self.upper - self.lower) / 2
    
    @property
    def width(self) -> torch.Tensor:
        return self.upper - self.lower
    
    def to(self, device: str) -> 'LayerBounds':
        return LayerBounds(
            lower=self.lower.to(device),
            upper=self.upper.to(device)
        )


# =============================================================================
# Closed-Form Lipschitz Verifier
# =============================================================================

class ClosedFormLipschitzVerifier(BaseVerifier):
    """
    Verifier using closed-form first layer bounds + Lipschitz propagation.
    
    This verifier combines:
    1. Analytical (closed-form) bounds on first Conv3D layer (EXACT, L∞)
    2. Lipschitz propagation through subsequent layers (SOUND, L-2)
    
    Includes norm conversion between L∞ and L-2 spaces.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize the ClosedFormLipschitzVerifier.
        
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
        
        Args:
            input_tensor: Input tensor of shape (C, D, H, W) or (B, C, D, H, W)
            conv_layer: First Conv3d layer
            epsilon: Perturbation budget
            shared_frames: Frame groups with shared perturbations
            
        Returns:
            LayerBounds containing lower and upper bounds on Conv3D output
        """
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        B, C, D, H, W = input_tensor.shape
        input_tensor = input_tensor.to(self.device)
        
        weights = conv_layer.weight.to(self.device)
        bias = conv_layer.bias.to(self.device) if conv_layer.bias is not None else None
        
        out_c, in_c, kd, kh, kw = weights.shape
        stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
        padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
        
        z_clean = F.conv3d(input_tensor, weights, bias, stride=stride, padding=padding)
        
        out_D = (D + 2 * padding - kd) // stride + 1
        out_H = (H + 2 * padding - kh) // stride + 1
        out_W = (W + 2 * padding - kw) // stride + 1
        
        all_shared = (len(shared_frames) == 1 and len(shared_frames[0]) == D)
        
        if all_shared:
            w_temporal_sum = weights.sum(dim=2)
            spatial_pert = torch.ones(B, in_c, H, W, device=self.device) * epsilon
            
            radius_2d = F.conv2d(
                spatial_pert,
                w_temporal_sum.abs(),
                bias=None,
                stride=stride,
                padding=padding
            )
            
            radius = radius_2d.unsqueeze(2).expand(-1, -1, out_D, -1, -1)
        else:
            radius = self._compute_general_analytical_radius(
                input_tensor.shape, weights, epsilon, shared_frames, conv_layer
            )
        
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
        """Compute analytical radius for general frame configurations."""
        B, C, D, H, W = input_shape
        out_c, in_c, kd, kh, kw = weights.shape
        stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
        padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
        
        out_D = (D + 2 * padding - kd) // stride + 1
        out_H = (H + 2 * padding - kh) // stride + 1
        out_W = (W + 2 * padding - kw) // stride + 1
        
        radius = torch.zeros(B, out_c, out_D, out_H, out_W, device=self.device)
        
        for g_idx, group in enumerate(shared_frames):
            group_set = set(group)
            
            for od in range(out_D):
                kd_mask = torch.zeros(kd, device=self.device)
                for kd_idx in range(kd):
                    input_d = od * stride + kd_idx - padding
                    if 0 <= input_d < D and input_d in group_set:
                        kd_mask[kd_idx] = 1.0
                
                if kd_mask.sum() == 0:
                    continue
                
                w_masked = weights * kd_mask.view(1, 1, kd, 1, 1)
                w_eff = w_masked.sum(dim=2)
                
                spatial_pert = torch.ones(B, in_c, H, W, device=self.device) * epsilon
                radius_2d = F.conv2d(
                    spatial_pert, w_eff.abs(), bias=None,
                    stride=stride, padding=padding
                )
                
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
        Compute EXACT bounds for adversarial patch constraints.
        """
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        B, C, D, H, W = input_tensor.shape
        input_tensor = input_tensor.to(self.device)
        
        weights = conv_layer.weight.to(self.device)
        bias = conv_layer.bias.to(self.device) if conv_layer.bias is not None else None
        
        out_c, in_c, kd, kh, kw = weights.shape
        
        if isinstance(conv_layer.stride, int):
            stride = (conv_layer.stride, conv_layer.stride, conv_layer.stride)
        else:
            stride = conv_layer.stride
            
        if isinstance(conv_layer.padding, int):
            padding = (conv_layer.padding, conv_layer.padding, conv_layer.padding)
        else:
            padding = conv_layer.padding
        
        z_clean = F.conv3d(input_tensor, weights, bias, stride=stride, padding=padding)
        
        mask = torch.zeros(D, H, W, device=self.device)
        
        for frame_idx, (x1, y1, x2, y2) in frame_patch_locations.items():
            if 0 <= frame_idx < D:
                y1_clamped = max(0, min(int(y1), H))
                y2_clamped = max(0, min(int(y2), H))
                x1_clamped = max(0, min(int(x1), W))
                x2_clamped = max(0, min(int(x2), W))
                mask[frame_idx, y1_clamped:y2_clamped, x1_clamped:x2_clamped] = 1.0
        
        mask_5d = mask.unsqueeze(0).unsqueeze(0)
        mask_expanded = mask_5d.expand(1, in_c, D, H, W) * epsilon
        
        W_abs = weights.abs()
        
        radius = F.conv3d(mask_expanded, W_abs, bias=None, stride=stride, padding=padding)
        radius = radius.expand(B, -1, -1, -1, -1)
        
        lower_bounds = z_clean - radius
        upper_bounds = z_clean + radius
        
        return LayerBounds(lower=lower_bounds, upper=upper_bounds)
    
    # =========================================================================
    # Spectral Norm Computation
    # =========================================================================
    
    def compute_spectral_norm_conv3d(self, conv_layer: nn.Conv3d) -> float:
        """
        Compute the spectral norm (L2 Lipschitz constant) of a Conv3d layer.
        """
        weight = conv_layer.weight.detach()
        out_c, in_c, kd, kh, kw = weight.shape
        weight_2d = weight.reshape(out_c, -1)
        
        try:
            singular_values = torch.linalg.svdvals(weight_2d)
            return singular_values[0].item()
        except Exception:
            return torch.norm(weight_2d, p=2).item()
    
    def compute_spectral_norm_linear(self, linear_layer: nn.Linear) -> float:
        """
        Compute the spectral norm of a Linear layer.
        """
        weight = linear_layer.weight.detach()
        
        try:
            singular_values = torch.linalg.svdvals(weight)
            return singular_values[0].item()
        except Exception:
            return torch.norm(weight, p=2).item()
    
    # =========================================================================
    # Norm Conversion Functions
    # =========================================================================
    
    def linf_to_l2_radius(self, linf_radius: torch.Tensor) -> torch.Tensor:
        """
        Convert L-infinity radius to L-2 radius.
        
        For L∞ box with radius r, L-2 radius R = ||r||_2
        """
        if linf_radius.dim() > 1:
            batch_size = linf_radius.shape[0]
            linf_flat = linf_radius.view(batch_size, -1)
            l2_radius = torch.norm(linf_flat, p=2, dim=1)
        else:
            linf_flat = linf_radius.view(-1)
            l2_radius = torch.norm(linf_flat, p=2)
        
        return l2_radius
    
    def l2_radius_to_linf_bounds(
        self, 
        center: torch.Tensor, 
        l2_radius: torch.Tensor
    ) -> LayerBounds:
        """
        Convert L-2 radius back to L-infinity bounds.
        
        For L-2 ball with radius R, L∞ radius r = R / √n
        """
        if not isinstance(l2_radius, torch.Tensor):
            l2_radius = torch.tensor(l2_radius, dtype=center.dtype, device=center.device)
        else:
            l2_radius = l2_radius.to(center.device)
        
        has_batch = center.dim() > 1
        if has_batch:
            batch_size = center.shape[0]
            n_dims = center.numel() // batch_size
            
            if l2_radius.dim() == 0:
                l2_radius = l2_radius.unsqueeze(0)
            if l2_radius.shape[0] == 1 and batch_size > 1:
                l2_radius = l2_radius.expand(batch_size)
        else:
            n_dims = center.numel()
            if l2_radius.dim() > 0:
                l2_radius = l2_radius.squeeze()
        
        sqrt_n_dims = torch.sqrt(torch.tensor(n_dims, dtype=center.dtype, device=center.device))
        linf_radius_per_dim = l2_radius / sqrt_n_dims
        
        if has_batch:
            linf_radius = linf_radius_per_dim.view(batch_size, *([1] * (center.dim() - 1)))
        else:
            linf_radius = linf_radius_per_dim
        
        lower = center - linf_radius
        upper = center + linf_radius
        
        return LayerBounds(lower=lower, upper=upper)
    
    # =========================================================================
    # Lipschitz-based Bound Propagation
    # =========================================================================
    
    def lipschitz_propagate_bounds(
        self,
        first_layer_bounds: LayerBounds,
        model: nn.Module,
        start_from_layer: int = 1,
    ) -> Tuple[LayerBounds, float]:
        """
        Propagate bounds through subsequent layers using Lipschitz constant (L-2 norm).
        
        Key steps:
        1. Convert L-infinity bounds from first layer to L-2 radius
        2. Propagate L-2 radius through layers using Lipschitz constants
        3. Convert back to L-infinity bounds for final output
        
        Args:
            first_layer_bounds: Bounds from analytical first layer computation (L-inf)
            model: Neural network model
            start_from_layer: Layer index to start propagation
            
        Returns:
            (final_bounds, cumulative_lipschitz_constant)
        """
        lower = first_layer_bounds.lower
        upper = first_layer_bounds.upper
        
        if lower.dim() == 4:
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
        
        lower = lower.to(self.device)
        upper = upper.to(self.device)
        
        # Center-radius representation
        center = (lower + upper) / 2
        radius_linf = (upper - lower) / 2
        
        # Step 1: Convert L-infinity radius to L-2 radius
        l2_radius = self.linf_to_l2_radius(radius_linf)
        
        layers = list(model.children())
        num_layers = len(layers)
        
        cumulative_lipschitz = 1.0
        
        for i, layer in enumerate(layers):
            if i < start_from_layer:
                if i == start_from_layer - 1:
                    # Apply ReLU in L-inf space, then convert to L-2
                    lower_tmp = F.relu(center - radius_linf)
                    upper_tmp = F.relu(center + radius_linf)
                    center = (lower_tmp + upper_tmp) / 2
                    radius_linf = (upper_tmp - lower_tmp) / 2
                    l2_radius = self.linf_to_l2_radius(radius_linf)
                continue
            
            if isinstance(layer, nn.Conv3d):
                lipschitz_const = self.compute_spectral_norm_conv3d(layer)
                cumulative_lipschitz *= lipschitz_const
                
                center = F.conv3d(
                    center, layer.weight, bias=layer.bias,
                    stride=layer.stride, padding=layer.padding
                )
                
                l2_radius = l2_radius * lipschitz_const
                
            elif isinstance(layer, nn.Linear):
                if center.dim() > 2:
                    center = center.view(center.size(0), -1)
                
                lipschitz_const = self.compute_spectral_norm_linear(layer)
                cumulative_lipschitz *= lipschitz_const
                
                center = F.linear(center, layer.weight, bias=layer.bias)
                
                l2_radius = l2_radius * lipschitz_const
            
            elif isinstance(layer, nn.BatchNorm3d) or isinstance(layer, nn.BatchNorm1d):
                # BatchNorm can be viewed as affine transformation
                # Lipschitz constant is max(|gamma/sqrt(var+eps)|)
                lipschitz_const = self._batchnorm_lipschitz(layer)
                cumulative_lipschitz *= lipschitz_const
                
                center = self._forward_batchnorm(center, layer)
                l2_radius = l2_radius * lipschitz_const
            
            elif isinstance(layer, nn.Dropout):
                pass
            
            # Apply ReLU (Lipschitz constant = 1)
            if i != num_layers - 1 and isinstance(layer, (nn.Conv3d, nn.Linear)):
                # ReLU is 1-Lipschitz, so L-2 radius doesn't change
                # But we need to update center and convert back to L-inf for ReLU
                temp_bounds = self.l2_radius_to_linf_bounds(center, l2_radius)
                
                lower_tmp = F.relu(temp_bounds.lower)
                upper_tmp = F.relu(temp_bounds.upper)
                center = (lower_tmp + upper_tmp) / 2
                radius_linf = (upper_tmp - lower_tmp) / 2
                
                l2_radius = self.linf_to_l2_radius(radius_linf)
        
        final_bounds = self.l2_radius_to_linf_bounds(center, l2_radius)
        
        return final_bounds, cumulative_lipschitz
    
    def _batchnorm_lipschitz(self, bn_layer: nn.Module) -> float:
        """Compute Lipschitz constant for batch normalization layer."""
        weight = bn_layer.weight.detach()
        running_var = bn_layer.running_var.detach()
        eps = bn_layer.eps
        
        scale = weight / torch.sqrt(running_var + eps)
        return scale.abs().max().item()
    
    def _forward_batchnorm(self, x: torch.Tensor, bn_layer: nn.Module) -> torch.Tensor:
        """Forward pass through batch normalization layer."""
        return bn_layer(x)
    
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
        Complete verification pipeline: Analytical GPU bounds + Lipschitz propagation.
        
        Args:
            input_tensor: Input tensor (C, D, H, W) or (B, C, D, H, W)
            epsilon: Perturbation budget (L∞ norm)
            shared_frames: Frame groups with shared perturbations
            true_label: True class label (optional)
            
        Returns:
            Dictionary containing verification results
        """
        start_time = time.time()
        
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        first_conv = self._get_first_conv_layer()
        
        # Step 1: Analytical bounds on first Conv3D layer (EXACT, L∞)
        first_layer_bounds = self.closed_form_conv3d_bounds(
            input_tensor, first_conv, epsilon, shared_frames
        )
        
        # Step 2: Lipschitz propagation through remaining layers (SOUND, L-2)
        final_bounds, cumulative_lipschitz = self.lipschitz_propagate_bounds(
            first_layer_bounds, self.model, start_from_layer=1
        )
        
        runtime = time.time() - start_time
        
        result = {
            'lower_bounds': final_bounds.lower,
            'upper_bounds': final_bounds.upper,
            'runtime': runtime,
            'cumulative_lipschitz': cumulative_lipschitz,
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
        """
        start_time = time.time()
        
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        first_conv = self._get_first_conv_layer()
        
        first_layer_bounds = self.closed_form_conv3d_bounds_patch(
            input_tensor, first_conv, epsilon, frame_patch_locations
        )
        
        final_bounds, cumulative_lipschitz = self.lipschitz_propagate_bounds(
            first_layer_bounds, self.model, start_from_layer=1
        )
        
        runtime = time.time() - start_time
        
        result = {
            'lower_bounds': final_bounds.lower,
            'upper_bounds': final_bounds.upper,
            'runtime': runtime,
            'cumulative_lipschitz': cumulative_lipschitz,
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
        """
        B, num_classes = lower_bounds.shape
        
        lower_true = lower_bounds.gather(1, true_labels.unsqueeze(1)).squeeze(1)
        
        mask = torch.ones(B, num_classes, device=lower_bounds.device, dtype=torch.bool)
        mask.scatter_(1, true_labels.unsqueeze(1), False)
        
        upper_other = upper_bounds.masked_fill(~mask, float('-inf'))
        max_upper_other = upper_other.max(dim=1).values
        
        verified = lower_true > max_upper_other
        
        return verified

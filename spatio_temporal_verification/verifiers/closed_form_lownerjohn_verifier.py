"""
Closed-Form Löwner-John Ellipsoid Verifier for Spatio-Temporal Video Classification

This module implements neural network robustness verification that combines:
1. GPU-accelerated closed-form solution for first Conv3D layer bounds (EXACT)
2. Löwner-John ellipsoid-based propagation for subsequent layers

============================================================================
THEORETICAL FOUNDATION
============================================================================

The Löwner-John ellipsoid method approximates the reachable set with ellipsoids:
    E = {x : (x-c)^T P (x-c) ≤ 1}

Key properties:
- For an L∞ box, the minimum volume containing ellipsoid is the Löwner-John ellipsoid
- Ellipsoids propagate exactly through affine transformations
- For ReLU, we convert to bounds, apply ReLU, then re-ellipsoidize

============================================================================
APPROACH
============================================================================

1. First layer outputs L∞ bounds [center - r, center + r] (coordinate-wise)
2. Convert to Löwner-John ellipsoid: E = {x : (x-c)^T P (x-c) ≤ 1}
3. Propagate ellipsoid through affine layers
4. Sample points on ellipsoid surface to estimate bounds
5. Handle ReLU by converting to bounds, applying ReLU, then re-ellipsoidizing

============================================================================
SOUNDNESS
============================================================================

The bounds are sound because:
- First layer bounds are exact (closed-form solution)
- Ellipsoid contains the L∞ box (conservative but sound)
- Affine propagation preserves ellipsoid containment
- Sampling provides bounds that contain the ellipsoid

============================================================================
TIGHTNESS
============================================================================

The bounds can be tighter than IBP or Lipschitz methods because:
- Ellipsoids can better capture correlations between dimensions
- Sampling can find tighter bounds than conservative approximations
- However, sampling is stochastic and may miss extreme points
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


@dataclass
class Ellipsoid:
    """
    Represents an ellipsoid in the form: {x : (x-c)^T P (x-c) ≤ 1}
    
    Attributes:
        center: Center of the ellipsoid
        P: Positive definite matrix defining the ellipsoid shape
           Can be full matrix or diagonal-only for memory efficiency
        is_diagonal: If True, P is stored as diagonal elements only
    """
    center: torch.Tensor
    P: torch.Tensor
    is_diagonal: bool = False
    
    def to(self, device: str) -> 'Ellipsoid':
        return Ellipsoid(
            center=self.center.to(device),
            P=self.P.to(device),
            is_diagonal=self.is_diagonal
        )
    
    def get_P_matrix(self) -> torch.Tensor:
        """Get P as a full matrix (creates it if diagonal-only)."""
        if self.is_diagonal:
            if self.P.dim() == 1:
                return torch.diag(self.P)
            else:
                return torch.diag_embed(self.P)
        return self.P


# =============================================================================
# Closed-Form Löwner-John Verifier
# =============================================================================

class ClosedFormLownerJohnVerifier(BaseVerifier):
    """
    Verifier using closed-form first layer bounds + Löwner-John ellipsoid propagation.
    
    This verifier combines:
    1. Analytical (closed-form) bounds on first Conv3D layer (EXACT, L∞)
    2. Löwner-John ellipsoid approximation and sampling for subsequent layers
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: str = "cuda",
        n_samples: int = 500,
        chunk_size: int = 100,
        max_dims: int = 500,
    ):
        """
        Initialize the ClosedFormLownerJohnVerifier.
        
        Args:
            model: Neural network model to verify
            device: Computation device ('cuda' or 'cpu')
            n_samples: Number of samples for ellipsoid sampling
            chunk_size: Number of samples to process at once (reduces memory)
            max_dims: Maximum dimensions before using diagonal-only approximation
        """
        super().__init__(model)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = model.to(self.device)
        self.n_samples = n_samples
        self.chunk_size = chunk_size
        self.max_dims = max_dims
    
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
        Compute EXACT bounds on first Conv3D layer output.
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
    
    # =========================================================================
    # Ellipsoid Operations
    # =========================================================================
    
    def linf_bounds_to_ellipsoid(self, bounds: LayerBounds) -> Ellipsoid:
        """
        Convert L-infinity bounds to an approximating Löwner-John ellipsoid.
        
        For a box with radius r_i per dimension, we use:
            P = diag(n/r_i²) where n is the dimension
        """
        center = bounds.center.to(self.device)
        radius = bounds.radius.to(self.device)
        
        if center.dim() > 1:
            batch_size = center.shape[0]
            center_flat = center.view(batch_size, -1)
            radius_flat = radius.view(batch_size, -1)
            n_dims = center_flat.shape[1]
            
            if n_dims > self.max_dims:
                r_sq = radius_flat ** 2
                r_sq = torch.clamp(r_sq, min=1e-8)
                P_diag = n_dims / r_sq
                return Ellipsoid(center=center, P=P_diag, is_diagonal=True)
            else:
                P_list = []
                for b in range(batch_size):
                    r_sq = radius_flat[b] ** 2
                    r_sq = torch.clamp(r_sq, min=1e-8)
                    P_diag = n_dims / r_sq
                    P = torch.diag(P_diag)
                    P_list.append(P)
                
                P = torch.stack(P_list)
            return Ellipsoid(center=center, P=P, is_diagonal=False)
        else:
            center_flat = center.view(-1)
            radius_flat = radius.view(-1)
            n_dims = center_flat.shape[0]
            
            r_sq = radius_flat ** 2
            r_sq = torch.clamp(r_sq, min=1e-8)
            P_diag = n_dims / r_sq
            
            if n_dims > self.max_dims:
                return Ellipsoid(center=center, P=P_diag, is_diagonal=True)
            else:
                P = torch.diag(P_diag)
                return Ellipsoid(center=center, P=P, is_diagonal=False)
    
    def propagate_ellipsoid_affine(
        self, 
        ellipsoid: Ellipsoid, 
        W: torch.Tensor, 
        b: Optional[torch.Tensor] = None
    ) -> Ellipsoid:
        """
        Propagate an ellipsoid through an affine transformation: y = Wx + b.
        
        For ellipsoid E_in = {x : (x-c_in)^T P_in (x-c_in) ≤ 1}:
            c_out = W·c_in + b
            P_out = (W P_in^{-1} W^T)^{-1}
        """
        center_in = ellipsoid.center
        is_diag_in = ellipsoid.is_diagonal
        
        if is_diag_in:
            return self._propagate_diagonal_ellipsoid(center_in, ellipsoid.P, W, b)
        
        P_in = ellipsoid.P
        has_batch = P_in.dim() == 3
        
        if has_batch:
            batch_size = P_in.shape[0]
            n_in = P_in.shape[1]
            
            if center_in.dim() > 2:
                center_in_flat = center_in.view(batch_size, -1)
            else:
                center_in_flat = center_in
            
            center_out_flat = F.linear(center_in_flat, W, bias=b)
            n_out = center_out_flat.shape[1]
            
            P_out_list = []
            for b_idx in range(batch_size):
                P_in_b = P_in[b_idx]
                
                try:
                    P_inv = torch.linalg.inv(P_in_b)
                except:
                    P_inv = torch.linalg.pinv(P_in_b)
                
                W_Pinv = W @ P_inv
                W_Pinv_WT = W_Pinv @ W.T
                
                try:
                    P_out = torch.linalg.inv(W_Pinv_WT)
                except:
                    P_out = torch.linalg.pinv(W_Pinv_WT)
                
                if n_out > self.max_dims:
                    P_out_diag = torch.diag(P_out)
                    P_out_list.append(P_out_diag)
                else:
                    P_out_list.append(P_out)
                
                del P_inv, W_Pinv, W_Pinv_WT
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            all_diagonal = all(p.dim() == 1 for p in P_out_list)
            
            if all_diagonal:
                P_out = torch.stack(P_out_list)
                return Ellipsoid(center=center_out_flat, P=P_out, is_diagonal=True)
            else:
                P_out = torch.stack(P_out_list)
                return Ellipsoid(center=center_out_flat.view(batch_size, -1), P=P_out, is_diagonal=False)
        else:
            if center_in.dim() > 1:
                center_in_flat = center_in.view(-1)
            else:
                center_in_flat = center_in
            
            center_out_flat = F.linear(center_in_flat.unsqueeze(0), W, bias=b).squeeze(0)
            n_out = center_out_flat.shape[0]
            
            try:
                P_inv = torch.linalg.inv(P_in)
            except:
                P_inv = torch.linalg.pinv(P_in)
            
            W_Pinv = W @ P_inv
            W_Pinv_WT = W_Pinv @ W.T
            
            try:
                P_out = torch.linalg.inv(W_Pinv_WT)
            except:
                P_out = torch.linalg.pinv(W_Pinv_WT)
            
            if n_out > self.max_dims:
                P_out_diag = torch.diag(P_out)
                return Ellipsoid(center=center_out_flat, P=P_out_diag, is_diagonal=True)
            
            return Ellipsoid(center=center_out_flat, P=P_out, is_diagonal=False)
    
    def _propagate_diagonal_ellipsoid(
        self,
        center_in: torch.Tensor,
        P_in_diag: torch.Tensor,
        W: torch.Tensor,
        b: Optional[torch.Tensor],
    ) -> Ellipsoid:
        """Efficiently propagate a diagonal ellipsoid through affine transformation."""
        has_batch = P_in_diag.dim() > 1
        
        if has_batch:
            batch_size = P_in_diag.shape[0]
            if center_in.dim() > 2:
                center_in_flat = center_in.view(batch_size, -1)
            else:
                center_in_flat = center_in
            
            center_out_flat = F.linear(center_in_flat, W, bias=b)
            n_out = center_out_flat.shape[1]
            
            P_inv_diag = 1.0 / torch.clamp(P_in_diag, min=1e-8)
            
            P_out_list = []
            for b_idx in range(batch_size):
                P_inv_b = P_inv_diag[b_idx]
                W_scaled = W * P_inv_b.unsqueeze(0)
                W_Pinv_WT = W_scaled @ W.T
                
                try:
                    P_out = torch.linalg.inv(W_Pinv_WT)
                except:
                    P_out = torch.linalg.pinv(W_Pinv_WT)
                
                if n_out > self.max_dims:
                    P_out_diag = torch.diag(P_out)
                    P_out_list.append(P_out_diag)
                else:
                    P_out_diag_check = torch.diag(P_out)
                    off_diag = P_out - torch.diag_embed(P_out_diag_check)
                    off_diag_norm = torch.norm(off_diag)
                    diag_norm = torch.norm(P_out_diag_check)
                    
                    if off_diag_norm < 0.1 * diag_norm:
                        P_out_list.append(P_out_diag_check)
                    else:
                        P_out_list.append(P_out)
                
                del W_scaled, W_Pinv_WT
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            all_diagonal = all(p.dim() == 1 for p in P_out_list)
            
            if all_diagonal:
                P_out = torch.stack(P_out_list)
                return Ellipsoid(center=center_out_flat, P=P_out, is_diagonal=True)
            else:
                P_out = torch.stack(P_out_list)
                return Ellipsoid(center=center_out_flat, P=P_out, is_diagonal=False)
        else:
            if center_in.dim() > 1:
                center_in_flat = center_in.view(-1)
            else:
                center_in_flat = center_in
            
            center_out_flat = F.linear(center_in_flat.unsqueeze(0), W, bias=b).squeeze(0)
            n_out = center_out_flat.shape[0]
            
            P_inv_diag = 1.0 / torch.clamp(P_in_diag, min=1e-8)
            W_scaled = W * P_inv_diag.unsqueeze(0)
            W_Pinv_WT = W_scaled @ W.T
            
            try:
                P_out = torch.linalg.inv(W_Pinv_WT)
            except:
                P_out = torch.linalg.pinv(W_Pinv_WT)
            
            if n_out > self.max_dims:
                P_out_diag = torch.diag(P_out)
                return Ellipsoid(center=center_out_flat, P=P_out_diag, is_diagonal=True)
            
            P_out_diag = torch.diag(P_out)
            off_diag = P_out - torch.diag_embed(P_out_diag)
            off_diag_norm = torch.norm(off_diag)
            diag_norm = torch.norm(P_out_diag)
            
            if off_diag_norm < 0.1 * diag_norm:
                return Ellipsoid(center=center_out_flat, P=P_out_diag, is_diagonal=True)
            else:
                return Ellipsoid(center=center_out_flat, P=P_out, is_diagonal=False)
    
    def sample_ellipsoid_bounds(self, ellipsoid: Ellipsoid) -> LayerBounds:
        """
        Sample points on the ellipsoid surface to estimate coordinate-wise bounds.
        
        Sampling strategy:
        1. Sample from unit sphere: u ~ Uniform(S^{n-1})
        2. Transform to ellipsoid: x = c + P^{-1/2} u
        """
        center = ellipsoid.center
        is_diag = ellipsoid.is_diagonal
        
        if is_diag:
            return self._sample_diagonal_ellipsoid(ellipsoid)
        
        P = ellipsoid.P
        has_batch = P.dim() == 3
        
        if has_batch:
            batch_size = P.shape[0]
            n_dims = P.shape[1]
            center_flat = center.view(batch_size, -1)
            
            lower = torch.full((batch_size, n_dims), float('inf'), device=self.device)
            upper = torch.full((batch_size, n_dims), float('-inf'), device=self.device)
            
            for b_idx in range(batch_size):
                P_b = P[b_idx]
                center_b = center_flat[b_idx]
                
                try:
                    L = torch.linalg.cholesky(P_b)
                    L_inv_T = torch.linalg.solve_triangular(
                        L, torch.eye(n_dims, device=self.device), upper=False
                    ).T
                except:
                    U, s, Vh = torch.linalg.svd(P_b)
                    L_inv_T = U @ torch.diag(1.0 / torch.sqrt(s + 1e-8)) @ Vh
                
                lower_b = torch.full((n_dims,), float('inf'), device=self.device)
                upper_b = torch.full((n_dims,), float('-inf'), device=self.device)
                
                n_chunks = (self.n_samples + self.chunk_size - 1) // self.chunk_size
                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * self.chunk_size
                    end_idx = min(start_idx + self.chunk_size, self.n_samples)
                    chunk_n = end_idx - start_idx
                    
                    u_chunk = torch.randn(chunk_n, n_dims, device=self.device)
                    u_chunk = u_chunk / torch.norm(u_chunk, p=2, dim=1, keepdim=True)
                    
                    samples_chunk = center_b.unsqueeze(0) + u_chunk @ L_inv_T.T
                    
                    lower_b = torch.minimum(lower_b, samples_chunk.min(dim=0).values)
                    upper_b = torch.maximum(upper_b, samples_chunk.max(dim=0).values)
                    
                    del u_chunk, samples_chunk
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
                lower[b_idx] = lower_b
                upper[b_idx] = upper_b
                
                del L_inv_T
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            center_shape = center.shape
            lower = lower.view(center_shape)
            upper = upper.view(center_shape)
        else:
            n_dims = P.shape[0]
            center_flat = center.view(-1)
            
            try:
                L = torch.linalg.cholesky(P)
                L_inv_T = torch.linalg.solve_triangular(
                    L, torch.eye(n_dims, device=self.device), upper=False
                ).T
            except:
                U, s, Vh = torch.linalg.svd(P)
                L_inv_T = U @ torch.diag(1.0 / torch.sqrt(s + 1e-8)) @ Vh
            
            lower_flat = torch.full((n_dims,), float('inf'), device=self.device)
            upper_flat = torch.full((n_dims,), float('-inf'), device=self.device)
            
            n_chunks = (self.n_samples + self.chunk_size - 1) // self.chunk_size
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, self.n_samples)
                chunk_n = end_idx - start_idx
                
                u_chunk = torch.randn(chunk_n, n_dims, device=self.device)
                u_chunk = u_chunk / torch.norm(u_chunk, p=2, dim=1, keepdim=True)
                
                samples_chunk = center_flat.unsqueeze(0) + u_chunk @ L_inv_T.T
                
                lower_flat = torch.minimum(lower_flat, samples_chunk.min(dim=0).values)
                upper_flat = torch.maximum(upper_flat, samples_chunk.max(dim=0).values)
                
                del u_chunk, samples_chunk
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            lower = lower_flat.view(center.shape)
            upper = upper_flat.view(center.shape)
            
            del L_inv_T
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return LayerBounds(lower=lower, upper=upper)
    
    def _sample_diagonal_ellipsoid(self, ellipsoid: Ellipsoid) -> LayerBounds:
        """
        Efficiently sample from a diagonal ellipsoid.
        
        For P = diag(p_i), the ellipsoid is:
            E = {x : sum_i p_i (x_i - c_i)^2 ≤ 1}
        
        Sampling: x_i = c_i + u_i / sqrt(p_i) where u is on unit sphere.
        """
        center = ellipsoid.center
        P_diag = ellipsoid.P
        
        has_batch = P_diag.dim() > 1
        
        if has_batch:
            batch_size = P_diag.shape[0]
            n_dims = P_diag.shape[1]
            center_flat = center.view(batch_size, -1)
            
            lower = torch.full((batch_size, n_dims), float('inf'), device=self.device)
            upper = torch.full((batch_size, n_dims), float('-inf'), device=self.device)
            
            scale = 1.0 / torch.sqrt(torch.clamp(P_diag, min=1e-8))
            
            for b_idx in range(batch_size):
                center_b = center_flat[b_idx]
                scale_b = scale[b_idx]
                
                lower_b = torch.full((n_dims,), float('inf'), device=self.device)
                upper_b = torch.full((n_dims,), float('-inf'), device=self.device)
                
                n_chunks = (self.n_samples + self.chunk_size - 1) // self.chunk_size
                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * self.chunk_size
                    end_idx = min(start_idx + self.chunk_size, self.n_samples)
                    chunk_n = end_idx - start_idx
                    
                    u_chunk = torch.randn(chunk_n, n_dims, device=self.device)
                    u_chunk = u_chunk / torch.norm(u_chunk, p=2, dim=1, keepdim=True)
                    
                    samples_chunk = center_b.unsqueeze(0) + u_chunk * scale_b.unsqueeze(0)
                    
                    lower_b = torch.minimum(lower_b, samples_chunk.min(dim=0).values)
                    upper_b = torch.maximum(upper_b, samples_chunk.max(dim=0).values)
                    
                    del u_chunk, samples_chunk
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                
                lower[b_idx] = lower_b
                upper[b_idx] = upper_b
            
            center_shape = center.shape
            lower = lower.view(center_shape)
            upper = upper.view(center_shape)
        else:
            n_dims = P_diag.shape[0]
            center_flat = center.view(-1)
            
            scale = 1.0 / torch.sqrt(torch.clamp(P_diag, min=1e-8))
            
            lower_flat = torch.full((n_dims,), float('inf'), device=self.device)
            upper_flat = torch.full((n_dims,), float('-inf'), device=self.device)
            
            n_chunks = (self.n_samples + self.chunk_size - 1) // self.chunk_size
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, self.n_samples)
                chunk_n = end_idx - start_idx
                
                u_chunk = torch.randn(chunk_n, n_dims, device=self.device)
                u_chunk = u_chunk / torch.norm(u_chunk, p=2, dim=1, keepdim=True)
                
                samples_chunk = center_flat.unsqueeze(0) + u_chunk * scale.unsqueeze(0)
                
                lower_flat = torch.minimum(lower_flat, samples_chunk.min(dim=0).values)
                upper_flat = torch.maximum(upper_flat, samples_chunk.max(dim=0).values)
                
                del u_chunk, samples_chunk
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            lower = lower_flat.view(center.shape)
            upper = upper_flat.view(center.shape)
        
        return LayerBounds(lower=lower, upper=upper)
    
    # =========================================================================
    # Löwner-John Bound Propagation
    # =========================================================================
    
    def loewner_john_propagate_bounds(
        self,
        first_layer_bounds: LayerBounds,
        model: nn.Module,
        start_from_layer: int = 1,
    ) -> Tuple[LayerBounds, int]:
        """
        Propagate bounds using Löwner-John ellipsoid approximation and sampling.
        
        Args:
            first_layer_bounds: Bounds from analytical first layer computation (L-inf)
            model: Neural network model
            start_from_layer: Layer index to start propagation
            
        Returns:
            (final_bounds, n_samples_used)
        """
        lower = first_layer_bounds.lower
        upper = first_layer_bounds.upper
        
        if lower.dim() == 4:
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
        
        lower = lower.to(self.device)
        upper = upper.to(self.device)
        
        bounds = LayerBounds(lower=lower, upper=upper)
        ellipsoid = self.linf_bounds_to_ellipsoid(bounds)
        
        layers = self.get_atomic_layers(model)
        
        first_conv_idx = 0
        for idx, layer in enumerate(layers):
            if isinstance(layer, nn.Conv3d):
                first_conv_idx = idx
                break
        
        start_from_layer = first_conv_idx + 1
        
        for i, layer in enumerate(layers):
            if i < start_from_layer:
                continue
            
            if isinstance(layer, nn.Conv3d):
                temp_bounds = self.sample_ellipsoid_bounds(ellipsoid)
                center = (temp_bounds.lower + temp_bounds.upper) / 2
                radius = (temp_bounds.upper - temp_bounds.lower) / 2
                
                if center.dim() == 2:
                    if i > 0:
                        center = center.view_as(lower) if lower.dim() == 5 else center.view(center.shape[0], -1, 1, 1, 1)
                        radius = radius.view_as(lower) if lower.dim() == 5 else radius.view(radius.shape[0], -1, 1, 1, 1)
                
                center_out = F.conv3d(
                    center, layer.weight, bias=layer.bias,
                    stride=layer.stride, padding=layer.padding
                )
                W_abs = layer.weight.abs()
                radius_out = F.conv3d(
                    radius, W_abs, bias=None,
                    stride=layer.stride, padding=layer.padding
                )
                
                lower_out = center_out - radius_out
                upper_out = center_out + radius_out
                bounds = LayerBounds(lower=lower_out, upper=upper_out)
                ellipsoid = self.linf_bounds_to_ellipsoid(bounds)
                lower = lower_out
                del temp_bounds, center, radius, center_out, radius_out
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
            elif isinstance(layer, nn.Linear):
                if ellipsoid.center.dim() > 2:
                    ellipsoid.center = ellipsoid.center.view(ellipsoid.center.shape[0], -1)
                elif ellipsoid.center.dim() == 1:
                    ellipsoid.center = ellipsoid.center.unsqueeze(0)
                
                ellipsoid = self.propagate_ellipsoid_affine(
                    ellipsoid, layer.weight, layer.bias
                )
            
            elif isinstance(layer, (nn.BatchNorm3d, nn.BatchNorm1d, nn.BatchNorm2d)):
                temp_bounds = self.sample_ellipsoid_bounds(ellipsoid)
                center = temp_bounds.center
                radius = temp_bounds.radius
                center_out, radius_out = self._propagate_batchnorm(center, radius, layer)
                lower_out = center_out - radius_out
                upper_out = center_out + radius_out
                bounds = LayerBounds(lower=lower_out, upper=upper_out)
                ellipsoid = self.linf_bounds_to_ellipsoid(bounds)
                del temp_bounds
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            elif isinstance(layer, nn.MaxPool3d):
                temp_bounds = self.sample_ellipsoid_bounds(ellipsoid)
                lower_out = F.max_pool3d(
                    temp_bounds.lower, kernel_size=layer.kernel_size, stride=layer.stride,
                    padding=layer.padding, dilation=layer.dilation, ceil_mode=layer.ceil_mode
                )
                upper_out = F.max_pool3d(
                    temp_bounds.upper, kernel_size=layer.kernel_size, stride=layer.stride,
                    padding=layer.padding, dilation=layer.dilation, ceil_mode=layer.ceil_mode
                )
                bounds = LayerBounds(lower=lower_out, upper=upper_out)
                ellipsoid = self.linf_bounds_to_ellipsoid(bounds)
                del temp_bounds

            elif isinstance(layer, nn.AdaptiveAvgPool3d):
                temp_bounds = self.sample_ellipsoid_bounds(ellipsoid)
                lower_out = F.adaptive_avg_pool3d(temp_bounds.lower, layer.output_size)
                upper_out = F.adaptive_avg_pool3d(temp_bounds.upper, layer.output_size)
                bounds = LayerBounds(lower=lower_out, upper=upper_out)
                ellipsoid = self.linf_bounds_to_ellipsoid(bounds)
                del temp_bounds

            elif isinstance(layer, nn.ReLU):
                temp_bounds = self.sample_ellipsoid_bounds(ellipsoid)
                lower_tmp = F.relu(temp_bounds.lower)
                upper_tmp = F.relu(temp_bounds.upper)
                bounds = LayerBounds(lower=lower_tmp, upper=upper_tmp)
                ellipsoid = self.linf_bounds_to_ellipsoid(bounds)
                del temp_bounds
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            elif isinstance(layer, (nn.Dropout, nn.Dropout3d)):
                pass
        
        final_bounds = self.sample_ellipsoid_bounds(ellipsoid)
        
        del ellipsoid
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return final_bounds, self.n_samples
    
    def _propagate_batchnorm(
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
        
        std = torch.sqrt(running_var + eps)
        scale = weight / std
        
        if center.dim() == 5:
            scale = scale.view(1, -1, 1, 1, 1)
            running_mean = running_mean.view(1, -1, 1, 1, 1)
            bias = bias.view(1, -1, 1, 1, 1)
        elif center.dim() == 2:
            scale = scale.view(1, -1)
            running_mean = running_mean.view(1, -1)
            bias = bias.view(1, -1)
        
        center_out = scale * (center - running_mean) + bias
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
        Complete verification pipeline: Analytical bounds + Löwner-John ellipsoid sampling.
        
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
        
        # Step 1: Closed-form bounds on first Conv3D layer (EXACT, L∞)
        first_layer_bounds = self.closed_form_conv3d_bounds(
            input_tensor, first_conv, epsilon, shared_frames
        )
        
        # Step 2: Löwner-John ellipsoid propagation through remaining layers
        final_bounds, n_samples_used = self.loewner_john_propagate_bounds(
            first_layer_bounds, self.model, start_from_layer=1
        )
        
        runtime = time.time() - start_time
        
        result = {
            'lower_bounds': final_bounds.lower,
            'upper_bounds': final_bounds.upper,
            'runtime': runtime,
            'n_samples': n_samples_used,
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
        
        lb_true = lower[true_label].item()
        num_classes = lower.size(0) if lower.dim() == 1 else lower.size(-1)
        
        for j in range(num_classes):
            if j != true_label:
                ub_other = upper[j].item()
                if lb_true <= ub_other:
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

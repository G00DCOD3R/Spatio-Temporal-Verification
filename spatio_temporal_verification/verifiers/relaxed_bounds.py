"""Shared first-layer bounds for relaxed temporal perturbation sharing."""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_relaxation_lambda(relaxation_lambda: float) -> None:
    if not math.isfinite(relaxation_lambda) or not 0 <= relaxation_lambda <= 1:
        raise ValueError("relaxation_lambda must be finite and in [0, 1].")


def relaxed_conv3d_bounds(
    input_tensor: torch.Tensor,
    conv_layer: nn.Conv3d,
    epsilon: float,
    shared_frames: List[List[int]],
    relaxation_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact coordinate-wise bounds for the first convolution.

    For each frame group and each channel/spatial position, perturbations have
    a shared scalar component of magnitude at most (1 - lambda) * epsilon and
    independent deviations of magnitude at most lambda * epsilon. Thus:

        radius_j = epsilon * sum_G(
            (1 - lambda) * abs(sum_i_in_G(w_ji))
            + lambda * sum_i_in_G(abs(w_ji))
        )

    Groups must be disjoint. Frames absent from all groups are unperturbed.
    Spatial positions and channels remain independent. No clipping to [0, 1]
    is imposed. Inputs may have shape (C, D, H, W) or (B, C, D, H, W).
    """
    validate_relaxation_lambda(relaxation_lambda)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative.")
    if conv_layer.padding_mode != "zeros" or isinstance(conv_layer.padding, str):
        raise NotImplementedError("Relaxed bounds require numeric zero padding.")
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)
    if input_tensor.dim() != 5:
        raise ValueError("Expected a 4D or 5D input tensor.")

    _, channels, depth, height, width = input_tensor.shape
    frame_groups = []
    seen = set()
    for group in shared_frames:
        group_set = set()
        for frame in group:
            if not isinstance(frame, int) or not 0 <= frame < depth:
                raise ValueError(f"Frame index {frame!r} must be an integer in [0, {depth}).")
            if frame in seen:
                raise ValueError("Frame groups must not overlap or contain duplicates.")
            seen.add(frame)
            group_set.add(frame)
        frame_groups.append(group_set)

    weights = conv_layer.weight
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    clean = F.conv3d(
        input_tensor, weights, conv_layer.bias, stride=stride,
        padding=padding, dilation=dilation, groups=conv_layer.groups,
    )
    radius = torch.zeros_like(clean)
    spatial_support = input_tensor.new_ones(1, channels, height, width)

    for output_depth in range(clean.shape[2]):
        for group in frame_groups:
            # Include only real, perturbable input slices in this receptive field.
            kernel_depths = [
                k for k in range(weights.shape[2])
                if output_depth * stride[0] - padding[0] + k * dilation[0] in group
            ]
            if not kernel_depths:
                continue
            selected_weights = weights[:, :, kernel_depths, :, :]
            shared_weights = selected_weights.sum(dim=2).abs()
            independent_weights = selected_weights.abs().sum(dim=2)
            radius_kernel = epsilon * (
                (1 - relaxation_lambda) * shared_weights
                + relaxation_lambda * independent_weights
            )
            contribution = F.conv2d(
                spatial_support, radius_kernel, stride=stride[1:],
                padding=padding[1:], dilation=dilation[1:], groups=conv_layer.groups,
            )
            radius[:, :, output_depth] = radius[:, :, output_depth] + contribution

    return clean - radius, clean + radius

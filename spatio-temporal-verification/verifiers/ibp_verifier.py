from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18 import VideoResNet
from models.simple3dcnn import Simple3DCNN

from .base_verifier import BaseVerifier


class IBPVerifier(BaseVerifier):
    def __init__(self, model: nn.Module, device: str = "cuda"):
        super().__init__(model)
        self.device = device

    def _compute_bounds(
        self, x: torch.Tensor, epsilon: float, frame_group: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IBP bounds for the entire sequence"""
        # Initialize bounds
        lower = x.clone()
        upper = x.clone()

        # Apply perturbation only to specified frames
        # Note: x shape is (batch_size, channels, num_frames, height, width)
        for frames in frame_group:
            for idx in frames:
                # lower[:, :, idx] = torch.clamp(x[:, :, idx] - epsilon, 0, 1)
                # upper[:, :, idx] = torch.clamp(x[:, :, idx] + epsilon, 0, 1)
                lower[:, :, idx] = x[:, :, idx] - epsilon
                upper[:, :, idx] = x[:, :, idx] + epsilon

        return lower, upper

    def _propagate_conv3d(
        self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor, layer: nn.Conv3d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through a 3D convolutional layer"""
        # Compute center and radius
        x_mu = (lower_bounds + upper_bounds) / 2
        x_r = (upper_bounds - lower_bounds) / 2

        # Compute bounds
        W = layer.weight
        abs_W = W.abs()

        conv0 = F.conv3d(
            x_mu,
            W,
            bias=layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        conv1 = F.conv3d(
            x_r,
            abs_W,
            bias=None,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )

        z_L = conv0 - conv1
        z_U = conv0 + conv1

        return z_L, z_U

    def _propagate_batchnorm3d(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        layer: nn.BatchNorm3d,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through a 3D batch normalization layer"""
        running_mean = layer.running_mean
        running_var = layer.running_var

        W = layer.state_dict()["weight"]
        b = layer.state_dict()["bias"]

        # Compute center and radius
        x_mu = (lower_bounds + upper_bounds) / 2
        x_r = (upper_bounds - lower_bounds) / 2

        lower_offset = F.batch_norm(
            -x_r, torch.zeros_like(running_mean), running_var, W, torch.zeros_like(b)
        )
        upper_offset = F.batch_norm(
            x_r, torch.zeros_like(running_mean), running_var, W, torch.zeros_like(b)
        )

        lower_offset = torch.minimum(lower_offset, upper_offset)
        upper_offset = torch.maximum(lower_offset, upper_offset)

        nominal = F.batch_norm(x_mu, running_mean, running_var, W, b)

        y_l = nominal + lower_offset
        y_u = nominal + upper_offset
        return y_l, y_u

    def _propagate_adaptive_avg_pool3d(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        output_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through an adaptive average pooling layer"""
        return (
            F.adaptive_avg_pool3d(lower_bounds, output_size),
            F.adaptive_avg_pool3d(upper_bounds, output_size),
        )

    # Interval bound propagation through the ResNet model with residual connections
    def propagate_bounds_resnet(
        self,
        input_tensor: torch.Tensor,
        epsilon: float = 0.1,
        lower_input: Optional[torch.Tensor] = None,
        upper_input: Optional[torch.Tensor] = None,
        skip_stem: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through the entire ResNet model"""
        # Set initial bounds
        if lower_input is not None and upper_input is not None:
            lower_bounds, upper_bounds = lower_input, upper_input
        else:
            lower_bounds = input_tensor - epsilon
            upper_bounds = input_tensor + epsilon

        # Manually evaluate the layers of the model and iterate only for BasicBlock
        assert isinstance(
            self.model, VideoResNet
        ), "Model is not an instance of VideoResNet."

        # Propagate through stem
        if not skip_stem and self.model.stem is not None:
            lower_bounds, upper_bounds = self._propagate_conv3d(
                lower_bounds, upper_bounds, self.model.stem[0]
            )
            lower_bounds, upper_bounds = self._propagate_batchnorm3d(
                lower_bounds, upper_bounds, self.model.stem[1]
            )
            lower_bounds, upper_bounds = self._propagate_relu(
                lower_bounds, upper_bounds
            )
            assert (
                lower_bounds <= upper_bounds
            ).all(), "Lower bounds should not exceed upper bounds!"

        # Propagate through layers
        for layer in [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ]:
            for block in layer:
                # Store residual bounds
                residual_lower, residual_upper = (
                    lower_bounds.clone(),
                    upper_bounds.clone(),
                )

                # First conv block
                if block.conv1 is not None:
                    lower_bounds, upper_bounds = self._propagate_conv3d(
                        lower_bounds, upper_bounds, block.conv1[0]
                    )
                    lower_bounds, upper_bounds = self._propagate_batchnorm3d(
                        lower_bounds, upper_bounds, block.conv1[1]
                    )
                    lower_bounds, upper_bounds = self._propagate_relu(
                        lower_bounds, upper_bounds
                    )

                # Second conv block
                if block.conv2 is not None:
                    lower_bounds, upper_bounds = self._propagate_conv3d(
                        lower_bounds, upper_bounds, block.conv2[0]
                    )
                    lower_bounds, upper_bounds = self._propagate_batchnorm3d(
                        lower_bounds, upper_bounds, block.conv2[1]
                    )

                # Downsample if needed
                if block.downsample is not None:
                    residual_lower, residual_upper = self._propagate_conv3d(
                        residual_lower, residual_upper, block.downsample[0]
                    )
                    residual_lower, residual_upper = self._propagate_batchnorm3d(
                        residual_lower, residual_upper, block.downsample[1]
                    )

                # Add residual and apply ReLU
                if block.relu is not None:
                    lower_bounds += residual_lower
                    upper_bounds += residual_upper
                    lower_bounds, upper_bounds = self._propagate_relu(
                        lower_bounds, upper_bounds
                    )

                assert (
                    lower_bounds <= upper_bounds
                ).all(), "Lower bounds should not exceed upper bounds!"

        # Final layers
        if self.model.avgpool is not None:
            lower_bounds, upper_bounds = self._propagate_adaptive_avg_pool3d(
                lower_bounds, upper_bounds, (1, 1, 1)
            )
            assert (
                lower_bounds <= upper_bounds
            ).all(), "Lower bounds should not exceed upper bounds!"

        if self.model.fc is not None:
            # Flatten if needed
            if lower_bounds.ndim > 2:
                lower_bounds = lower_bounds.view(lower_bounds.size(0), -1)
                upper_bounds = upper_bounds.view(upper_bounds.size(0), -1)
            lower_bounds, upper_bounds = self._propagate_linear(
                lower_bounds, upper_bounds, self.model.fc
            )
            assert (
                lower_bounds <= upper_bounds
            ).all(), "Lower bounds should not exceed upper bounds!"

        return lower_bounds, upper_bounds
    
    
    # Interval bound propagation through the Simple3DCNN model
    def propagate_bounds_simple3dcnn(
        self,
        input_tensor: torch.Tensor,
        epsilon: float = 0.1,
        lower_input: Optional[torch.Tensor] = None,
        upper_input: Optional[torch.Tensor] = None,
        skip_first_conv: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through the entire Simple3DCNN model"""
        # Set initial bounds
        if lower_input is not None and upper_input is not None:
            lower_bounds, upper_bounds = lower_input, upper_input
        else:
            lower_bounds = input_tensor - epsilon
            upper_bounds = input_tensor + epsilon

        # Verify model instance is a Simple3DCNN or a subclass of it
        assert isinstance(
            self.model, Simple3DCNN
        ), f"Model is not an instance of Simple3DCNN but {type(self.model).__name__}"

        # Process convolutional layers
        # Conv layer 1
        if not skip_first_conv:
            lower_bounds, upper_bounds = self._propagate_conv3d(
                lower_bounds, upper_bounds, self.model.conv1
            )
        
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        lower_bounds, upper_bounds = self._propagate_batchnorm3d(
            lower_bounds, upper_bounds, self.model.bn1
        )
        
        # Conv layer 2
        lower_bounds, upper_bounds = self._propagate_conv3d(
            lower_bounds, upper_bounds, self.model.conv2
        )
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        lower_bounds, upper_bounds = self._propagate_batchnorm3d(
            lower_bounds, upper_bounds, self.model.bn2
        )
        
        # Conv layer 3
        lower_bounds, upper_bounds = self._propagate_conv3d(
            lower_bounds, upper_bounds, self.model.conv3
        )
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        lower_bounds, upper_bounds = self._propagate_batchnorm3d(
            lower_bounds, upper_bounds, self.model.bn3
        )
        
        # Flatten
        if lower_bounds.ndim > 2:
            lower_bounds = lower_bounds.view(lower_bounds.size(0), -1)
            upper_bounds = upper_bounds.view(upper_bounds.size(0), -1)
            
        # FC layer 1
        lower_bounds, upper_bounds = self._propagate_linear(
            lower_bounds, upper_bounds, self.model.fc1
        )
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        lower_bounds, upper_bounds = self._propagate_batchnorm(
            lower_bounds, upper_bounds, self.model.bn_fc
        )
        
        # Dropout (during evaluation, it's just identity)
        # We use our custom dropout propagation function
        lower_bounds, upper_bounds = self._propagate_dropout(
            lower_bounds, upper_bounds, self.model.dropout
        )
        
        # FC layer 2
        lower_bounds, upper_bounds = self._propagate_linear(
            lower_bounds, upper_bounds, self.model.fc2
        )

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

        print(f"\nIBP Verification with epsilon={epsilon}")
        print(f"Input shape: {x.shape}")
        print(f"Number of samples: {total}")

        for i in range(total):
            print(f"\nVerifying sample {i+1}/{total}")
            input_tensor = x[i:i+1]
            y_i = y[i:i+1]
        
            with torch.no_grad():
                if isinstance(self.model, VideoResNet):
                    lower_bounds, upper_bounds = self.propagate_bounds_resnet(
                        input_tensor, epsilon=epsilon
                    )
                elif isinstance(self.model, Simple3DCNN):
                    lower_bounds, upper_bounds = self.propagate_bounds_simple3dcnn(
                        input_tensor, epsilon=epsilon
                    )
            
            # Print bound statistics
            print(f"Lower bounds range: [{lower_bounds.min().item():.4f}, {lower_bounds.max().item():.4f}]")
            print(f"Upper bounds range: [{upper_bounds.min().item():.4f}, {upper_bounds.max().item():.4f}]")
            print(f"Bound width: {(upper_bounds - lower_bounds).mean().item():.4f}")

            # Check robustness
            margin = lower_bounds[0, y_i] - torch.max(
                torch.cat([upper_bounds[0, :y_i], upper_bounds[0, y_i + 1 :]], dim=0)
            )

            if margin.item() > 0:
                verified_correct += 1
                print(f"Sample {i+1} is robust (margin: {margin.item():.4f})")
            else:
                print(f"Sample {i+1} is not robust (margin: {margin.item():.4f})")

        results[epsilon] = verified_correct / total
        return results

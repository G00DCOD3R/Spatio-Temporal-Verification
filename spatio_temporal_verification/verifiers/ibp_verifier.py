from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.resnet18 import VideoResNet
except ImportError:
    try:
        from spatio_temporal_verification.models.resnet18 import VideoResNet
    except ImportError:
        VideoResNet = type(None)

try:
    from models.simple3dcnn import Simple3DCNN
except ImportError:
    try:
        from spatio_temporal_verification.models.simple3dcnn import Simple3DCNN
    except ImportError:
        Simple3DCNN = type(None)

try:
    from experiments.udacity_steering_angle.models.udacity_3dcnn import UdacitySteeringAngle_3DCNN, CNN3D
except ImportError:
    try:
        from spatio_temporal_verification.experiments.udacity_steering_angle.models.udacity_3dcnn import UdacitySteeringAngle_3DCNN, CNN3D
    except ImportError:
        UdacitySteeringAngle_3DCNN, CNN3D = type(None), type(None)

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
        self, lower_bounds: torch.Tensor, upper_bounds: Union[torch.Tensor, nn.Conv3d], layer: Optional[nn.Conv3d] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Propagate bounds or single tensor through a 3D conv layer."""
        if layer is None and isinstance(upper_bounds, nn.Conv3d):
            return self._forward_conv3d(lower_bounds, upper_bounds)
        
        conv_layer = layer
        x_mu = (lower_bounds + upper_bounds) / 2
        x_r = (upper_bounds - lower_bounds) / 2
        W = conv_layer.weight
        abs_W = W.abs()

        conv0 = F.conv3d(
            x_mu,
            W,
            bias=conv_layer.bias,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
        )
        conv1 = F.conv3d(
            x_r,
            abs_W,
            bias=None,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
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

        # Verify model has conv1 layer
        assert hasattr(
            self.model, "conv1"
        ), f"Model does not have conv1 layer: {type(self.model).__name__}"

        # Process convolutional layers
        layers = self.get_atomic_layers(self.model)
        start_idx = 0
        if skip_first_conv:
            for idx, l in enumerate(layers):
                if isinstance(l, nn.Conv3d):
                    start_idx = idx + 1
                    break
        
        for layer in layers[start_idx:]:
            lower_bounds, upper_bounds = self.propagate_bounds((lower_bounds, upper_bounds), layer)

        return lower_bounds, upper_bounds

    def verify_sequence(
        self, x: torch.Tensor, y: torch.Tensor, frame_group: List[List[int]], epsilon: float
    ) -> Dict[float, float]:
        """Verify robustness of a sequence of frames using Interval Bound Propagation (IBP)"""
        # Get lower and upper bounds for the input
        lower, upper = self._compute_bounds(x, epsilon, frame_group)

        if Simple3DCNN is not None and isinstance(self.model, Simple3DCNN):
            output_lower, output_upper = self._ibp_propagate_bounds_simple3dcnn(
                x, lower, upper
            )
        elif VideoResNet is not None and isinstance(self.model, VideoResNet):
            output_lower, output_upper = self._ibp_propagate_bounds_resnet18(
                x, lower, upper
            )
        elif (UdacitySteeringAngle_3DCNN is not None and isinstance(self.model, UdacitySteeringAngle_3DCNN)) or (hasattr(self.model, 'cnn3d') and CNN3D is not None and isinstance(getattr(self.model, 'cnn3d'), CNN3D)):
            output_lower, output_upper = self._ibp_propagate_bounds_udacity_cnn3d(
                x, lower, upper
            )
        else:
            # Generic atomic layer IBP propagation fallback
            layers = self.get_atomic_layers(self.model)
            curr_lower, curr_upper = lower, upper
            for layer in layers:
                curr_lower, curr_upper = self.propagate_bounds((curr_lower, curr_upper), layer)
            output_lower, output_upper = curr_lower, curr_upper

        # Compute robustness
        batch_size = x.size(0)
        robust_count = 0

        for i in range(batch_size):
            # Get the true class label
            true_class = y[i].item()

            # Get the lower bound for the true class
            lb_true_class = output_lower[i, true_class].item()

            # Check if the lower bound of the true class is greater than the upper bound of all other classes
            robust = True
            for j in range(output_upper.size(1)):
                if j != true_class and output_upper[i, j] >= lb_true_class:
                    robust = False
                    break

            if robust:
                robust_count += 1

        results = {epsilon: robust_count / batch_size}
        return results

    def _ibp_propagate_bounds_simple3dcnn(
        self,
        x: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through a Simple3DCNN model"""
        # Shorthand for model
        model = self.model

        # Propagate through conv1
        lower = self._propagate_conv3d(lower, model.conv1)
        upper = self._propagate_conv3d(upper, model.conv1)

        # Propagate through relu
        lower, upper = self._propagate_relu(lower, upper)

        # Propagate through conv2
        lower = self._propagate_conv3d(lower, model.conv2)
        upper = self._propagate_conv3d(upper, model.conv2)

        # Propagate through relu
        lower, upper = self._propagate_relu(lower, upper)

        # Propagate through pool
        lower = self._propagate_maxpool3d(lower, model.pool)
        upper = self._propagate_maxpool3d(upper, model.pool)

        # Flatten
        lower = lower.view(lower.size(0), -1)
        upper = upper.view(upper.size(0), -1)

        # Propagate through fc1
        lower = self._forward_linear(lower, model.fc1)
        upper = self._forward_linear(upper, model.fc1)

        # Propagate through relu
        lower, upper = self._propagate_relu(lower, upper)

        # Propagate through fc2
        lower = self._forward_linear(lower, model.fc2)
        upper = self._forward_linear(upper, model.fc2)

        return lower, upper

    def _ibp_propagate_bounds_resnet18(
        self,
        x: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through a VideoResNet18 model"""
        # TODO: Implement bound propagation for ResNet18
        pass

    def _ibp_propagate_bounds_udacity_cnn3d(
        self,
        input_tensor: torch.Tensor,
        lower_input: torch.Tensor,
        upper_input: torch.Tensor,
        skip_first_conv: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through the UdacitySteeringAngle_3DCNN model"""
        # Get the CNN3D model from UdacitySteeringAngle_3DCNN wrapper if needed
        if isinstance(self.model, UdacitySteeringAngle_3DCNN):
            model = self.model.cnn3d
        else:
            model = self.model

        # Set initial bounds
        lower_bounds = lower_input
        upper_bounds = upper_input

        # Verify model instance is a CNN3D or a subclass of it
        assert isinstance(
            model, CNN3D
        ), f"Model is not an instance of CNN3D but {type(model).__name__}"

        # Process convolutional layers
        # Conv layer 1
        if not skip_first_conv:
            lower_bounds = self._propagate_conv3d(lower_bounds, model.conv1)
            upper_bounds = self._propagate_conv3d(upper_bounds, model.conv1)
        
        lower_bounds = self._propagate_batchnorm3d(lower_bounds, model.bn1)
        upper_bounds = self._propagate_batchnorm3d(upper_bounds, model.bn1)
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        
        # Conv layer 2
        lower_bounds = self._propagate_conv3d(lower_bounds, model.conv2)
        upper_bounds = self._propagate_conv3d(upper_bounds, model.conv2)
        lower_bounds = self._propagate_batchnorm3d(lower_bounds, model.bn2)
        upper_bounds = self._propagate_batchnorm3d(upper_bounds, model.bn2)
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        
        lower_bounds = self._propagate_adaptive_avg_pool3d(lower_bounds, (2, 2, 2))
        upper_bounds = self._propagate_adaptive_avg_pool3d(upper_bounds, (2, 2, 2))
        
        # Flatten
        if lower_bounds.ndim > 2:
            lower_bounds = lower_bounds.view(lower_bounds.size(0), -1)
            upper_bounds = upper_bounds.view(upper_bounds.size(0), -1)
            
        # FC layer 1
        lower_bounds = self._forward_linear(lower_bounds, model.fc1)
        upper_bounds = self._forward_linear(upper_bounds, model.fc1)
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        
        # FC layer 2
        lower_bounds = self._forward_linear(lower_bounds, model.fc2)
        upper_bounds = self._forward_linear(upper_bounds, model.fc2)
        lower_bounds, upper_bounds = self._propagate_relu(lower_bounds, upper_bounds)
        
        # FC layer 3
        lower_bounds = self._forward_linear(lower_bounds, model.fc3)
        upper_bounds = self._forward_linear(upper_bounds, model.fc3)

        return lower_bounds, upper_bounds

    def _forward_conv3d(
        self, x: torch.Tensor, layer: nn.Conv3d
    ) -> torch.Tensor:
        """Forward pass single tensor through 3D conv layer"""
        return F.conv3d(
            x,
            layer.weight,
            layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )

    def _propagate_batchnorm3d(
        self, lower_bounds: torch.Tensor, upper_bounds: Union[torch.Tensor, nn.BatchNorm3d], layer: Optional[nn.BatchNorm3d] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Propagate bounds or single tensor through a 3D batch normalization layer"""
        if layer is None and isinstance(upper_bounds, nn.BatchNorm3d):
            bn_layer = upper_bounds
            x = lower_bounds
            weight = bn_layer.weight
            bias = bn_layer.bias
            running_mean = bn_layer.running_mean
            running_var = bn_layer.running_var
            eps = bn_layer.eps

            x_normalized = (x - running_mean.view(1, -1, 1, 1, 1)) / torch.sqrt(
                running_var.view(1, -1, 1, 1, 1) + eps
            )
            return x_normalized * weight.view(1, -1, 1, 1, 1) + bias.view(1, -1, 1, 1, 1)
        else:
            return self._propagate_batchnorm(lower_bounds, upper_bounds, layer)

    def _propagate_adaptive_avg_pool3d(
        self, x: torch.Tensor, output_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Propagate bounds through an adaptive average pooling 3D layer"""
        return F.adaptive_avg_pool3d(x, output_size)

    def _propagate_maxpool3d(
        self, x: torch.Tensor, layer: nn.MaxPool3d
    ) -> torch.Tensor:
        """Propagate bounds through a 3D max pooling layer"""
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation
        return F.max_pool3d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=layer.ceil_mode,
        )

    def _forward_linear(
        self, x: torch.Tensor, layer: nn.Linear
    ) -> torch.Tensor:
        """Forward pass single tensor through a linear layer"""
        return F.linear(x, layer.weight, layer.bias)

    def _propagate_relu(
        self, lower: torch.Tensor, upper: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through a ReLU activation"""
        lower = F.relu(lower)
        upper = F.relu(upper)
        return lower, upper

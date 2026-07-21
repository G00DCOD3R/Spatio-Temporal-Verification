from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVerifier:
    def __init__(self, model: nn.Module):
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_atomic_layers(self, module: Optional[nn.Module] = None) -> List[nn.Module]:
        if module is None:
            module = self.model

        def _extract(m):
            layers = []
            for child in m.children():
                if isinstance(child, (nn.Sequential, nn.ModuleList)) or (
                    len(list(child.children())) > 0
                    and not isinstance(
                        child,
                        (
                            nn.Conv3d,
                            nn.Conv2d,
                            nn.Linear,
                            nn.BatchNorm3d,
                            nn.BatchNorm2d,
                            nn.BatchNorm1d,
                            nn.MaxPool3d,
                            nn.AdaptiveAvgPool3d,
                            nn.Dropout,
                            nn.Dropout3d,
                            nn.ReLU,
                        ),
                    )
                ):
                    layers.extend(_extract(child))
                else:
                    layers.append(child)
            return layers

        raw_layers = _extract(module)
        seen = set()
        unique_layers = []
        for l in raw_layers:
            if id(l) not in seen:
                seen.add(id(l))
                unique_layers.append(l)
        return unique_layers

    def propagate_bounds(
        self, input_bounds: Tuple[torch.Tensor, torch.Tensor], layer: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate bounds through a single layer using IBP.

        Args:
            input_bounds: Tuple of (lower_bounds, upper_bounds) for the input
            layer: The layer to propagate bounds through

        Returns:
            Tuple of (lower_bounds, upper_bounds) for the output
        """
        lower_bounds, upper_bounds = input_bounds

        if isinstance(layer, nn.Linear):
            return self._propagate_linear(lower_bounds, upper_bounds, layer)
        elif isinstance(layer, nn.Conv2d):
            return self._propagate_conv2d(lower_bounds, upper_bounds, layer)
        elif isinstance(layer, nn.Conv3d):
            return self._propagate_conv3d(lower_bounds, upper_bounds, layer)
        elif isinstance(layer, nn.ReLU):
            return self._propagate_relu(lower_bounds, upper_bounds)
        elif isinstance(layer, nn.MaxPool3d):
            return self._propagate_maxpool3d(lower_bounds, upper_bounds, layer)
        elif isinstance(layer, nn.BatchNorm3d) or isinstance(layer, nn.BatchNorm2d):
            return self._propagate_batchnorm(lower_bounds, upper_bounds, layer)
        elif isinstance(layer, nn.Dropout):
            return self._propagate_dropout(lower_bounds, upper_bounds, layer)
        elif isinstance(layer, nn.AdaptiveAvgPool3d):
            return self._propagate_adaptive_avgpool3d(
                lower_bounds, upper_bounds, layer.output_size
            )
        else:
            raise NotImplementedError(
                f"IBP propagation not implemented for {type(layer)}"
            )

    # Convolutional layer IBP
    def _propagate_conv3d(self, lower_bound, upper_bound, conv_layer):
        # Compute center and radius
        x_mu = (lower_bound + upper_bound) / 2
        x_r = (upper_bound - lower_bound) / 2

        # Compute bounds
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

    def _propagate_linear(self, lower_bound, upper_bound, fc_layer):
        if lower_bound.dim() > 2:
            lower_bound = lower_bound.view(lower_bound.size(0), -1)
            upper_bound = upper_bound.view(upper_bound.size(0), -1)
        # Compute center and radius
        x_mu = (lower_bound + upper_bound) / 2
        x_r = (upper_bound - lower_bound) / 2

        # Compute bounds
        W = fc_layer.weight
        abs_W = W.abs()
        z_L = F.linear(x_mu, W, bias=fc_layer.bias) - F.linear(x_r, abs_W, bias=None)
        z_U = F.linear(x_mu, W, bias=fc_layer.bias) + F.linear(x_r, abs_W, bias=None)

        return z_L, z_U

    def _propagate_relu(self, lower_bound, upper_bound):
        return F.relu(lower_bound), F.relu(upper_bound)

    def _propagate_maxpool3d(self, lower_bound, upper_bound, maxpool_layer):
        """
        Propagate bounds through a MaxPool3d layer.
        
        MaxPool is a monotonic function, so we can directly apply it to both bounds.
        
        Args:
            lower_bound: Lower bounds of the input
            upper_bound: Upper bounds of the input
            maxpool_layer: The MaxPool3d layer to propagate through
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) for the output
        """
        # MaxPool3d is monotonic, so we can directly apply it to bounds
        z_L = F.max_pool3d(
            lower_bound,
            kernel_size=maxpool_layer.kernel_size,
            stride=maxpool_layer.stride,
            padding=maxpool_layer.padding,
            dilation=maxpool_layer.dilation,
            ceil_mode=maxpool_layer.ceil_mode,
            return_indices=False
        )
        
        z_U = F.max_pool3d(
            upper_bound,
            kernel_size=maxpool_layer.kernel_size,
            stride=maxpool_layer.stride,
            padding=maxpool_layer.padding,
            dilation=maxpool_layer.dilation,
            ceil_mode=maxpool_layer.ceil_mode,
            return_indices=False
        )
        
        return z_L, z_U

    def _propagate_batchnorm(self, lower_bound, upper_bound, bn_layer):
        running_mean = bn_layer.running_mean
        running_var = bn_layer.running_var

        W = bn_layer.state_dict()["weight"]
        b = bn_layer.state_dict()["bias"]

        # Compute center and radius
        x_mu = (lower_bound + upper_bound) / 2
        x_r = (upper_bound - lower_bound) / 2

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

    def _propagate_adaptive_avgpool3d(self, lower_bound, upper_bound, output_size):
        return F.adaptive_avg_pool3d(lower_bound, output_size), F.adaptive_avg_pool3d(
            upper_bound, output_size
        )

    def _propagate_dropout(self, lower_bound, upper_bound, dropout_layer):
        """
        Propagate bounds through a dropout layer.
        
        During verification, we assume the model is in evaluation mode,
        so dropout should effectively be an identity function.
        
        Args:
            lower_bound: Lower bounds of the input
            upper_bound: Upper bounds of the input
            dropout_layer: The dropout layer to propagate through
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) for the output
        """
        if self.model.training:
            # Only apply scaling during training
            # During training, dropout scales values by 1/(1-p)
            scale = 1.0 / (1.0 - dropout_layer.p)
            return lower_bound * scale, upper_bound * scale
        else:
            # During evaluation, dropout is an identity function
            return lower_bound, upper_bound

"""Relaxed first-layer bounds followed by the existing propagation methods.

Set relaxation_lambda on construction. The inherited verify() methods dispatch
to the shared relaxed first-layer implementation. Experiment/CLI wiring is
separate. The patch verification API is not part of this relaxation.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .closed_form_ibp_verifier import ClosedFormIBPVerifier, LayerBounds
from .closed_form_lipschitz_verifier import ClosedFormLipschitzVerifier
from .closed_form_lownerjohn_verifier import ClosedFormLownerJohnVerifier
from .relaxed_bounds import relaxed_conv3d_bounds, validate_relaxation_lambda


class _RelaxedFirstLayerMixin:
    def __init__(self, model: nn.Module, device: str = "cuda", *,
                 relaxation_lambda: float = 0.0, **kwargs):
        validate_relaxation_lambda(relaxation_lambda)
        super().__init__(model, device=device, **kwargs)
        self.relaxation_lambda = relaxation_lambda

    def closed_form_conv3d_bounds(
        self,
        input_tensor: torch.Tensor,
        conv_layer: nn.Conv3d,
        epsilon: float,
        shared_frames: List[List[int]],
        relaxation_lambda: Optional[float] = None,
    ) -> LayerBounds:
        """Use the instance's lambda unless explicitly overridden for this call."""
        if relaxation_lambda is None:
            relaxation_lambda = self.relaxation_lambda
        lower, upper = relaxed_conv3d_bounds(
            input_tensor.to(self.device), conv_layer, epsilon, shared_frames,
            relaxation_lambda,
        )
        return LayerBounds(lower=lower, upper=upper)

    def verify_patch(self, *args, **kwargs):
        raise NotImplementedError(
            "Relaxed verifiers currently support shared frame groups via verify(), "
            "not patch verification."
        )


class RelaxedIBPVerifier(_RelaxedFirstLayerMixin, ClosedFormIBPVerifier):
    """Relaxed first-layer intervals followed by the existing IBP propagation."""


class RelaxedLipschitzVerifier(_RelaxedFirstLayerMixin, ClosedFormLipschitzVerifier):
    """Relaxed first-layer intervals followed by existing Lipschitz propagation."""


class RelaxedLownerJohnVerifier(_RelaxedFirstLayerMixin, ClosedFormLownerJohnVerifier):
    """Relaxed first-layer intervals followed by existing empirical LJ sampling.

    Accepts the parent's n_samples, chunk_size, and max_dims constructor options.
    Sampling results are not formal robustness certificates.
    """

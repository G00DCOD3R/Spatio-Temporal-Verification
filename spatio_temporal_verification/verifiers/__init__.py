"""
Verifiers for Spatio-Temporal Video Classification Robustness

This module provides various verification methods for neural network robustness:

1. Base Verifier (BaseVerifier):
   - Base class with common bound propagation methods

2. IBP Verifier (IBPVerifier):
   - Standard Interval Bound Propagation

3. MILP-IBP Verifier (MILPIBPVerifier):
   - Mixed Integer Linear Programming with IBP

4. Closed-Form IBP Verifier (ClosedFormIBPVerifier):
   - Closed-form first layer bounds + IBP propagation
   - Efficient GPU-accelerated verification

5. Closed-Form Lipschitz Verifier (ClosedFormLipschitzVerifier):
   - Closed-form first layer bounds + Lipschitz propagation
   - Handles L∞ to L-2 norm conversion

6. Closed-Form Löwner-John Verifier (ClosedFormLownerJohnVerifier):
   - Closed-form first layer bounds + Löwner-John ellipsoid propagation
   - Uses ellipsoid approximations with sampling for tighter bounds
"""

from .base_verifier import BaseVerifier
from .ibp_verifier import IBPVerifier
from .closed_form_ibp_verifier import ClosedFormIBPVerifier, LayerBounds, PerturbationSpec
from .closed_form_lipschitz_verifier import ClosedFormLipschitzVerifier
from .closed_form_lownerjohn_verifier import ClosedFormLownerJohnVerifier, Ellipsoid

try:
    from .milp_ibp_verifier import MILPIBPVerifier
    from .milp_ibp_patch_verifier import MILPIBPPatchVerifier
except Exception:
    MILPIBPVerifier = None
    MILPIBPPatchVerifier = None

__all__ = [
    # Base verifier
    "BaseVerifier",
    # Standard verifiers
    "IBPVerifier",
    "MILPIBPVerifier",
    "MILPIBPPatchVerifier",
    # Closed-form verifiers
    "ClosedFormIBPVerifier",
    "ClosedFormLipschitzVerifier",
    "ClosedFormLownerJohnVerifier",
    # Data classes
    "LayerBounds",
    "PerturbationSpec",
    "Ellipsoid",
]

import pytest
import torch
import torch.nn as nn
from spatio_temporal_verification.verifiers.base_verifier import BaseVerifier

def test_base_verifier_linear():
    # Create a simple linear model
    model = nn.Linear(2, 2)
    verifier = BaseVerifier(model)
    
    # Create input bounds
    lower_bounds = torch.tensor([-1.0, -1.0])
    upper_bounds = torch.tensor([1.0, 1.0])
    
    # Propagate bounds
    lower_out, upper_out = verifier.propagate_bounds((lower_bounds, upper_bounds), model)
    
    # Basic assertions
    assert lower_out.shape == upper_out.shape
    assert torch.all(lower_out <= upper_out)

def test_base_verifier_relu():
    # Create a simple ReLU model
    model = nn.ReLU()
    verifier = BaseVerifier(model)
    
    # Create input bounds
    lower_bounds = torch.tensor([-1.0, -1.0])
    upper_bounds = torch.tensor([1.0, 1.0])
    
    # Propagate bounds
    lower_out, upper_out = verifier.propagate_bounds((lower_bounds, upper_bounds), model)
    
    # Basic assertions
    assert lower_out.shape == upper_out.shape
    assert torch.all(lower_out <= upper_out)
    assert torch.all(lower_out >= 0)  # ReLU should ensure non-negative outputs 
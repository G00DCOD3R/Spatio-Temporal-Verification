import pytest
import torch
import torch.nn as nn
from spatio_temporal_verification.verifiers import (
    BaseVerifier,
    IBPVerifier,
    ClosedFormIBPVerifier,
    ClosedFormLipschitzVerifier,
    ClosedFormLownerJohnVerifier,
    MILPIBPVerifier,
    MILPIBPPatchVerifier,
)

class SimpleTest3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 5 * 14 * 14, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_base_verifier_linear():
    model = nn.Linear(2, 2)
    verifier = BaseVerifier(model)
    lower_bounds = torch.tensor([-1.0, -1.0])
    upper_bounds = torch.tensor([1.0, 1.0])
    lower_out, upper_out = verifier.propagate_bounds((lower_bounds, upper_bounds), model)
    assert lower_out.shape == upper_out.shape
    assert torch.all(lower_out <= upper_out)


def test_base_verifier_relu():
    model = nn.ReLU()
    verifier = BaseVerifier(model)
    lower_bounds = torch.tensor([-1.0, -1.0])
    upper_bounds = torch.tensor([1.0, 1.0])
    lower_out, upper_out = verifier.propagate_bounds((lower_bounds, upper_bounds), model)
    assert lower_out.shape == upper_out.shape
    assert torch.all(lower_out <= upper_out)
    assert torch.all(lower_out >= 0)


def test_ibp_verifier():
    model = SimpleTest3DCNN()
    model.eval()
    x = torch.randn(1, 1, 5, 14, 14)
    target = torch.tensor([0])
    epsilon = 0.01
    frame_group = [[0, 1, 2, 3, 4]]

    verifier = IBPVerifier(model, device="cpu")
    results = verifier.verify_sequence(x, target, frame_group, epsilon)
    assert epsilon in results
    assert 0.0 <= results[epsilon] <= 1.0


def test_closed_form_ibp_verifier():
    model = SimpleTest3DCNN()
    model.eval()
    x = torch.randn(1, 1, 5, 14, 14)
    epsilon = 0.01
    shared_frames = [[0, 1, 2, 3, 4]]

    verifier = ClosedFormIBPVerifier(model, device="cpu")
    res = verifier.verify(x, epsilon, shared_frames, true_label=0)
    assert "lower_bounds" in res and "upper_bounds" in res
    assert "verified" in res

    # Test patch verification
    patch_locations = {0: (2, 2, 8, 8)}
    res_patch = verifier.verify_patch(x, epsilon, patch_locations, true_label=0)
    assert "lower_bounds" in res_patch and "upper_bounds" in res_patch


def test_closed_form_lipschitz_verifier():
    model = SimpleTest3DCNN()
    model.eval()
    x = torch.randn(1, 1, 5, 14, 14)
    epsilon = 0.01
    shared_frames = [[0, 1, 2, 3, 4]]

    verifier = ClosedFormLipschitzVerifier(model, device="cpu")
    res = verifier.verify(x, epsilon, shared_frames, true_label=0)
    assert "lower_bounds" in res and "upper_bounds" in res
    assert "verified" in res

    # Test patch verification
    patch_locations = {0: (2, 2, 8, 8)}
    res_patch = verifier.verify_patch(x, epsilon, patch_locations, true_label=0)
    assert "lower_bounds" in res_patch and "upper_bounds" in res_patch


def test_closed_form_lowner_john_verifier():
    model = SimpleTest3DCNN()
    model.eval()
    x = torch.randn(1, 1, 5, 14, 14)
    epsilon = 0.01
    shared_frames = [[0, 1, 2, 3, 4]]

    verifier = ClosedFormLownerJohnVerifier(model, device="cpu")
    res = verifier.verify(x, epsilon, shared_frames, true_label=0)
    assert "lower_bounds" in res and "upper_bounds" in res
    assert "verified" in res


def test_milp_ibp_verifiers_if_gurobi_available():
    if MILPIBPVerifier is None or MILPIBPPatchVerifier is None:
        pytest.skip("Gurobi is not installed, skipping MILP verifier tests.")

    model = SimpleTest3DCNN()
    model.eval()
    x = torch.randn(1, 1, 5, 14, 14)
    target = torch.tensor([0])
    epsilon = 0.01
    frame_group = [[0, 1, 2, 3, 4]]

    milp_verifier = MILPIBPVerifier(model, device="cpu")
    res = milp_verifier.verify_sequence(x, target, frame_group, epsilon)
    assert epsilon in res
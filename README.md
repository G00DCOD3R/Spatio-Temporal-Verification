# Spatio-temporal Verification

This project implements various verification methods for neural networks, focusing on spatio-temporal properties.

## Project Structure

```
spatio-temporal-verification/
├── experiments/
│   └── mnist/
│       ├── checkpoints/     # Model checkpoints
│       └── models/          # MNIST-specific model implementations
├── models/
│   └── resnet18.py         # Base ResNet18 implementation
├── utils/                  # Utility functions
├── verifiers/
│   ├── base_verifier.py    # IBP propagation code
│   ├── ibp_verifier.py     # IBP for ResNet18
│   ├── milp_verifier.py    # MILP verifier for ResNet18
│   └── milp_patch_verifier.py  # MILP Patch verifier
└── tests/                  # Unit tests
```

## Setup

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Running Tests

To run the test suite:

```bash
poetry run pytest
```

## Verification Methods

The project implements several verification methods:

1. Interval Bound Propagation (IBP)
2. Mixed Integer Linear Programming (MILP)
3. MILP with Patch Verification

Each verifier is implemented as a separate class in the `verifiers` directory. 
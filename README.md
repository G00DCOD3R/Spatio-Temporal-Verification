# Spatio-Temporal Verification

This project implements clean, robust verification methods for 3D Convolutional Neural Networks (3D-CNNs) on spatio-temporal video classification and sequence prediction tasks.

## Model Checkpoints

Place each checkpoint in `spatio_temporal_verification/experiments/<experiment>/checkpoints/`

For example, place the MedMNIST 32-frame checkpoint at:

```text
spatio_temporal_verification/experiments/medmnist/checkpoints/best_medmnist_32_model.pth
```

Use the model architecture and preprocessing settings that match the checkpoint. 

## Project Structure

```
spatio_temporal_verification/
├── notebooks/                       # Experimental Jupyter Notebooks
│   ├── MNISTVideoTraining.ipynb
│   ├── MNISTVideoVerification-Closed_form_IBP_Lipschitz_LJ_FULLVALSET.ipynb
│   ├── MNISTVideoVerification-Closed_form_ablation.ipynb
│   ├── Simple3DCNN-MedMNISTSynapse3D-Verification-ClosedForm-Lipschitz.ipynb
│   ├── UCF-101-TrainingVerification-3DCNN-RGB-32_32_30_rgb_5_class-LIPS.ipynb
│   ├── UdacitySteeringAngle-Training-3DCNN.ipynb
│   └── UdacitySteeringAngle-Verification-3DCNN-ClosedFormLips.ipynb
├── experiments/                     # Experiment implementations by domain
│   ├── mnist/                       # MNIST video classification experiment
│   │   ├── checkpoints/
│   │   ├── models/                  # MNISTSimple3DCNN, MNISTResNet18
│   │   └── utils/                   # MNISTVideoDataset, training functions
│   ├── ucf_101/                     # UCF-101 action recognition experiment
│   │   ├── checkpoints/
│   │   ├── models/                  # UCF101SimpleCNN
│   │   └── utils/                   # UCF-101 dataset utilities and training
│   ├── udacity_steering_angle/      # Udacity steering angle prediction experiment
│   │   ├── checkpoints/
│   │   ├── models/                  # UdacitySteeringAngle_3DCNN, CNN3D
│   │   └── utils/                   # UdacitySteeringAngleDataset, training
│   └── medmnist/                    # MedMNIST 3D Synapse experiment
│       ├── checkpoints/
│       ├── models/                  # MedMNISTSimple3DCNN
│       └── utils/                   # MedMNIST3DDataset, Transform3D
├── verifiers/                       # Centralized robustness verifier implementations
│   ├── base_verifier.py             # Base verifier class with recursive atomic layer extraction
│   ├── ibp_verifier.py              # Interval Bound Propagation (IBP)
│   ├── closed_form_ibp_verifier.py   # Analytical closed-form 1st layer bounds + IBP
│   ├── closed_form_lipschitz_verifier.py # Analytical 1st layer bounds + L_∞ to L_2 Lipschitz propagation
│   ├── closed_form_lownerjohn_verifier.py # Analytical 1st layer bounds + Löwner-John ellipsoid propagation
│   ├── milp_ibp_verifier.py         # MILP + IBP hybrid verifier (requires Gurobi)
│   └── milp_ibp_patch_verifier.py   # MILP + IBP patch verifier (requires Gurobi)
├── main.py                          # Unified CLI entry point for running experiments & verifiers
└── tests/                           # Pytest test suite
```

> [!NOTE]
> All reference experimental code, interactive evaluations, and ablation studies reside inside the [notebooks](https://github.com/SherwinVarghese/Spatio-Temporal-Verification/tree/main/spatio_temporal_verification/notebooks) directory.

## Setup

This project uses standard Python dependency environments or Poetry:

```bash
# Using conda
conda activate vas

# Or using poetry
poetry install
```

## Running Experiments

You can run training and verification for any experiment using the unified CLI in `main.py`:

```bash
PYTHONPATH=spatio_temporal_verification python spatio_temporal_verification/main.py [options]
```

### CLI Arguments

- `--experiment`: Experiment domain to run (`mnist`, `ucf101`, `udacity`, `medmnist`, or `all`). Default: `mnist`.
- `--mode`: Execution mode (`verify` or `train`). Default: `verify`.
- `--verifier`: Robustness verification method (`closed_form_ibp`, `closed_form_lips`, `closed_form_lj`, `ibp`, `milp`, `milp_patch`). Default: `closed_form_ibp`.
- `--eps`: Perturbation magnitude(s) (e.g. `--eps 0.001 0.005`). Default: `0.001`.
- `--num_samples`: Number of test samples to verify. Default: `20`.
- `--batch_size`: Batch size for dataloaders. Default: `8`.
- `--dataset_path`: Custom dataset path (optional).

### Usage Examples

1. **Run MNIST Verification with Closed-Form IBP**:
   ```bash
   PYTHONPATH=spatio_temporal_verification python spatio_temporal_verification/main.py --experiment mnist --mode verify --verifier closed_form_ibp --num_samples 50
   ```

2. **Run UCF-101 Verification with Closed-Form Lipschitz**:
   ```bash
   PYTHONPATH=spatio_temporal_verification python spatio_temporal_verification/main.py --experiment ucf101 --mode verify --verifier closed_form_lips --num_samples 20
   ```

3. **Run Udacity Steering Angle Verification with Closed-Form Löwner-John**:
   ```bash
   PYTHONPATH=spatio_temporal_verification python spatio_temporal_verification/main.py --experiment udacity --mode verify --verifier closed_form_lj --num_samples 20
   ```

4. **Run MedMNIST 3D Verification**:
   ```bash
   PYTHONPATH=spatio_temporal_verification python spatio_temporal_verification/main.py --experiment medmnist --mode verify --verifier closed_form_ibp --num_samples 20
   ```

5. **Run Verification across ALL Experiments**:
   ```bash
   PYTHONPATH=spatio_temporal_verification python spatio_temporal_verification/main.py --experiment all --mode verify --num_samples 10
   ```

## Verification Methods

1. **Closed-Form IBP (`ClosedFormIBPVerifier`)**: Derives exact analytical bounds for structured $L_\infty$ perturbations on the first Conv3D layer, followed by Interval Bound Propagation (IBP) through subsequent layers.
2. **Closed-Form Lipschitz (`ClosedFormLipschitzVerifier`)**: Combines analytical first-layer bounds with $L_\infty \to L_2$ norm conversion and spectral-norm Lipschitz propagation.
3. **Closed-Form Löwner-John (`ClosedFormLownerJohnVerifier`)**: Combines analytical first-layer bounds with Löwner-John ellipsoid propagation (sampling, analytical, or inscribed ellipsoids).
4. **Standard IBP (`IBPVerifier`)**: Standard coordinate-wise interval bound propagation.
5. **MILP-IBP Hybrid (`MILPIBPVerifier`)**: Exact Mixed-Integer Linear Programming for the first layer, followed by IBP.

## Running Tests

To execute the test suite:

```bash
PYTHONPATH=spatio_temporal_verification pytest
```

## Datasets

All datasets used in the experiments are publicly available or can be downloaded from the [Zenodo Repository](https://zenodo.org/records/15426146).

## Publications

If you use this codebase or verification methods in your research, please cite our paper:

> **Hybrid Robustness Verification for Spatio-Temporal Neural Networks**  
> Sherwin Varghese, Matthew Wicker, and Alessio Lomuscio.  
> *AI Verification*, Springer Nature Switzerland, 2027.  
> Link: [https://link.springer.com/chapter/10.1007/978-3-032-32357-6_1](https://link.springer.com/chapter/10.1007/978-3-032-32357-6_1)

### BibTeX

```bibtex
@InProceedings{10.1007/978-3-032-32357-6_1,
author="Varghese, Sherwin
and Wicker, Matthew
and Lomuscio, Alessio",
editor="Avni, Guy
and Schilling, Christian",
title="Hybrid Robustness Verification for Spatio-Temporal Neural Networks",
booktitle="AI Verification",
year="2027",
publisher="Springer Nature Switzerland",
address="Cham",
pages="3--35",
isbn="978-3-032-32357-6"
}
```

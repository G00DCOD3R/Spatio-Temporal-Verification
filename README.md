# Spatio-temporal Verification

This project implements various verification methods for neural networks, focusing on spatio-temporal properties.

## Project Structure

```
spatio-temporal-verification/
├── experiments/
│   ├── mnist/
│   │   ├── checkpoints/      # Model checkpoints
│   │   └── models/           # MNIST-specific model implementations
│   └── udacity_steering_angle/
│       ├── checkpoints/      # Steering angle model checkpoints
│       ├── models/           # Steering angle model implementations
│       └── utils/            # Dataset, training and verification utilities
├── models/
│   └── resnet18.py          # Base ResNet18 implementation
├── utils/                   # Utility functions
├── verifiers/
│   ├── base_verifier.py     # Base verifier class
│   ├── ibp_verifier.py      # IBP for neural networks
│   ├── milp_ibp_verifier.py # MILP-IBP hybrid verifier
│   ├── milp_ibp_patch_verifier.py  # MILP-IBP Patch verifier
│   ├── closed_form_ibp_verifier.py # Closed form IBP hybrid verifier
│   ├── closed_form_lipschitz_verifier.py # Closed form Lipschitz hybrid verifier
│   └── closed_form_lownerjohn_verifier.py  # Closed form Löwner-John hybrid verifier
└── tests/                   # Unit tests
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

## Datasets
All the datasets used for the experiments are either publicly available or can be downloaded from the [Zenodo](https://zenodo.org/records/15426146) repository.

## Verification Methods

The project implements several verification methods:

1. Interval Bound Propagation (IBP)
2. Spatio-Temporal Bound Propagation (STBP)
3. Spatio-Temporal Bound Propagation with Patch Verification

Each verifier is implemented as a separate class in the `verifiers` directory.

## Experiments

### MNIST Video Classification
This experiment focuses on classifying MNIST digits from video sequences using 3D CNNs and verifying their robustness.

### Udacity Steering Angle Prediction

This module implements a 3D CNN model for predicting steering angles from sequences of frames from the Udacity self-driving car dataset, along with robustness verification methods.

#### Model Architecture

The architecture consists of a 3D CNN with:
- Two 3D convolutional layers with batch normalization and ReLU activations
- Adaptive pooling to handle variable input sizes
- Three fully connected layers with dropout for regularization

#### Training

To train the model, you can use either:

1. The standalone script:

```bash
python experiments/udacity_steering_angle/train_udacity_model.py \
    --image_dir /path/to/udacity/center \
    --labels_file /path/to/udacity/final_example.csv \
    --output_dir ./outputs \
    --img_x 90 \
    --img_y 120 \
    --num_frames 30 \
    --num_classes 3 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --epochs 50
```

2. The main module with environment variables:

```bash
TRAIN_UDACITY=1 \
UDACITY_DATASET_PATH=/path/to/udacity/dataset \
NUM_FRAMES=30 \
NUM_CLASSES=3 \
BATCH_SIZE=8 \
EPOCHS=50 \
python spatio-temporal-verification/main.py
```

#### Verification

The module includes three verification methods:
1. **IBP (Interval Bound Propagation)**: Fast but less precise bounds.
2. **MILP-IBP**: Mixed Integer Linear Programming for the first layer, followed by IBP for the rest.
3. **MILP-IBP-Patch**: Similar to MILP-IBP but only applying perturbations to specific patches (e.g., objects detected in frames).

##### Running Verification

1. Using the standalone script:

```bash
python experiments/udacity_steering_angle/verify_udacity_model.py \
    --dataset_path /path/to/udacity \
    --bounding_box_path /path/to/udacity/bounding_boxes \
    --model_path /path/to/model/checkpoint.pth \
    --num_frames 30 \
    --num_classes 3 \
    --epsilons 0.00001 0.0001 0.001 \
    --frames 0 1 2 3 4 \
    --batch_size 1 \
    --num_batches 1 \
    --run_ibp \
    --run_milp_ibp \
    --run_milp_ibp_patch
```

2. Using the main module with environment variables:

```bash
VERIFY_UDACITY=1 \
UDACITY_DATASET_PATH=/path/to/udacity/dataset \
BOUNDING_BOX_PATH=/path/to/udacity/bounding_boxes \
MODEL_PATH=/path/to/model/checkpoint.pth \
NUM_FRAMES=30 \
NUM_CLASSES=3 \
EPSILONS="0.00001,0.0001,0.001" \
FRAMES="0,1,2,3,4" \
BATCH_SIZE=1 \
NUM_BATCHES=1 \
RUN_IBP=1 \
RUN_MILP_IBP=1 \
RUN_MILP_IBP_PATCH=1 \
python spatio-temporal-verification/main.py
```

##### Bounding Box Format

For patch-based verification (MILP-IBP-Patch), you need to provide bounding box files in JSON format. Each file should be named after the corresponding frame ID with a "_tracked.json" suffix and contain the following structure:

```json
{
  "xyxyn": [
    [x1, y1, x2, y2],  // Normalized coordinates for object 1
    [x1, y1, x2, y2]   // Normalized coordinates for object 2
  ]
}
```

##### Dataset

The Udacity steering angle dataset should be organized as follows:
The dataset is available to download from the [Zenodo](https://zenodo.org/records/15426146) repository.


```
udacity_dataset/
├── center/                # Directory containing center camera images
│   ├── frame_id1.jpg
│   ├── frame_id2.jpg
│   └── ...
├── final_example.csv      # CSV file containing frame IDs and steering angles
└── bounding_boxes/        # Directory containing bounding box JSON files
    ├── frame_id1_tracked.json
    ├── frame_id2_tracked.json
    └── ...
```

The CSV file should have at least these columns:
- `frame_id`: Identifier for each frame
- `steering_angle`: Continuous steering angle value between -1.0 and 1.0 
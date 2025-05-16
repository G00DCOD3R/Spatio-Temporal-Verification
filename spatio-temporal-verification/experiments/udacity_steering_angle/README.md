# Udacity Steering Angle Prediction and Verification

This module implements a 3D CNN model for predicting steering angles from sequences of frames from the Udacity self-driving car dataset, along with robustness verification methods.

## Model Architecture

The architecture consists of a 3D CNN with:
- Two 3D convolutional layers with batch normalization and ReLU activations
- Adaptive pooling to handle variable input sizes
- Three fully connected layers with dropout for regularization

## Training

To train the model, you can use either:

1. The standalone script:

```bash
python train_udacity_model.py \
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
python ../../main.py
```

## Verification

The module includes three verification methods:
1. **IBP (Interval Bound Propagation)**: Fast but less precise bounds.
2. **MILP-IBP**: Mixed Integer Linear Programming for the first layer, followed by IBP for the rest.
3. **MILP-IBP-Patch**: Similar to MILP-IBP but only applying perturbations to specific patches (e.g., objects detected in frames).

### Running Verification

#### 1. Using the standalone script:

```bash
python verify_udacity_model.py \
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

#### 2. Using the main module with environment variables:

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
python ../../main.py
```

### Bounding Box Format

For patch-based verification (MILP-IBP-Patch), you need to provide bounding box files in JSON format. Each file should be named after the corresponding frame ID with a "_tracked.json" suffix and contain the following structure:

```json
{
  "xyxyn": [
    [x1, y1, x2, y2],  // Normalized coordinates for object 1
    [x1, y1, x2, y2]   // Normalized coordinates for object 2
  ]
}
```

## Dataset

The Udacity steering angle dataset should be organized as follows:

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

## Acknowledgements

This implementation is based on the Udacity self-driving car dataset and includes verification methods like Interval Bound Propagation (IBP) and Mixed Integer Linear Programming (MILP) to verify the robustness of the model against perturbations. 
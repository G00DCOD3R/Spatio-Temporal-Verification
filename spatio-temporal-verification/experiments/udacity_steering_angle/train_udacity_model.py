import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# Import model and dataset
from experiments.udacity_steering_angle.models.udacity_3dcnn import UdacitySteeringAngle_3DCNN
from experiments.udacity_steering_angle.utils.udacity_dataset import UdacitySteeringAngleDataset
from experiments.udacity_steering_angle.utils.training import train, evaluate


def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up dataset parameters
    img_x = args.img_x
    img_y = args.img_y
    num_frames = args.num_frames
    num_classes = args.num_classes

    # Create datasets
    train_dataset = UdacitySteeringAngleDataset(
        image_dir=args.image_dir,
        labels_file=args.labels_file,
        train=True,
        test_size=args.test_size,
        resize_shape=(img_x, img_y),
        num_classes=num_classes,
        num_frames=num_frames
    )
    
    test_dataset = UdacitySteeringAngleDataset(
        image_dir=args.image_dir,
        labels_file=args.labels_file,
        train=False,
        test_size=args.test_size,
        resize_shape=(img_x, img_y),
        num_classes=num_classes,
        num_frames=num_frames
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # Create model
    model = UdacitySteeringAngle_3DCNN(
        num_frames=num_frames,
        img_x=img_x,
        img_y=img_y,
        fc_hidden1=256,
        fc_hidden2=128,
        dropout=args.dropout,
        num_classes=num_classes,
        in_channels=3  # RGB images
    )
    
    model = model.to(device)
    
    # Set up model checkpoint path
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    model_path = os.path.join(args.output_dir, "checkpoints", "best_udacity_model.pth")
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    print("Training Udacity steering angle model...")
    train_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation set
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_path=model_path,
        epochs=args.epochs,
        scheduler=scheduler,
        early_stopping_patience=args.patience,
        clip_grad=args.clip_grad
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_acc, all_preds, all_targets = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoints", "final_udacity_model.pth"))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Udacity steering angle prediction model")
    
    # Dataset parameters
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels_file", type=str, required=True, help="CSV file with frame IDs and steering angles")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for model checkpoints")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to use for testing")
    
    # Model parameters
    parser.add_argument("--img_x", type=int, default=90, help="Image width")
    parser.add_argument("--img_y", type=int, default=120, help="Image height")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames in each sequence")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of steering angle classes")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    main(args) 
    
# TRAIN_UDACITY=1 UDACITY_DATASET_PATH=/path/to/udacity/dataset python main.py
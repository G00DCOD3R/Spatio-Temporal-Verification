"""Train the authors' CNN on SynapseMNIST3D downsampled to 32³.

Run from the repository root:
    python -m spatio_temporal_verification.experiments.medmnist.train_medmnist_model
    Add --check-data to inspect the dataset without training.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from medmnist import SynapseMNIST3D
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import MedMNISTSimple3DCNN


def downsample(volume):
    # MedMNIST supplies (1, 64, 64, 64) intensities already scaled to [0, 1].
    volume = torch.as_tensor(volume, dtype=torch.float32)
    return F.interpolate(
        volume.unsqueeze(0), size=(32, 32, 32), mode="area"
    ).squeeze(0)


def load_data(root):
    root.mkdir(parents=True, exist_ok=True)
    return {
        split: SynapseMNIST3D(
            split=split, root=str(root), size=64,
            download=True, transform=downsample,
        )
        for split in ("train", "val", "test")
    }


def run_epoch(model, loader, optimizer, description="Training"):
    model.train()
    loss_sum = 0.0
    correct = 0
    with tqdm(loader, desc=description, unit="batch") as progress:
        for volumes, labels in progress:
            labels = labels.flatten().long()
            optimizer.zero_grad(set_to_none=True)  # Clear previous gradients.
            logits = model(volumes)  # Predict two class scores per volume.
            loss = F.cross_entropy(logits, labels)  # Measure prediction error.
            loss.backward()  # Compute gradients for the model's weights.
            optimizer.step()  # Update weights using those gradients.

            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, description="Evaluation"):
    model.eval()
    loss_sum = 0.0
    confusion = torch.zeros(2, 2, dtype=torch.long)
    with tqdm(loader, desc=description, unit="batch") as progress:
        for volumes, labels in progress:
            labels = labels.flatten().long()
            logits = model(volumes)
            loss = F.cross_entropy(logits, labels)
            loss_sum += loss.item() * len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            predictions = logits.argmax(dim=1)
            confusion += torch.bincount(
                labels * 2 + predictions, minlength=4
            ).reshape(2, 2)
    accuracy = confusion.diag().sum().item() / len(loader.dataset)
    return loss_sum / len(loader.dataset), accuracy, confusion


def train(datasets, args):
    torch.manual_seed(args.seed)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    model = MedMNISTSimple3DCNN(
        in_channels=1, num_classes=2, input_frames=32, resize_shape=(32, 32)
    )
    best_loss = float("inf")
    if args.resume:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}", flush=True)
        best_loss, _, _ = evaluate(model, loaders["val"], description="Initial validation")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], optimizer, description=f"Epoch {epoch}/{args.epochs} train"
        )
        val_loss, val_acc, _ = evaluate(
            model, loaders["val"], description=f"Epoch {epoch}/{args.epochs} val"
        )
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train loss={train_loss:.4f}, accuracy={train_acc:.2%}; "
            f"val loss={val_loss:.4f}, accuracy={val_acc:.2%}", flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best checkpoint: {args.checkpoint}", flush=True)

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    test_loss, test_acc, confusion = evaluate(model, loaders["test"], description="Test")
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp_min(1)
    print(f"Test: loss={test_loss:.4f}, accuracy={test_acc:.2%}")
    print(f"Recall: class 0={recall[0]:.2%}, class 1={recall[1]:.2%}")
    print(f"Confusion matrix (rows=true, columns=predicted):\n{confusion}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/medmnist"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume", action="store_true",
        help="Load --checkpoint weights and train for additional --epochs (fresh Adam optimizer).",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(__file__).parent / "checkpoints/best_medmnist_32_model.pth",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.lr <= 0:
        parser.error("batch size, epochs, and learning rate must be positive")
    if args.resume and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    datasets = load_data(args.data_dir)
    train(datasets, args)


if __name__ == "__main__":
    main()

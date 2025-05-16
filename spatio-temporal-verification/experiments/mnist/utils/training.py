import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List
import os
import itertools
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import numpy as np


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate the model on the given data loader"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    return acc, avg_loss


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, criterion: nn.Module, 
          device: torch.device, model_path: str, epochs: int = 10) -> None:
    """Train the model and save the best checkpoint based on validation accuracy
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer to use for training
        criterion: Loss function
        device: Device to train on
        model_path: Path where the best model checkpoint will be saved
        epochs: Number of epochs to train for
    """
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # Validation phase
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch}:')
        print(f'  Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.2f}%')
        print(f'  Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.2f}%')
        
        # Save model if this is the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%! Saving model...')
            torch.save(model.state_dict(), model_path)
    
    print(f'\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best model saved to {model_path}') 
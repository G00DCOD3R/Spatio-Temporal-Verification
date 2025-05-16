import os
import torch
import numpy as np
import time
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=None):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for a nice progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        
        # Apply gradient clipping if specified
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item(), "acc": 100. * correct / total})
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model performance on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to track gradients
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate validation statistics
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train(model, train_loader, val_loader, optimizer, criterion, device, model_path, epochs=50, scheduler=None, early_stopping_patience=10, clip_grad=None):
    """Train the model with early stopping and learning rate scheduling."""
    
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, clip_grad)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Check if we have a new best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), model_path)
            print(f'  New best model saved with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'  EarlyStopping: {patience_counter}/{early_stopping_patience}')
            
            # Check if we should stop training
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Load the best model
    model.load_state_dict(torch.load(model_path))
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'epochs_completed': epoch + 1,
        'training_time': training_time
    }


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Store predictions and targets for confusion matrix
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # No need to track gradients
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate test statistics
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_targets) 
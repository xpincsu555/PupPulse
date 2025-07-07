"""
Training script for PupPulse dog emotion classification model.
Implements training loop with validation, early stopping, and model checkpointing.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from typing import Dict, Tuple
import json

# Import custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import create_data_loaders, load_config
from models.model import create_model, get_device, count_parameters

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs with no improvement to wait
            min_delta (float): Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_epoch(model: nn.Module, train_loader, criterion, optimizer, 
                   device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Progress bar for training
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100.0 * correct_predictions / total_samples
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy

def validate_model(model: nn.Module, val_loader, criterion, 
                  device: torch.device) -> Tuple[float, float]:
    """
    Validate the model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Progress bar for validation
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_loss = running_loss / len(val_loader)
            current_acc = 100.0 * correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy

def save_checkpoint(model: nn.Module, optimizer, epoch: int, 
                   train_loss: float, val_loss: float, val_acc: float,
                   config: Dict, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_acc: Validation accuracy
        config: Configuration dictionary
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': config
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")

def train_model(config: Dict):
    """
    Main training function.
    
    Args:
        config (Dict): Configuration dictionary
    """
    # Setup output directory
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, config['paths']['log_file'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting PupPulse emotion classification training")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Get device
    device = get_device(config)
    
    # Create data loaders
    logging.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Create model
    logging.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Log model information
    num_params = count_parameters(model)
    logging.info(f"Model has {num_params:,} trainable parameters")
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta']
    )
    
    # Setup TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'runs'))
    
    # Training loop
    best_val_acc = 0.0
    epochs = config['training']['epochs']
    
    logging.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logging.info(f"Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, val_acc,
                config, best_model_path
            )
            logging.info(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, val_acc,
                config, checkpoint_path
            )
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    save_checkpoint(
        model, optimizer, epoch, train_loss, val_loss, val_acc,
        config, final_model_path
    )
    
    writer.close()
    logging.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train PupPulse emotion classification model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start training
    train_model(config)

if __name__ == '__main__':
    main()

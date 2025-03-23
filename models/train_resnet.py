import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import logging

from initialization import init_weights, get_gradient_norm
from utils import load_config, set_seed, get_device, load_cifar10, save_model

logger = logging.getLogger(__name__)

def get_resnet_model():
    """
    Get ResNet-18 model adapted for CIFAR-10.
    The first convolutional layer is modified for 32x32 input.
    """
    model = models.resnet18(pretrained=False)
    
    # Modify first layer for CIFAR-10 (32x32 RGB images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool as CIFAR images are small
    
    # Modify final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        tuple: (average loss, accuracy, average gradient norm)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        
        # Compute gradient norm for monitoring stability
        grad_norm = get_gradient_norm(model)
        gradient_norms.append(grad_norm)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Calculate metrics
    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    avg_grad_norm = sum(gradient_norms) / len(gradient_norms)
    
    return epoch_loss, accuracy, avg_grad_norm

def validate(model, test_loader, criterion, device):
    """
    Evaluate model on validation data.
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate metrics
    val_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return val_loss, accuracy

def train_resnet(init_method, config):
    """
    Train ResNet-18 on CIFAR-10 with specified initialization method.
    
    Args:
        init_method: Initialization method ('xavier', 'he', 'lecun', 'random')
        config: Configuration dictionary
        
    Returns:
        dict: Training metrics
    """
    device = get_device()
    set_seed(config['seed'])
    
    # Load datasets
    train_loader, test_loader = load_cifar10(config)
    
    # Initialize model
    model = get_resnet_model()
    model = init_weights(model, init_method)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if config['cnn']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['cnn']['learning_rate'],
            weight_decay=config['cnn']['weight_decay']
        )
    else:  # SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['cnn']['learning_rate'],
            momentum=config['cnn']['momentum'],
            weight_decay=config['cnn']['weight_decay']
        )
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    grad_norms = []
    
    logger.info(f"Starting ResNet-18 training with {init_method} initialization")
    
    # Training loop
    for epoch in range(config['cnn']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['cnn']['epochs']}")
        
        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Grad Norm: {grad_norm:.4f}"
        )
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        grad_norms.append(grad_norm)
    
    # Save model
    if config['cnn']['save_model']:
        save_dir = os.path.join(config['cnn']['results_dir'], 'models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"resnet18_{init_method}.pth")
        save_model(model, save_path)
    
    # Collect all metrics
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'grad_norms': grad_norms,
        'final_accuracy': val_accuracies[-1],
        'convergence_epoch': next(
            (i+1 for i, acc in enumerate(val_accuracies) if acc >= 0.95 * val_accuracies[-1]),
            config['cnn']['epochs']
        )
    }
    
    logger.info(f"ResNet-18 training with {init_method} initialization completed")
    logger.info(f"Final accuracy: {metrics['final_accuracy']:.2f}%")
    logger.info(f"Convergence epoch: {metrics['convergence_epoch']}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--init', type=str, required=True, 
                        choices=['xavier', 'he', 'lecun', 'random'],
                        help='Weight initialization method')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_resnet(args.init, config)

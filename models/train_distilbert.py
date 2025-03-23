import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import logging

from initialization import init_transformer, get_gradient_norm
from utils import load_config, set_seed, get_device, load_imdb, save_model

logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Train transformer for one epoch.
    
    Returns:
        tuple: (average loss, accuracy, average gradient norm)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm for monitoring stability
        grad_norm = get_gradient_norm(model)
        gradient_norms.append(grad_norm)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Statistics
        running_loss += loss.item() * batch["input_ids"].size(0)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total += batch["labels"].size(0)
        correct += (predictions == batch["labels"]).sum().item()
    
    # Calculate metrics
    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    avg_grad_norm = sum(gradient_norms) / len(gradient_norms)
    
    return epoch_loss, accuracy, avg_grad_norm

def validate(model, test_loader, device):
    """
    Evaluate transformer model on validation data.
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Statistics
            running_loss += loss.item() * batch["input_ids"].size(0)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total += batch["labels"].size(0)
            correct += (predictions == batch["labels"]).sum().item()
    
    # Calculate metrics
    val_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return val_loss, accuracy

def train_distilbert(init_method, config):
    """
    Fine-tune DistilBERT on IMDB with specified initialization method.
    
    Args:
        init_method: Initialization method ('xavier', 'he', 'lecun', 'random')
        config: Configuration dictionary
        
    Returns:
        dict: Training metrics
    """
    device = get_device()
    set_seed(config['seed'])
    
    # Load datasets
    train_loader, test_loader = load_imdb(config)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        config['transformer']['model'], 
        num_labels=2
    )
    
    # Apply initialization to classifier head
    model = init_transformer(model, init_method)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['transformer']['learning_rate'],
        weight_decay=config['transformer']['weight_decay']
    )
    
    # Compute total training steps
    total_steps = len(train_loader) * config['transformer']['epochs']
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    grad_norms = []
    
    logger.info(f"Starting DistilBERT training with {init_method} initialization")
    
    # Training loop
    for epoch in range(config['transformer']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['transformer']['epochs']}")
        
        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, None, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, device)
        
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
    if config['transformer']['save_model']:
        save_dir = os.path.join(config['transformer']['results_dir'], 'models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"distilbert_{init_method}")
        model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    # Calculate gradient explosion percentage reduction compared to random
    grad_explosion_reduction = 0
    if init_method != 'random' and 'random' in config['init_methods']:
        # This will be calculated later in evaluate.py when all methods are run
        grad_explosion_reduction = None
    
    # Collect all metrics
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'grad_norms': grad_norms,
        'final_accuracy': val_accuracies[-1],
        'avg_grad_norm': sum(grad_norms) / len(grad_norms),
        'max_grad_norm': max(grad_norms),
        'grad_explosion_reduction': grad_explosion_reduction
    }
    
    logger.info(f"DistilBERT training with {init_method} initialization completed")
    logger.info(f"Final accuracy: {metrics['final_accuracy']:.2f}%")
    logger.info(f"Average gradient norm: {metrics['avg_grad_norm']:.4f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DistilBERT on IMDB')
    parser.add_argument('--init', type=str, required=True, 
                        choices=['xavier', 'he', 'lecun', 'random'],
                        help='Weight initialization method')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_distilbert(args.init, config)

import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset
from transformers import DistilBertTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Get the device to run training on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_directories(config):
    """Create necessary directories for saving results."""
    dirs = [
        config['cnn']['results_dir'],
        config['transformer']['results_dir'],
        config['evaluation']['plot_dir'],
        config['evaluation']['log_dir']
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    logger.info(f"Created directories: {dirs}")

def load_cifar10(config):
    """Load and prepare CIFAR-10 dataset."""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=config['cnn']['data_dir'],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=config['cnn']['data_dir'],
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['cnn']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['test_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"CIFAR-10 dataset loaded: {len(train_dataset)} training, {len(test_dataset)} testing")
    return train_loader, test_loader

def load_imdb(config):
    """Load and prepare IMDB dataset."""
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config['transformer']['model'])
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config['transformer']['max_length']
        )
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Rename label column
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Set format for PyTorch
    tokenized_datasets.set_format(type="torch")
    
    # Create data loaders
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=config['transformer']['batch_size'],
        shuffle=True
    )
    
    test_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=config['evaluation']['test_batch_size'],
        shuffle=False
    )
    
    logger.info(f"IMDB dataset loaded: {len(tokenized_datasets['train'])} training, {len(tokenized_datasets['test'])} testing")
    return train_loader, test_loader

def save_model(model, path):
    """Save model weights."""
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def load_model(model, path):
    """Load model weights."""
    model.load_state_dict(torch.load(path))
    logger.info(f"Model loaded from {path}")
    return model

def plot_training_curves(metrics, title, save_path):
    """
    Plot training curves for different initialization methods.
    
    Args:
        metrics: Dict of metric values by initialization method
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for method, values in metrics.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2, label=method)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")

def plot_comparison_bar(metrics, title, save_path):
    """
    Create a bar plot for comparing different initialization methods.
    
    Args:
        metrics: Dict of metric values by initialization method
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    methods = list(metrics.keys())
    values = [metrics[method] for method in methods]
    
    barplot = sns.barplot(x=methods, y=values)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        barplot.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.title(title, fontsize=16)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Bar plot saved to {save_path}")

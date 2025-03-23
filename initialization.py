import torch
import torch.nn as nn
import numpy as np
import math
from transformers import DistilBertForSequenceClassification

def xavier_init(module):
    """
    Xavier/Glorot initialization for linear and convolutional layers.
    Maintains variance across forward/backward passes.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def he_init(module):
    """
    He initialization, specifically designed for ReLU activations.
    Scales variance by a factor of 2/n.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def lecun_init(module):
    """
    LeCun initialization, optimized for tanh activations.
    Scales by 1/n.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        scale = 1 / math.sqrt(fan_in)
        nn.init.uniform_(module.weight, -scale, scale)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def random_init(module):
    """
    Random initialization from normal distribution.
    Used as a baseline.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_weights(model, method):
    """
    Apply specified weight initialization method to model.
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'he', 'lecun', or 'random')
    
    Returns:
        Initialized model
    """
    if method == 'xavier':
        model.apply(xavier_init)
    elif method == 'he':
        model.apply(he_init)
    elif method == 'lecun':
        model.apply(lecun_init)
    elif method == 'random':
        model.apply(random_init)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return model

def init_transformer(model, method):
    """
    Apply initialization to transformer model (DistilBERT).
    Only initializes the classification head and final layers since
    pre-trained weights are generally kept for the main transformer.
    
    Args:
        model: Transformer model (DistilBERT)
        method: Initialization method
    
    Returns:
        Initialized model
    """
    # Only initialize classifier head and final layers
    if isinstance(model, DistilBertForSequenceClassification):
        init_weights(model.pre_classifier, method)
        init_weights(model.classifier, method)
    
    return model

def get_gradient_norm(model):
    """
    Calculate the L2 norm of gradients for gradient stability analysis.
    
    Args:
        model: PyTorch model with gradients
    
    Returns:
        float: L2 norm of all gradients
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

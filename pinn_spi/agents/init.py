"""Weight initialization helpers"""
import torch
import torch.nn as nn

def xavier_init(module, gain=1.0):
    """Apply Xavier initialization to a module"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def he_init(module):
    """Apply He initialization to a module"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
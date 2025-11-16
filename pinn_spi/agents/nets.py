"""
Neural Network Building Blocks

This module provides fundamental neural network components used across all algorithms.
Implementations exactly match the original experimental scripts for reliability.
"""

import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """Multi-layer perceptron with Xavier initialization - EXACTLY matching original"""
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], act=nn.ReLU, xavier_gain=5.0):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.model = nn.Sequential(*layers)

        # Xavier initialization with custom gain (exactly as in original)
        if xavier_gain:
            for m in self.model:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=xavier_gain)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
from __future__ import annotations
from typing import Protocol, Dict, Any
import torch

Tensor = torch.Tensor

class Algorithm(Protocol):
    """Base protocol for all RL algorithms"""

    def to(self, device: torch.device) -> "Algorithm":
        """Move algorithm to device"""
        ...

    def train_mode(self) -> None:
        """Set algorithm to training mode"""
        ...

    def eval_mode(self) -> None:
        """Set algorithm to evaluation mode"""
        ...

    def act(self, x: Tensor, deterministic: bool = False) -> Tensor:
        """Sample action from policy"""
        ...

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Update algorithm parameters and return metrics"""
        ...

    def save(self, path: str) -> None:
        """Save algorithm state"""
        ...

    def load(self, path: str, map_location="cpu") -> None:
        """Load algorithm state"""
        ...

    def get_training_steps(self, exp_cfg: Dict[str, Any]) -> int:
        """Get total training steps for this algorithm.

        Args:
            exp_cfg: Experiment configuration

        Returns:
            Total number of steps/iterations
        """
        ...

"""
Additional evaluation utilities.
This module can be extended with more sophisticated evaluation metrics
and analysis tools as needed.
"""
import torch
import numpy as np
from typing import Dict, List
from .rollout import evaluate_policy

@torch.no_grad()
def compute_statistics(returns: torch.Tensor) -> Dict[str, float]:
    """Compute statistics of trajectory returns"""
    returns_np = returns.cpu().numpy()
    return {
        "mean": float(np.mean(returns_np)),
        "std": float(np.std(returns_np)),
        "min": float(np.min(returns_np)),
        "max": float(np.max(returns_np)),
        "median": float(np.median(returns_np))
    }

@torch.no_grad()
def evaluate_with_stats(env, algo, num_traj, traj_len, x0=None, gamma=None):
    """Evaluate policy and return detailed statistics"""
    result = evaluate_policy(env, algo, num_traj, traj_len, x0, gamma)
    stats = compute_statistics(result["G"])
    result.update(stats)
    return result
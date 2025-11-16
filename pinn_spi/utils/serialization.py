"""
Utilities for saving and loading experiment data, checkpoints, etc.
"""
import os
import pickle
import torch
from typing import Any, Dict

def save_pickle(obj: Any, path: str):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """Load object from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)

def save_checkpoint(
    algo,
    step: int,
    optimizer_states: Dict[str, Any] = None,
    path: str = None,
    **kwargs
):
    """Save training checkpoint"""
    checkpoint = {
        "step": step,
        "algo_state": algo.save if hasattr(algo, 'save') else None,
        "optimizer_states": optimizer_states,
        **kwargs
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    """Load training checkpoint"""
    return torch.load(path, map_location=map_location)
"""
Replay Buffer Implementation

This module provides experience replay functionality for off-policy algorithms like SAC.
The buffer stores transitions (s, a, r, s') and supports random sampling for training.
"""

from collections import deque
import random
import torch

class Replay:
    """Experience replay buffer for SAC"""
    def __init__(self, maxlen=1_000_000, device="cpu"):
        self.buf = deque(maxlen=maxlen)
        self.device = device

    def add(self, s, a, r, s2, done=None):
        """
        Add transition (s, a, r, s2, done) to buffer.

        Args:
            s: State
            a: Action
            r: Reward
            s2: Next state
            done: Done flag (optional, defaults to 0.0)
        """
        if done is None:
            done = torch.tensor(0.0, device=self.device)
        self.buf.append((s.detach(), a.detach(), r.detach(), s2.detach(), done.detach()))

    def sample(self, batch_size):
        """Sample batch of transitions"""
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return dict(
            s=torch.stack(s).to(self.device),
            a=torch.stack(a).to(self.device),
            r=torch.stack(r).to(self.device),
            s2=torch.stack(s2).to(self.device),
            done=torch.stack(done).to(self.device),
        )

    def __len__(self):
        return len(self.buf)
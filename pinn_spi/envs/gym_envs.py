"""
Optional wrappers for Gymnasium environments.
This allows integration with standard gym environments like CartPole and Pendulum.
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from .base import StochasticControlEnv, BoxSpec

Tensor = torch.Tensor

@dataclass
class GymEnvSpec:
    """Specification for a gym environment wrapper"""
    d: int  # state dimension
    m: int  # action dimension
    max_x: float  # state bounds
    max_u: float  # action bounds
    sigma: float  # noise level
    dt: float  # time step
    env_name: str  # gym environment name

class GymEnvWrapper(StochasticControlEnv):
    """
    Wrapper to adapt gymnasium environments to our StochasticControlEnv interface.
    Adds Gaussian noise to make deterministic environments stochastic.
    """
    def __init__(self, spec: GymEnvSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self._X = BoxSpec(
            low=torch.full((spec.d,), -spec.max_x, device=device),
            high=torch.full((spec.d,), spec.max_x, device=device),
        )
        self._U = BoxSpec(
            low=torch.full((spec.m,), -spec.max_u, device=device),
            high=torch.full((spec.m,), spec.max_u, device=device),
        )
        self._I = torch.eye(spec.d, device=device)

    @property
    def d(self):
        return self.spec.d

    @property
    def m(self):
        return self.spec.m

    @property
    def dt(self):
        return self.spec.dt

    def clip_state(self, x: Tensor) -> Tensor:
        return torch.max(torch.min(x, self._X.high), self._X.low)

    def clip_action(self, u: Tensor) -> Tensor:
        return torch.max(torch.min(u, self._U.high), self._U.low)

    def b(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Drift function - should be implemented for specific environments.
        For now, returns zero drift as placeholder.
        """
        return torch.zeros_like(x)

    def sigma(self, x: Tensor, u: Tensor) -> Tensor:
        """Constant diagonal noise"""
        S = self.spec.sigma * self._I
        return S.unsqueeze(0).expand(x.shape[0], self.d, self.d)

    def r(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Reward function - should be implemented for specific environments.
        For now, returns zero reward as placeholder.
        """
        return torch.zeros(x.shape[0], device=self.device)
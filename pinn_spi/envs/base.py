from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple
import torch

Tensor = torch.Tensor

class StochasticControlEnv(Protocol):
    @property
    def d(self) -> int:
        """State dimension"""
        ...

    @property
    def m(self) -> int:
        """Action dimension"""
        ...

    @property
    def dt(self) -> float:
        """Time step"""
        ...

    def clip_state(self, x: Tensor) -> Tensor:
        """Clip state to valid bounds"""
        ...

    def clip_action(self, u: Tensor) -> Tensor:
        """Clip action to valid bounds"""
        ...

    def get_state_sample_bounds(self) -> Tuple[Tensor, Tensor]:
        """
        Get bounds for sampling states during PINN training.

        Returns:
            (low, high): Tensors of shape [d] with lower and upper bounds per dimension
        """
        ...

    def get_action_sample_bounds(self) -> Tuple[Tensor, Tensor]:
        """
        Get bounds for sampling actions during PINN training.

        Returns:
            (low, high): Tensors of shape [m] with lower and upper bounds per dimension
        """
        ...

    def b(self, x: Tensor, u: Tensor) -> Tensor:
        """Drift function f(x,u) [B,d]"""
        ...

    def sigma(self, x: Tensor, u: Tensor) -> Tensor:
        """Diffusion matrix [B,d,d] (can be diag/const)"""
        ...

    def r(self, x: Tensor, u: Tensor) -> Tensor:
        """Running reward [B]"""
        ...

    # Discrete action support
    @property
    def is_discrete_action(self) -> bool:
        """
        Whether environment has discrete actions.

        Returns:
            True if actions are discrete (e.g., CartPole with {0, 1}),
            False if actions are continuous (e.g., LQR)
        """
        ...

    @property
    def num_discrete_actions(self) -> int:
        """
        Number of discrete actions (only valid if is_discrete_action=True).

        Returns:
            Number of discrete actions (e.g., 2 for CartPole)
        """
        ...

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Compute deterministic next state (without noise): x' = x + b(x,u)*dt

        This method is used for:
        1. PINN-PI/PINN-SPI discrete action enumeration (needs gradients)
        2. Evaluating multiple discrete actions in parallel

        For discrete actions, u should be integer action indices.
        Must be differentiable w.r.t. x for gradient-based methods.

        Args:
            x: State tensor [B, d]
            u: Action tensor [B, m] for continuous or [B] for discrete

        Returns:
            next_x: Next state [B, d] (deterministic, no noise)
        """
        ...

    def sync_state(self, x: Tensor) -> None:
        """
        Synchronize internal gymnasium environment state with external SDE state.

        This method is called after Euler-Maruyama steps to ensure that any
        wrapped gymnasium environment (e.g., CartPole, HalfCheetah) has its
        internal state synchronized with the externally-computed SDE state.

        For environments without internal state management (LQR, Pendulum),
        this is a no-op.

        Args:
            x: Current state tensor [B, d] from Euler-Maruyama integration
                For batch_size=1 (SAC/PPO): [1, d]
                For vectorized (PPO): [num_envs, d]
        """
        ...

@dataclass
class BoxSpec:
    low: Tensor
    high: Tensor
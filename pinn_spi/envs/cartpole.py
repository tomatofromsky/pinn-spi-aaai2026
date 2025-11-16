"""
CartPole Environment with SDE Dynamics

This module implements a CartPole environment for discrete control tasks with
Gaussian noise injection following SDE dynamics. The system uses Gymnasium's
CartPole-v1 as the base environment with added stochasticity.

State space: [x, x_dot, theta, theta_dot] (4D)
Action space: {0, 1} (discrete left/right force)
Dynamics: Deterministic CartPole physics + Gaussian noise
Reward: +1 per timestep survived (original CartPole-v1 convention)
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from .base import StochasticControlEnv, BoxSpec

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Warning: gymnasium not installed. CartPole environment will not be available.")

Tensor = torch.Tensor


@dataclass
class CartPoleSpec:
    """CartPole environment specifications"""
    sigma: float          # Diffusion coefficient for Gaussian noise
    exponent: float       # Discount rate (rho) for value function
    dt: float            # Timestep (tau in Gymnasium)

    # Gymnasium CartPole parameters (loaded from env)
    force_mag: float
    gravity: float
    masspole: float
    masscart: float
    length: float
    x_threshold: float
    theta_threshold_radians: float


class CartPoleEnv(StochasticControlEnv):
    """
    CartPole environment with SDE dynamics for PINN-PI.

    Wraps Gymnasium's CartPole-v1 and adds Gaussian noise to simulate
    stochastic dynamics: dx = f(x,u)dt + σdW
    """

    def __init__(self, spec: CartPoleSpec, device: torch.device, render_mode: Optional[str] = None):
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for CartPole environment. Install with: pip install gymnasium")

        self.spec = spec
        self.device = device

        # Create Gymnasium environment
        self.gym_env = gym.make('CartPole-v1', render_mode=render_mode)

        # State space: [x, x_dot, theta, theta_dot]
        # Sampling bounds from original script
        self._X = BoxSpec(
            low=torch.tensor([-4.8, -5.0, -0.418, -5.0], device=device),
            high=torch.tensor([4.8, 5.0, 0.418, 5.0], device=device),
        )

        # Action space: {0, 1}
        self._U = BoxSpec(
            low=torch.tensor([0], device=device),
            high=torch.tensor([1], device=device),
        )

        self._I = torch.eye(4, device=device)

    @property
    def d(self) -> int:
        """State dimension"""
        return 4

    @property
    def m(self) -> int:
        """Action dimension (discrete: 0 or 1)"""
        return 1

    @property
    def dt(self) -> float:
        """Timestep"""
        return self.spec.dt

    def clip_state(self, x: Tensor) -> Tensor:
        """Clip state to valid bounds"""
        return torch.max(torch.min(x, self._X.high), self._X.low)

    def clip_action(self, u: Tensor) -> Tensor:
        """Clip action to {0, 1}"""
        return torch.clamp(torch.round(u), 0, 1).long()

    def get_state_sample_bounds(self):
        """
        Return bounds for sampling states during PINN training.

        Uses wider bounds than termination thresholds for better coverage:
        - x: [-4.8, 4.8] (2x termination threshold of 2.4)
        - x_dot: [-5.0, 5.0] (unbounded in env, practical limit)
        - theta: [-0.418, 0.418] (2x termination threshold of 0.2095)
        - theta_dot: [-5.0, 5.0] (unbounded in env, practical limit)
        """
        return self._X.low, self._X.high

    def get_action_sample_bounds(self):
        """
        Return bounds for sampling actions during PINN training.

        CartPole has discrete actions {0, 1}, so return [0, 1].
        """
        return self._U.low, self._U.high

    def b(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Drift dynamics: f(x, u) = (next_state - current_state) / dt

        Implements CartPole physics equations using PyTorch for batched computation.

        Args:
            x: State tensor [B, 4] with [x, x_dot, theta, theta_dot]
            u: Action tensor [B] or [B, 1] with discrete actions {0, 1}

        Returns:
            Drift term [B, 4]
        """
        # Ensure u is 1D
        if u.dim() == 2:
            u = u.squeeze(-1)
        u = u.long()

        # Extract state components
        x_pos = x[:, 0]      # Cart position
        x_dot = x[:, 1]      # Cart velocity
        theta = x[:, 2]      # Pole angle
        theta_dot = x[:, 3]  # Pole angular velocity

        # Action to force: 0 -> -force_mag, 1 -> +force_mag
        force = torch.where(u == 1,
                          torch.tensor(self.spec.force_mag, device=self.device),
                          torch.tensor(-self.spec.force_mag, device=self.device))

        # CartPole physics
        g = self.spec.gravity
        m = self.spec.masspole
        M = self.spec.masscart
        L = self.spec.length

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        total_mass = m + M
        polemass_length = m * L

        # Compute accelerations
        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (g * sintheta - costheta * temp) / (
            L * (4.0/3.0 - m * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Compute next state using Euler integration
        x_pos_new = x_pos + self.spec.dt * x_dot
        x_dot_new = x_dot + self.spec.dt * xacc
        theta_new = theta + self.spec.dt * theta_dot
        theta_dot_new = theta_dot + self.spec.dt * thetaacc

        next_state = torch.stack([x_pos_new, x_dot_new, theta_new, theta_dot_new], dim=1)

        # Return drift: (next_state - current_state) / dt
        return (next_state - x) / self.spec.dt

    def sigma(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Diffusion term: σ * I

        Returns isotropic Gaussian noise with coefficient σ.

        Args:
            x: State tensor [B, 4]
            u: Action tensor [B] or [B, 1]

        Returns:
            Diffusion matrix [B, 4, 4]
        """
        batch_size = x.shape[0]
        S = self.spec.sigma * self._I
        return S.unsqueeze(0).expand(batch_size, self.d, self.d)

    def r(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Reward function: +1 per timestep survived (original CartPole convention)

        ORIGINAL CARTPOLE CONVENTION:
        - Reward: +1 for each timestep the pole remains balanced
        - Reward: 0 if pole falls or cart goes out of bounds
        - Goal: Maximize cumulative reward (survive as long as possible)
        - Range: {0, 1}

        This differs from LQR/Pendulum convention (where max=0) but matches
        the standard CartPole-v1 reward structure from Gymnasium/OpenAI Gym.

        Episode terminates if:
        - Cart position |x| > x_threshold (2.4)
        - Pole angle |theta| > theta_threshold (0.2095 rad ≈ 12°)

        Args:
            x: State tensor [B, 4]
            u: Action tensor [B] or [B, 1]

        Returns:
            Reward tensor [B]
        """
        # Ensure u is correct shape for dynamics
        if u.dim() == 2:
            u_for_dynamics = u.squeeze(-1)
        else:
            u_for_dynamics = u
        u_for_dynamics = u_for_dynamics.long()

        # Compute next state to check termination
        drift = self.b(x, u)
        next_state = x + drift * self.spec.dt

        x_pos = next_state[:, 0]
        theta = next_state[:, 2]

        # Check termination conditions
        terminated = (
            (x_pos < -self.spec.x_threshold) |
            (x_pos > self.spec.x_threshold) |
            (theta < -self.spec.theta_threshold_radians) |
            (theta > self.spec.theta_threshold_radians)
        )

        # Reward: +1 if not terminated (surviving), 0 if terminated
        # ORIGINAL CartPole convention: maximize cumulative reward
        return (~terminated).float()

    # Discrete action support
    @property
    def is_discrete_action(self) -> bool:
        """CartPole has discrete actions {0, 1}"""
        return True

    @property
    def num_discrete_actions(self) -> int:
        """CartPole has 2 discrete actions"""
        return 2

    # NOTE: dynamics() method commented out - currently unused in the framework
    # The method was designed for PINN-PI discrete action enumeration, but current
    # implementation only uses PINN-SPI with stochastic policy sampling.
    # Keeping the code for reference in case PINN-PI is implemented in the future.
    #
    # def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
    #     """
    #     Compute deterministic next state (without noise): x' = x + b(x,u)*dt
    #
    #     This is the differentiable dynamics function used for:
    #     1. PINN-PI discrete action enumeration (evaluates both actions)
    #     2. PINN-SPI policy improvement (evaluates all actions)
    #
    #     Args:
    #         x: State tensor [B, 4] with [x, x_dot, theta, theta_dot]
    #         u: Action tensor [B] with discrete actions {0, 1}
    #
    #     Returns:
    #         next_x: Next state [B, 4] (deterministic, no noise added)
    #     """
    #     # Ensure u is 1D and integer
    #     if u.dim() == 2:
    #         u = u.squeeze(-1)
    #     u = u.long()
    #
    #     # Extract state components
    #     x_pos = x[:, 0]      # Cart position
    #     x_dot = x[:, 1]      # Cart velocity
    #     theta = x[:, 2]      # Pole angle
    #     theta_dot = x[:, 3]  # Pole angular velocity
    #
    #     # Action to force: 0 -> -force_mag, 1 -> +force_mag
    #     force = torch.where(u == 1,
    #                       torch.tensor(self.spec.force_mag, device=self.device, dtype=x.dtype),
    #                       torch.tensor(-self.spec.force_mag, device=self.device, dtype=x.dtype))
    #
    #     # CartPole physics parameters
    #     g = self.spec.gravity
    #     m = self.spec.masspole
    #     M = self.spec.masscart
    #     L = self.spec.length
    #     dt = self.spec.dt
    #
    #     costheta = torch.cos(theta)
    #     sintheta = torch.sin(theta)
    #     total_mass = m + M
    #     polemass_length = m * L
    #
    #     # Compute accelerations (same as TODO/2_4_cartpole.md)
    #     temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    #     thetaacc = (g * sintheta - costheta * temp) / (
    #         L * (4.0/3.0 - m * costheta**2 / total_mass)
    #     )
    #     xacc = temp - polemass_length * thetaacc * costheta / total_mass
    #
    #     # Euler integration (deterministic, no noise)
    #     x_pos_new = x_pos + dt * x_dot
    #     x_dot_new = x_dot + dt * xacc
    #     theta_new = theta + dt * theta_dot
    #     theta_dot_new = theta_dot + dt * thetaacc
    #
    #     return torch.stack([x_pos_new, x_dot_new, theta_new, theta_dot_new], dim=1)

    def reset(self):
        """Reset environment to initial state"""
        init = self.gym_env.reset()
        state = init[0] if isinstance(init, tuple) else init
        return state

    def step(self, action):
        """Take environment step with Gymnasium"""
        return self.gym_env.step(int(action))

    def render(self):
        """Render environment"""
        return self.gym_env.render()

    def close(self):
        """Close environment"""
        self.gym_env.close()

    def sync_state(self, x: Tensor) -> None:
        """
        Synchronize gymnasium environment internal state with SDE state.

        After Euler-Maruyama integration computes x2 externally, this method
        updates the internal gym_env.state to match, ensuring consistency for
        rendering and potential gym API usage.

        Args:
            x: State tensor [B, 4] with [x_pos, x_dot, theta, theta_dot]
               For SAC/single-env: [1, 4]
               For PPO/vectorized: [num_envs, 4]
        """
        if x.shape[0] == 1:
            # Single environment (SAC, PINN-SPI evaluation)
            # Convert from torch tensor to numpy array for gym
            state_np = x.squeeze(0).cpu().numpy()
            self.gym_env.unwrapped.state = state_np
        else:
            # Vectorized environments (PPO)
            # Only sync the first environment (gym_env is single instance)
            # In vectorized setting, each env should have its own gym instance
            # For now, sync first env only as a best-effort approach
            state_np = x[0].cpu().numpy()
            self.gym_env.unwrapped.state = state_np


def load_cartpole_from_yaml(cfg: dict, dt: float, device: torch.device,
                           render_mode: Optional[str] = None, verbose: bool = False) -> CartPoleEnv:
    """
    Load CartPole environment from YAML configuration.

    Args:
        cfg: Configuration dictionary with 'params' key
        dt: Timestep (tau from Gymnasium)
        device: PyTorch device
        render_mode: Gymnasium render mode ('rgb_array', 'human', or None)
        verbose: Print loading information

    Returns:
        CartPoleEnv instance
    """
    if not GYMNASIUM_AVAILABLE:
        raise ImportError("gymnasium is required. Install with: pip install gymnasium")

    p = cfg["params"]

    if verbose:
        print(f"\n=== Loading CartPole environment ===")
        print(f"Device: {device}")
        print(f"Time step (dt): {dt}")
        print(f"Render mode: {render_mode}")

    # Create temporary gym environment to extract parameters
    temp_env = gym.make('CartPole-v1')
    unwrapped = temp_env.unwrapped

    spec = CartPoleSpec(
        sigma=float(p["sigma"]),
        exponent=float(p["exponent"]),
        dt=dt,
        force_mag=unwrapped.force_mag,
        gravity=unwrapped.gravity,
        masspole=unwrapped.masspole,
        masscart=unwrapped.masscart,
        length=unwrapped.length,
        x_threshold=unwrapped.x_threshold,
        theta_threshold_radians=unwrapped.theta_threshold_radians,
    )

    temp_env.close()

    if verbose:
        print(f"\nEnvironment Parameters:")
        print(f"  Noise level (σ): {spec.sigma}")
        print(f"  Discount rate (ρ): {spec.exponent}")
        print(f"  Force magnitude: {spec.force_mag}")
        print(f"  Cart mass: {spec.masscart}")
        print(f"  Pole mass: {spec.masspole}")
        print(f"  Pole length: {spec.length}")
        print(f"  Position threshold: ±{spec.x_threshold}")
        print(f"  Angle threshold: ±{spec.theta_threshold_radians:.4f} rad")
        print(f"=== CartPole environment loaded ===\n")

    return CartPoleEnv(spec, device, render_mode)

"""
Pendulum Environment for PINN-SPI

This module implements the Pendulum-v1 environment from Gymnasium for use with
PINN-PI, PINN-SPI, and SAC algorithms. The environment uses continuous actions
and SDE dynamics with Gaussian noise.

Key Components:
- PendulumSpec: Environment parameters matching Gymnasium's Pendulum-v1
- PendulumEnv: Main environment class implementing StochasticControlEnv protocol
- Dynamics: Vectorized pendulum physics with Euler integration
- Reward: Standard pendulum cost function: -(θ² + 0.1·θ̇² + 0.001·u²)
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from ..envs.base import StochasticControlEnv, Tensor


@dataclass
class PendulumSpec:
    """Pendulum environment specification matching Gymnasium Pendulum-v1."""

    # Physics parameters (matching Gymnasium defaults)
    g: float = 10.0  # Gravity (m/s²)
    m: float = 1.0  # Mass (kg)
    l: float = 1.0  # Length (m)
    max_speed: float = 8.0  # Maximum angular velocity (rad/s)
    max_torque: float = 2.0  # Maximum torque (N⋅m)

    # Time discretization
    dt: float = 0.05  # Time step (s)

    # Noise parameters
    sigma: float = 0.005  # Diffusion coefficient (diagonal covariance)

    # Cost parameters
    rho: float = 0.5  # Discount rate for HJB equation


class PendulumEnv(StochasticControlEnv):
    """
    Pendulum environment with SDE dynamics.

    State: x = [θ, θ̇] where θ is angle from vertical (rad), θ̇ is angular velocity (rad/s)
    Action: u ∈ [-max_torque, max_torque] (continuous torque)

    Dynamics: dθ̇ = [(3g/2l)sin(θ) + (3/ml²)u]dt + σdW
              dθ = θ̇ dt

    Cost: r(x,u) = -(θ² + 0.1·θ̇² + 0.001·u²)
    """

    def __init__(self, spec: PendulumSpec, device: torch.device, verbose: bool = False):
        self.spec = spec
        self.device = device

        if verbose:
            print("\n=== Loading Pendulum environment ===")
            print(f"Device: {device}")
            print(f"Time step (dt): {spec.dt}")
            print(f"\nEnvironment Parameters:")
            print(f"  Gravity (g): {spec.g} m/s²")
            print(f"  Mass (m): {spec.m} kg")
            print(f"  Length (l): {spec.l} m")
            print(f"  Max speed: {spec.max_speed} rad/s")
            print(f"  Max torque: {spec.max_torque} N⋅m")
            print(f"  Noise level (σ): {spec.sigma}")
            print(f"  Discount rate (ρ): {spec.rho}")
            print("=== Pendulum environment loaded ===\n")

    @property
    def d(self) -> int:
        """State dimension: [θ, θ̇]"""
        return 2

    @property
    def m(self) -> int:
        """Action dimension: [torque]"""
        return 1

    @property
    def dt(self) -> float:
        """Time step"""
        return self.spec.dt

    # ============================================================================
    # Protocol Methods (StochasticControlEnv)
    # ============================================================================

    def clip_state(self, x: Tensor) -> Tensor:
        """
        Clip state to valid bounds.

        For Pendulum, we don't clip states - angles can wrap around.
        """
        return x

    def clip_action(self, u: Tensor) -> Tensor:
        """Clip action to torque bounds."""
        return torch.clamp(u, -self.spec.max_torque, self.spec.max_torque)

    def sigma(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Diffusion matrix (diagonal Gaussian noise).

        Args:
            x: States [B, d]
            u: Actions [B, m]

        Returns:
            diffusion: [B, d, d] diagonal diffusion matrix
        """
        batch_size = x.shape[0]
        # Diagonal diffusion: σI
        S = self.spec.sigma * torch.eye(self.d, device=self.device)
        return S.unsqueeze(0).expand(batch_size, self.d, self.d)

    def b(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Drift term of SDE dynamics: b(x,u)

        Computes deterministic dynamics: dx = b(x,u)dt

        Args:
            x: States [B, 2] containing [θ, θ̇]
            u: Actions [B, 1] containing torque

        Returns:
            drift: [B, 2] drift vector
        """
        # Extract state components
        theta = x[:, 0]  # Angle from vertical
        theta_dot = x[:, 1]  # Angular velocity

        # Clamp torque to bounds
        torque = torch.clamp(u.squeeze(-1), -self.spec.max_torque, self.spec.max_torque)

        # Angular acceleration: θ̈ = (3g/2l)sin(θ) + (3/ml²)u
        g, m, l = self.spec.g, self.spec.m, self.spec.l
        theta_ddot = (3 * g / (2 * l)) * torch.sin(theta) + (3.0 / (m * l**2)) * torque

        # Clamp angular velocity to max_speed
        theta_dot_new = torch.clamp(
            theta_dot + theta_ddot * self.spec.dt,
            -self.spec.max_speed,
            self.spec.max_speed,
        )

        # Drift: [dθ/dt, dθ̇/dt] = [θ̇, θ̈]
        drift = torch.stack(
            [theta_dot, (theta_dot_new - theta_dot) / self.spec.dt], dim=1
        )

        return drift

    def angle_normalize(self, x: Tensor):
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

    def r(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Instantaneous cost function (negative reward).

        Standard pendulum cost: r(x,u) = -(θ² + 0.1·θ̇² + 0.001·u²)

        Args:
            x: States [B, 2]
            u: Actions [B, 1]

        Returns:
            rewards: [B] instantaneous rewards
        """
        theta = x[:, 0]
        theta_dot = x[:, 1]

        # https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/classic_control/pendulum.py#L129
        normalized_angle = self.angle_normalize(theta)

        # Cost function (negative because we maximize reward but minimize cost)
        cost = normalized_angle**2 + 0.1 * theta_dot**2 + 0.001 * u.squeeze(-1) ** 2

        return -cost  # Negative cost = reward

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Deterministic next state (without noise): x' = x + b(x,u)·dt

        Used for discrete action evaluation in PINN-PI.
        Must be differentiable w.r.t. x for gradient-based methods.

        Args:
            x: States [B, 2]
            u: Actions [B, 1] or [B] (torque)

        Returns:
            next_state: [B, 2] next state without noise
        """
        # Extract state components
        theta = x[:, 0]
        theta_dot = x[:, 1]

        # Handle both [B, 1] and [B] shaped actions
        torque = u.squeeze(-1) if u.dim() > 1 else u
        torque = torch.clamp(torque, -self.spec.max_torque, self.spec.max_torque)

        # Angular acceleration
        g, m, l, dt = self.spec.g, self.spec.m, self.spec.l, self.spec.dt
        theta_ddot = (3 * g / (2 * l)) * torch.sin(theta) + (3.0 / (m * l**2)) * torque

        # Euler integration
        theta_dot_new = theta_dot + theta_ddot * dt
        theta_dot_new = torch.clamp(
            theta_dot_new, -self.spec.max_speed, self.spec.max_speed
        )
        theta_new = theta + theta_dot_new * dt

        # Next state
        next_state = torch.stack([theta_new, theta_dot_new], dim=1)

        return next_state

    def get_state_sample_bounds(self) -> tuple[Tensor, Tensor]:
        """
        Get bounds for sampling states during training.

        Returns:
            (low, high): Lower and upper bounds for each state dimension
        """
        # Sample states within reasonable bounds
        # θ ∈ [-π, π], θ̇ ∈ [-max_speed, max_speed]
        low = torch.tensor([-np.pi, -self.spec.max_speed], device=self.device)
        high = torch.tensor([np.pi, self.spec.max_speed], device=self.device)

        return low, high

    def get_action_sample_bounds(self) -> tuple[Tensor, Tensor]:
        """
        Get bounds for sampling actions during training.

        Returns:
            (low, high): Lower and upper bounds for each action dimension
        """
        low = torch.tensor([-self.spec.max_torque], device=self.device)
        high = torch.tensor([self.spec.max_torque], device=self.device)

        return low, high

    @property
    def is_discrete_action(self) -> bool:
        """Pendulum uses continuous actions."""
        return False

    @property
    def num_discrete_actions(self) -> int:
        """Not applicable for continuous actions."""
        raise NotImplementedError("Pendulum has continuous actions")

    def sync_state(self, x: Tensor) -> None:
        """
        Synchronize internal state (no-op for Pendulum).

        Pendulum environment has no internal state to synchronize - it only provides
        dynamics functions. State is managed externally by the training loop.

        Args:
            x: State tensor [B, 2] (unused)
        """
        pass  # No-op: Pendulum has no internal state

    def reset(
        self,
        batch_size: int | None = 1,
        *,
        x0: Tensor | None = None,
        mode: str = "l2_ball",  # ["l2_ball", "small_uniform", "full_uniform"]
        l2_radius: float = 0.1,  # used when mode == "l2_ball"
        small_angle: float = 0.05,  # used when mode == "small_uniform"
        small_speed: float = 0.05,  # used when mode == "small_uniform"
        angle_range: float = torch.pi,  # used when mode == "full_uniform"
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> Tensor:
        """
        Returns a batch of initial states (stateless reset).

        Args:
            batch_size: number of states to sample (ignored if x0 is provided with a batch dim)
            x0: if provided, returned (after device/dtype/normalization)
            mode:
                - "l2_ball": sample near upright and project onto L2 ball of radius `l2_radius`
                - "small_uniform": θ ~ U[-small_angle, small_angle], θ̇ ~ U[-small_speed, small_speed]
                - "full_uniform":  θ ~ U[-angle_range, angle_range], θ̇ ~ U[-max_speed, max_speed]
            l2_radius: radius for L2-ball projection (angle/speed treated in same units)
            small_angle/small_speed: tight uniform box around the origin
            angle_range: symmetric range for angle when using "full_uniform"
            seed: optional seed for reproducibility
            dtype: tensor dtype
            requires_grad: whether returned tensor requires grad

        Returns:
            x: tensor of shape [B, 2] on `self.device`, representing [θ, θ̇]
        """
        # If user provided x0, just sanitize, normalize, and return
        if x0 is not None:
            x = x0.to(device=self.device, dtype=dtype)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # [2] -> [1,2]
            # normalize angle and clamp speed
            x[:, 0] = self.angle_normalize(x[:, 0])
            x[:, 1] = torch.clamp(x[:, 1], -self.spec.max_speed, self.spec.max_speed)
            x.requires_grad_(requires_grad)
            return x

        assert (
            batch_size is not None and batch_size >= 1
        ), "batch_size must be positive when x0 is None"

        # RNG (torch) for reproducibility
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)

        if mode == "small_uniform":
            theta = torch.empty(batch_size, device=self.device, dtype=dtype).uniform_(
                -small_angle, small_angle, generator=g
            )
            theta_dot = torch.empty(
                batch_size, device=self.device, dtype=dtype
            ).uniform_(-small_speed, small_speed, generator=g)

        elif mode == "full_uniform":
            theta = torch.empty(batch_size, device=self.device, dtype=dtype).uniform_(
                -angle_range, angle_range, generator=g
            )
            theta_dot = torch.empty(
                batch_size, device=self.device, dtype=dtype
            ).uniform_(-self.spec.max_speed, self.spec.max_speed, generator=g)

        elif mode == "l2_ball":
            # sample from N(0, I), scale to be within L2 ball radius
            z = torch.randn(batch_size, 2, device=self.device, dtype=dtype, generator=g)
            norms = torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(1e-12)
            # scale to uniform radius in [0, l2_radius] (use r ~ U[0,1]^(1/d))
            # for 2D, r ~ sqrt(U[0,1]) gives uniform area; here we keep simple U[0,1] for slight center bias
            r = (
                torch.rand(batch_size, 1, device=self.device, dtype=dtype, generator=g)
                * l2_radius
            )
            x = (z / norms) * r
            theta, theta_dot = x[:, 0], x[:, 1]

        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        # sanitize: wrap angle and clamp speed
        theta = self.angle_normalize(theta)
        theta_dot = torch.clamp(theta_dot, -self.spec.max_speed, self.spec.max_speed)

        x = torch.stack([theta, theta_dot], dim=1)
        x.requires_grad_(requires_grad)
        return x


def project_onto_l2_ball(state: np.ndarray, radius: float = 0.1) -> np.ndarray:
    """
    Project state onto L2 ball for initial state sampling.

    This ensures initial states are close to the origin (upright position)
    for more stable training and evaluation.

    Args:
        state: State vector [θ, θ̇]
        radius: L2 ball radius (default: 0.1)

    Returns:
        projected_state: State projected onto L2 ball of given radius
    """
    norm = np.linalg.norm(state)
    if norm <= radius:
        return state
    else:
        return state * (radius / norm)


def load_pendulum_from_yaml(
    cfg: dict, dt: float, device: torch.device, verbose: bool = False
) -> PendulumEnv:
    """
    Load Pendulum environment from YAML configuration.

    Args:
        cfg: Configuration dictionary from YAML
        dt: Time step (from experiment config)
        device: PyTorch device
        verbose: Print environment details

    Returns:
        env: Configured PendulumEnv instance
    """
    # Create spec from config with defaults
    spec = PendulumSpec(
        g=float(cfg.get("g", 10.0)),
        m=float(cfg.get("m", 1.0)),
        l=float(cfg.get("l", 1.0)),
        max_speed=float(cfg.get("max_speed", 8.0)),
        max_torque=float(cfg.get("max_torque", 2.0)),
        dt=dt,
        sigma=float(cfg.get("sigma", 0.005)),
        rho=float(cfg.get("rho", 0.5)),
    )

    return PendulumEnv(spec, device, verbose=verbose)

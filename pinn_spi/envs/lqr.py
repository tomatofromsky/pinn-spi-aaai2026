"""
Linear Quadratic Regulator (LQR) Environment

This module implements a stochastic LQR environment for continuous control tasks.
The system follows SDE dynamics: dx = (Ax + Bu)dt + σdW, where:
- A is the drift matrix
- B is the control matrix
- σ is the diffusion coefficient
- dW is Brownian motion

The cost function is quadratic: r(x,u) = -(r_x_ratio*||x||² + r_u_ratio*||u||²)
"""

from __future__ import annotations
import os
import numpy as np
import torch
from dataclasses import dataclass
from .base import StochasticControlEnv, BoxSpec

Tensor = torch.Tensor

@dataclass
class LQRSpec:
    d: int
    m: int
    max_x: float
    max_u: float
    sigma: float
    r_x_ratio: float
    r_u_ratio: float
    A: Tensor
    B: Tensor
    dt: float
    reset_radius: float | None = None
    terminate_radius: float | None = None
    horizon: int | None = None

class LQREnv(StochasticControlEnv):
    def __init__(self, spec: LQRSpec, device: torch.device):
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
        self.horizon = spec.horizon
        self.terminate_radius = spec.terminate_radius
        if spec.reset_radius is not None:
            self.reset_radius = float(spec.reset_radius)
        else:
            self.reset_radius = float(min(1.0, spec.max_x))

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

    def get_state_sample_bounds(self):
        """Return bounds for sampling states during PINN training"""
        return self._X.low, self._X.high

    def get_action_sample_bounds(self):
        """Return bounds for sampling actions during PINN training"""
        return self._U.low, self._U.high

    def b(self, x: Tensor, u: Tensor) -> Tensor:
        return x @ self.spec.A.T + u @ self.spec.B

    def sigma(self, x: Tensor, u: Tensor) -> Tensor:
        S = self.spec.sigma * self._I
        return S.unsqueeze(0).expand(x.shape[0], self.d, self.d)

    def r(self, x: Tensor, u: Tensor) -> Tensor:
        cx = self.spec.r_x_ratio * (x**2).sum(-1)
        cu = self.spec.r_u_ratio * (u**2).sum(-1)
        return -(cx + cu)

    # Discrete action support (LQR has continuous actions)
    @property
    def is_discrete_action(self) -> bool:
        """LQR has continuous actions"""
        return False

    @property
    def num_discrete_actions(self) -> int:
        """Not applicable for continuous action spaces"""
        raise NotImplementedError("LQR has continuous actions, not discrete")

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Compute deterministic next state (without noise): x' = x + b(x,u)*dt

        For LQR: x' = x + (Ax + Bu)*dt

        Args:
            x: State tensor [B, d]
            u: Action tensor [B, m]

        Returns:
            next_x: Next state [B, d] (deterministic, no noise)
        """
        drift = self.b(x, u)
        return x + drift * self.spec.dt

    def sample_initial_state(self, batch_size: int, device: torch.device | None = None) -> Tensor:
        """Sample initial states from a uniform box [-reset_radius, reset_radius]^d."""
        if device is None:
            device = self.device
        return torch.zeros(batch_size, self.d, device=device)

    def sync_state(self, x: Tensor) -> None:
        """
        Synchronize internal state (no-op for LQR).

        LQR environment has no internal state to synchronize - it only provides
        dynamics functions. State is managed externally by the training loop.

        Args:
            x: State tensor [B, d] (unused)
        """
        pass  # No-op: LQR has no internal state

def load_lqr_from_yaml(cfg: dict, dt: float, device: torch.device, verbose: bool = False) -> LQREnv:
    d, m = cfg["params"]["d"], cfg["params"]["m"]
    p = cfg["params"]

    if verbose:
        print(f"\n=== Loading LQR {d}D system ===")
        print(f"Device: {device}")
        print(f"Time step (dt): {dt}")

    try:
        # Check if separate A_path and B_path are specified (for 20D)
        if "B_path" in p:
            if verbose:
                print(f"Loading separate matrix files:")
                print(f"  A matrix: {p['A_path']}")
                print(f"  B matrix: {p['B_path']}")
            A_np = np.load(p["A_path"])
            B_np = np.load(p["B_path"])
        else:
            # Load from .npz file with keys (for 5D and 10D)
            if verbose:
                print(f"Loading combined matrix file:")
                print(f"  File: {p['A_path']}")
                print(f"  A key: {p['A_key']}, B key: {p['B_key']}")
            data = np.load(p["A_path"])
            A_np, B_np = data[p["A_key"]], data[p["B_key"]]

        if verbose:
            print(f"✅ Successfully loaded matrices from file(s)")

    except FileNotFoundError as e:
        print(f"Warning: Matrix files not found ({e}). Using random matrices.")
        A_np, B_np = np.random.randn(d, d), np.random.randn(d, m)

        if verbose:
            print(f"⚠️  Using randomly generated matrices")

    # Print matrix information if verbose
    if verbose:
        print(f"\nMatrix Information:")
        print(f"  A shape: {A_np.shape}")
        print(f"  B shape: {B_np.shape}")
        print(f"  A eigenvalues range: [{np.min(np.real(np.linalg.eigvals(A_np))):.4f}, "
              f"{np.max(np.real(np.linalg.eigvals(A_np))):.4f}]")
        print(f"  A spectral radius: {np.max(np.abs(np.linalg.eigvals(A_np))):.4f}")
        print(f"  A matrix norm (Frobenius): {np.linalg.norm(A_np, 'fro'):.4f}")
        print(f"  B matrix norm (Frobenius): {np.linalg.norm(B_np, 'fro'):.4f}")

        # Print first few elements for verification
        print(f"\nA matrix (top-left 3x3):")
        print(A_np[:3, :3])
        print(f"\nB matrix (top-left 3x3):")
        print(B_np[:3, :min(3, m)])

    spec = LQRSpec(
        d=d, m=m,
        max_x=float(p["max_x"]), max_u=float(p["max_u"]),
        sigma=float(p["sigma"]),
        r_x_ratio=float(p["r_x_ratio"]), r_u_ratio=float(p["r_u_ratio"]),
        A=torch.tensor(A_np, dtype=torch.float32, device=device),
        B=torch.tensor(B_np, dtype=torch.float32, device=device),
        dt=dt,
        reset_radius=float(p.get("reset_radius", p["max_x"])),
        terminate_radius=p.get("terminate_radius"),
        horizon=p.get("horizon"),
    )

    if verbose:
        print(f"\nEnvironment Parameters:")
        print(f"  State bounds: [-{spec.max_x}, {spec.max_x}]")
        print(f"  Action bounds: [-{spec.max_u}, {spec.max_u}]")
        print(f"  Noise level (σ): {spec.sigma}")
        print(f"  Reward ratios - r_x: {spec.r_x_ratio}, r_u: {spec.r_u_ratio}")
        print(f"=== LQR system loaded ===\n")

    return LQREnv(spec, device)

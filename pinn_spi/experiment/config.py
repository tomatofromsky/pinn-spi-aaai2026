"""
Configuration dataclasses for training and evaluation.

Centralizes parameter handling and default values to avoid scattered .get() calls
and deep nesting in the training loop.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""

    # Core evaluation parameters
    num_traj: int
    traj_len: int
    every_steps: int

    # Discount and reward parameters
    gamma: Optional[float] = None
    rho: float = 0.5
    dt: float = 0.05

    # State handling
    clip_state: bool = True
    project_l2_ball: bool = False
    l2_radius: float = 0.1

    # Dynamics mode
    dynamics_mode: str = "b"  # "b" (PyTorch SDE) or "gym" (Gymnasium)

    # Evaluation type
    evaluation_type: str = "vectorized"  # "sequential" or "vectorized"
    deterministic: bool = True

    # Reward scaling
    x0_weight: Optional[float] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "EvaluationConfig":
        """Create from experiment config dict."""
        eval_cfg = cfg.get("eval", {})

        return cls(
            num_traj=int(eval_cfg.get("num_traj", 10)),
            traj_len=int(eval_cfg.get("traj_len", 100)),
            every_steps=int(eval_cfg.get("every_steps", 100)),
            gamma=eval_cfg.get("gamma"),
            rho=float(eval_cfg.get("rho", 0.5)),
            dt=float(eval_cfg.get("dt", 0.05)),
            clip_state=bool(eval_cfg.get("clip_state", True)),
            project_l2_ball=bool(eval_cfg.get("project_l2_ball", False)),
            l2_radius=float(eval_cfg.get("l2_radius", 0.1)),
            dynamics_mode=eval_cfg.get("dynamics_mode", "b"),
            evaluation_type=eval_cfg.get("evaluation_type", "vectorized"),
            deterministic=bool(eval_cfg.get("deterministic", True)),
            x0_weight=eval_cfg.get("x0_weight"),
        )

    def to_params_dict(self) -> Dict[str, Any]:
        """Convert to parameters dict for evaluation functions."""
        return {
            "gamma": self.gamma,
            "clip_state": self.clip_state,
            "project_l2_ball": self.project_l2_ball,
            "l2_radius": self.l2_radius,
            "dynamics_mode": self.dynamics_mode,
        }

    def get_sequential_params(self) -> Dict[str, Any]:
        """Get parameters for sequential evaluation (filtered subset)."""
        return {
            "gamma": self.gamma,
            "clip_state": self.clip_state,
            "dynamics_mode": self.dynamics_mode,
        }


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    # Core training parameters
    total_steps: int
    seed: int = 1337
    verbose: bool = False

    # Checkpointing
    checkpoint_every: Optional[int] = None
    results_root: str = "results"

    # Display
    print_interval: int = 100

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        """Create from experiment config dict."""
        save_cfg = cfg.get("save", {})

        return cls(
            total_steps=0,  # Will be set by algorithm
            seed=int(cfg.get("seed", 1337)),
            verbose=bool(cfg.get("verbose", False)),
            checkpoint_every=save_cfg.get("checkpoint_every"),
            results_root=cfg.get("results_root", "results"),
            print_interval=int(cfg.get("print_interval", 100)),
        )


@dataclass
class RewardScaleConfig:
    """Configuration for reward scaling."""

    base_scale: float = 1.0
    scale_by_dt: bool = True
    dt: float = 1.0

    @property
    def effective_scale(self) -> float:
        """Compute effective reward scale."""
        return self.base_scale * (self.dt if self.scale_by_dt else 1.0)

    @classmethod
    def from_algo_and_env(cls, algo, env) -> "RewardScaleConfig":
        """Create from algorithm and environment."""
        return cls(
            base_scale=float(getattr(algo, "reward_scale", 1.0)),
            scale_by_dt=bool(getattr(algo, "scale_reward_by_dt", True)),
            dt=float(getattr(env, "dt", 1.0)),
        )

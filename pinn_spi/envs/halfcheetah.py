from __future__ import annotations
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from .base import StochasticControlEnv, BoxSpec

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Warning: gymnasium not installed. HalfCheetah environment will not be available.")

Tensor = torch.Tensor


@dataclass
class HalfCheetahSpec:
    sigma: float
    exponent: float
    dt: float
    train_dynamics: bool = True
    train_dynamic_params: dict = None  # Dynamics training parameters

class HalfCheetahEnv(StochasticControlEnv):
    def __init__(self, spec: HalfCheetahSpec, device: torch.device, render_mode: Optional[str] = None):
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for HalfCheetah environment. Install with: pip install gymnasium")

        self.spec = spec
        self.device = device

        # === Gym 환경 초기화 ===
        self.gym_env = gym.make("HalfCheetah-v5", render_mode=render_mode)
        obs_space = self.gym_env.observation_space
        act_space = self.gym_env.action_space
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(act_space.shape))

        # === 상태/행동 BoxSpec ===
        obs_clip = 10.0
        obs_low = np.where(np.isfinite(obs_space.low), obs_space.low, -obs_clip).astype(np.float32)
        obs_high = np.where(np.isfinite(obs_space.high), obs_space.high, obs_clip).astype(np.float32)
        self._X = BoxSpec(
            low=torch.tensor(obs_low, device=device),
            high=torch.tensor(obs_high, device=device),
        )
        self._U = BoxSpec(
            low=torch.tensor(act_space.low, device=device, dtype=torch.float32),
            high=torch.tensor(act_space.high, device=device, dtype=torch.float32),
        )
        self._I = torch.eye(self.obs_dim, device=device, dtype=torch.float32)
        self.train_dynamics = spec.train_dynamics

        # === MLP dynamics model ===
        self.mlp_dynamics = self.get_dynamic_models(spec.train_dynamic_params, device)
        self.mlp_ready = True
        # nn.Sequential(
        #     nn.Linear(self.obs_dim + self.act_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.obs_dim)
        # ).to(device)

    def run_train_dynamics(self, dynamic_params: dict):
         # === 동역학 학습 제어 ===
        if self.train_dynamics:
            print("[HalfCheetahEnv] Training MLP dynamics model ...")
            self._train_mlp_dynamics(dynamic_params)
            
        else:
            raise AssertionError("train_dynamics=False: 아직 구현되지 않은 비학습 모드입니다.")
    def clip_state(self, x: Tensor) -> Tensor:
        """Clip state to valid bounds"""
        return torch.max(torch.min(x, self._X.high), self._X.low)

    def get_dynamic_models(self, params: dict, device: torch.device):
        """
        Create dynamics model based on specified parameters.

        Args:
            params: Dictionary with model parameters:
                - model_type: "MLP" or "LSTM"
                - hidden: List of hidden layer sizes (for MLP)
                - activation: Activation function ("ReLU", "SiLU", "Tanh")
            device: PyTorch device  
            """
        model_type = params.get("model_type", "MLP")
        if model_type == "MLP":
            hidden_sizes = params.get("hidden", [256, 256])
            activation_str = params.get("activation", "ReLU")
            if activation_str == "ReLU":
                activation = nn.ReLU
            elif activation_str == "SiLU":
                activation = nn.SiLU
            elif activation_str == "Tanh":
                activation = nn.Tanh
            else:
                raise ValueError(f"Unsupported activation: {activation_str}")

            layers = []
            input_size = self.obs_dim + self.act_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(input_size, h))
                layers.append(activation())
                input_size = h
            layers.append(nn.Linear(input_size, self.obs_dim))

            model = nn.Sequential(*layers).to(device)
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    # --------------------------------------------------
    # MuJoCo로 샘플 수집 및 MLP 학습
    # --------------------------------------------------
    def _train_mlp_dynamics(self, dynamic_params: dict):
        #  num_samples: int = 20000, batch_size: int = 256, lr: float = 1e-3
        start_time = time.time()
        env = self.gym_env
        dt = self.spec.dt
        print(dynamic_params)
        optimizer = torch.optim.Adam(self.mlp_dynamics.parameters(), lr=dynamic_params["lr"], weight_decay=dynamic_params.get("weight_decay", 0.0))
        loss_fn = nn.MSELoss()

        X_buf, U_buf, Y_buf = [], [], []

        obs, _ = env.reset()
        for _ in range(dynamic_params["num_samples"]):
            u = env.action_space.sample()
            obs_next, _, _, _, _ = env.step(u)
            dx = (obs_next - obs) / dt
            X_buf.append(obs)
            U_buf.append(u)
            Y_buf.append(dx)
            obs = obs_next

        X_buf = torch.tensor(np.array(X_buf), dtype=torch.float32, device=self.device)
        U_buf = torch.tensor(np.array(U_buf), dtype=torch.float32, device=self.device)
        Y_buf = torch.tensor(np.array(Y_buf), dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X_buf, U_buf, Y_buf)
        loader = torch.utils.data.DataLoader(dataset, batch_size=dynamic_params["batch_size"], shuffle=True)
        epochs = dynamic_params["epochs"]
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, ub, yb in loader:
                pred = self.mlp_dynamics(torch.cat([xb, ub], dim=1))
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss/len(loader):.6f}")

        print("[HalfCheetahEnv] MLP dynamics training complete.")
        self.train_dynamics_time = time.time() - start_time
        print(f"[HalfCheetahEnv] Training time: {self.train_dynamics_time:.2f} seconds")

        time_log_path = dynamic_params.get("time_log_path", "halfcheetah_dynamics_time.json")
        payload = {"seconds": self.train_dynamics_time}
        try:
            directory = os.path.dirname(time_log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(time_log_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
        except OSError as exc:
            print(f"[HalfCheetahEnv] Warning: Failed to write dynamics time log ({exc})")

    # --------------------------------------------------
    # Drift/dynamics: MLP 기반 예측
    # --------------------------------------------------
    def b(self, x: Tensor, u: Tensor) -> Tensor:
        if not getattr(self, "mlp_ready", False):
            raise AssertionError("Dynamics MLP not trained yet. train_dynamics=True로 재시도하십시오.")
        xu = torch.cat([x, u], dim=-1)
        return self.mlp_dynamics(xu)

    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        if not getattr(self, "mlp_ready", False):
            raise AssertionError("Dynamics MLP not trained yet. train_dynamics=True로 재시도하십시오.")
        return x + self.b(x, u) * self.spec.dt

    # --------------------------------------------------
    # 기타 기본 함수
    # --------------------------------------------------
    @property
    def d(self) -> int:
        return 17

    @property
    def m(self) -> int:
        return 6

    @property
    def dt(self) -> float:
        return self.spec.dt

    def r(self, x: Tensor, u: Tensor) -> Tensor:
        forward_vel = x[:, 8]
        ctrl_cost = 0.1 * (u ** 2).sum(dim=1)
        return forward_vel - ctrl_cost

    def sigma(self, x: Tensor, u: Tensor) -> Tensor:
        batch_size = x.shape[0]
        S = self.spec.sigma * self._I
        return S.unsqueeze(0).expand(batch_size, self.d, self.d)

    def reset(self):
        obs, _ = self.gym_env.reset()
        return obs

    def close(self):
        self.gym_env.close()

    def sync_state(self, x: Tensor) -> None:
        """
        Synchronize MuJoCo environment internal state with SDE state.

        After Euler-Maruyama integration computes x2 externally, this method
        updates the internal MuJoCo state (qpos, qvel) to match, ensuring
        consistency for rendering and potential gym API usage.

        Args:
            x: State tensor [B, 17] with HalfCheetah observation
               For SAC/single-env: [1, 17]
               For PPO/vectorized: [num_envs, 17]

        Note:
            HalfCheetah's observation is 17D, but MuJoCo state has 18D (qpos: 9D, qvel: 9D).
            The observation excludes the x-position (rootx) from qpos.
            We reconstruct the full state by: qpos = [rootx, obs[:8]], qvel = obs[8:17]
        """
        if x.shape[0] == 1:
            # Single environment (SAC, PINN-SPI evaluation)
            state_np = x.squeeze(0).cpu().numpy()

            # HalfCheetah state: qpos[9], qvel[9]
            # Observation: [qpos[1:], qvel] (17D, excludes rootx)
            # Reconstruct full state
            current_state = self.gym_env.unwrapped.state_vector()
            rootx = current_state[0]  # Preserve x-position

            qpos = np.concatenate([[rootx], state_np[:8]])  # [rootx, qpos[1:]]
            qvel = state_np[8:17]

            self.gym_env.unwrapped.set_state(qpos, qvel)
        else:
            # Vectorized environments (PPO)
            # Only sync the first environment (gym_env is single instance)
            state_np = x[0].cpu().numpy()

            current_state = self.gym_env.unwrapped.state_vector()
            rootx = current_state[0]

            qpos = np.concatenate([[rootx], state_np[:8]])
            qvel = state_np[8:17]

            self.gym_env.unwrapped.set_state(qpos, qvel)

    def get_action_sample_bounds(self):
        """
        Return true continuous 6D action bounds for HalfCheetah.
        Each action dimension lies in [-1, 1].
        """
        low = torch.tensor([-1.0] * self.act_dim, device=self.device, dtype=torch.float32)
        high = torch.tensor([1.0] * self.act_dim, device=self.device, dtype=torch.float32)
        return low, high
    
    def get_state_sample_bounds(self):
        """
        Return bounds for sampling states during PINN or model training.
        HalfCheetah observation space = 17D continuous Box(-inf, inf)
        → 무한대는 obs_clip(≈10)으로 이미 클리핑되어 있음.
        따라서 그대로 _X.low / _X.high 사용 가능.
        """
        return self._X.low, self._X.high
# --------------------------------------------------
# YAML 로더 (유지)
# --------------------------------------------------
def load_halfcheetah_from_yaml(cfg: dict, dt: float, device: torch.device,
                               render_mode: Optional[str] = None, verbose: bool = False) -> HalfCheetahEnv:
    if not GYMNASIUM_AVAILABLE:
        raise ImportError("gymnasium is required. Install with: pip install gymnasium")

    p = cfg["params"]
    train_dynamic_params = p.get("train_dynamic_params", {}) 
    if verbose:
        print(f"\n=== Loading HalfCheetah environment ===")
        print(f"Device: {device}")
        print(f"Time step (dt): {dt}")
        print(f"Render mode: {render_mode}")

    spec = HalfCheetahSpec(
        sigma=float(p["sigma"]),
        exponent=float(p["exponent"]),
        dt=dt,
        train_dynamics=bool(p.get("train_dynamics", True)),
        train_dynamic_params = train_dynamic_params
    )

    if verbose:
        print(f"\nEnvironment Parameters:")
        print(f"  Noise level (σ): {spec.sigma}")
        print(f"  Discount rate (ρ): {spec.exponent}")
        print(f"  Train dynamics: {spec.train_dynamics}")
        print(f"=== HalfCheetah environment loaded ===\n")


    return HalfCheetahEnv(spec, device, render_mode)
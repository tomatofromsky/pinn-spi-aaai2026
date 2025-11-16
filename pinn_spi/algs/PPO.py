"""
Proximal Policy Optimization (PPO)

Self-contained PPO implementation that conforms to the Algorithm protocol.
Supports both continuous (tanh-Gaussian) and discrete (Categorical) actions.

Expected training batch (on-policy):
    batch = {
        's':    [B, d] states,
        'a':    [B, m] (continuous) or [B, 1] (discrete) actions,
        'logp': [B, 1] old log-probabilities under behavior policy,
        'adv':  [B, 1] advantage estimates (e.g., GAE-Lambda),
        'ret':  [B, 1] discounted returns (for value function targets),
    }

Note: This file implements PPO core; wiring rollout/collector is left to the
experiment runner. To integrate end-to-end, extend the runner with an
on-policy data collection loop that computes (adv, ret, logp_old).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .base import Algorithm, Tensor
from ..agents.nets import MLP
from ..core.normalization import StateNormalization


# =====================
# Common utilities
# =====================

LOG_STD_MIN, LOG_STD_MAX = -20, 2


def atanh(y: Tensor, eps: float = 1e-6) -> Tensor:
    y = y.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(y) - torch.log1p(-y))


# =====================
# Policies and Value Nets
# =====================

class TanhGaussianActor(nn.Module):
    """Continuous policy: tanh-squashed Gaussian with configurable log-std mapping."""

    def __init__(
        self,
        d: int,
        m: int,
        hidden,
        *,
        log_std_bounds: Tuple[float, float] = (LOG_STD_MIN, LOG_STD_MAX),
        use_tanh_log_std: bool = False,
        normalize_inputs: bool = False,
        max_x=None,
    ):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            assert max_x is not None, "max_x must be provided when normalize_inputs=True"
            self.input_norm = StateNormalization(max_x)
        self.backbone = MLP(d, 2 * m, hidden)
        self.log_std_bounds = log_std_bounds
        self.use_tanh_log_std = use_tanh_log_std

    def _map_log_std(self, raw: Tensor) -> Tensor:
        low, high = self.log_std_bounds
        if self.use_tanh_log_std:
            # Map raw logits → [-1, 1] via tanh, then to [low, high]
            raw = torch.tanh(raw)
            return low + 0.5 * (raw + 1.0) * (high - low)
        return raw.clamp(low, high)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.normalize_inputs:
            x = self.input_norm(x)
        out = self.backbone(x)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = self._map_log_std(log_std)
        std = log_std.exp()
        return mu, std

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mu, std = self(x)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        u = torch.tanh(z)
        # Gaussian logprob
        logp = -0.5 * ((eps ** 2) + 2 * std.log() + math.log(2 * math.pi))
        logp = logp.sum(-1, keepdim=True)
        # Tanh correction
        logp -= torch.log(1 - torch.tanh(z).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return u, logp

    def log_prob(self, x: Tensor, u: Tensor) -> Tensor:
        # Invert tanh to latent z and compute corrected log-prob
        mu, std = self(x)
        z = atanh(u)
        log_unnorm = -0.5 * (((z - mu) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))
        log_unnorm = log_unnorm.sum(-1, keepdim=True)
        log_correction = torch.log(1 - u.pow(2) + 1e-6).sum(-1, keepdim=True)
        return log_unnorm - log_correction


class CatActor(nn.Module):
    """Discrete policy: Categorical over actions 0..K-1"""
    def __init__(self, d: int, num_actions: int, hidden, normalize_inputs: bool = False, max_x=None):
        super().__init__()
        self.num_actions = num_actions
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            assert max_x is not None, "max_x must be provided when normalize_inputs=True"
            self.input_norm = StateNormalization(max_x)
        self.net = MLP(d, num_actions, hidden)

    def logits(self, x: Tensor) -> Tensor:
        if self.normalize_inputs:
            x = self.input_norm(x)
        return self.net(x)

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.logits(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample().unsqueeze(-1)  # [B,1]
        logp = dist.log_prob(a.squeeze(-1)).unsqueeze(-1)
        return a.float(), logp

    def log_prob(self, x: Tensor, a: Tensor) -> Tensor:
        logits = self.logits(x)
        dist = torch.distributions.Categorical(logits=logits)
        a_long = a.long().squeeze(-1)
        return dist.log_prob(a_long).unsqueeze(-1)


class ValueNet(nn.Module):
    def __init__(self, d: int, hidden, normalize_inputs: bool = False, max_x=None):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            assert max_x is not None, "max_x must be provided when normalize_inputs=True"
            self.input_norm = StateNormalization(max_x)
        self.net = MLP(d, 1, hidden)

    def forward(self, x: Tensor) -> Tensor:
        if self.normalize_inputs:
            x = self.input_norm(x)
        return self.net(x)


# =====================
# PPO Algorithm
# =====================

class PPO(Algorithm):
    """
    Proximal Policy Optimization

    - Supports continuous and discrete action spaces
    - Clipped surrogate objective
    - Value function loss with optional clipping
    - Entropy bonus for exploration

    Expects on-policy batches providing old log-probs, advantages, and returns.
    """

    def __init__(self,
                 d: int,
                 m: int,
                 cfg: dict,
                 expt: dict,
                 device: torch.device,
                 env):
        self.d = d
        self.m = m
        self.env = env
        self.device = device
        self.is_discrete = env.is_discrete_action

        hidden = cfg.get("hidden", [64, 64])
        log_std_bounds = tuple(cfg.get("log_std_bounds", [LOG_STD_MIN, LOG_STD_MAX]))
        use_tanh_log_std = bool(cfg.get("tanh_log_std", False))

        # Input normalization (default: False)
        normalize_inputs = cfg.get("normalize_inputs", False)
        max_x = None
        if normalize_inputs:
            # Compute max_x from environment bounds
            state_low, state_high = env.get_state_sample_bounds()
            max_x = torch.maximum(
                torch.abs(torch.as_tensor(state_low, dtype=torch.float32)),
                torch.abs(torch.as_tensor(state_high, dtype=torch.float32))
            )

        if self.is_discrete:
            self.actor = CatActor(d, env.num_discrete_actions, hidden,
                                 normalize_inputs=normalize_inputs, max_x=max_x).to(device)
        else:
            self.actor = TanhGaussianActor(
                d,
                m,
                hidden,
                log_std_bounds=log_std_bounds,
                use_tanh_log_std=use_tanh_log_std,
                normalize_inputs=normalize_inputs,
                max_x=max_x,
            ).to(device)
            # Per-dimension action scaling from environment bounds
            # Maps normalized actions [-1,1] to environment action space via affine transform
            action_low, action_high = env.get_action_sample_bounds()
            self.act_low = torch.as_tensor(action_low, device=device, dtype=torch.float32)
            self.act_high = torch.as_tensor(action_high, device=device, dtype=torch.float32)
            self.act_scale = 0.5 * (self.act_high - self.act_low)  # per-dim scale
            self.act_bias = 0.5 * (self.act_high + self.act_low)   # per-dim center

        self.critic = ValueNet(d, cfg.get("v_hidden", hidden),
                              normalize_inputs=normalize_inputs, max_x=max_x).to(device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("lr_actor", 3e-4)))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.get("lr_critic", 3e-4)))

        # PPO hyperparameters
        self.clip_ratio = float(cfg.get("clip_ratio", 0.2))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.0))
        self.vf_coef = float(cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.train_iters = int(cfg.get("train_iters", 80))
        self.minibatch_size = int(cfg.get("minibatch_size", 256))
        # Value loss options for stability
        self.use_value_clip = cfg.get("use_value_clip", True)  # Clip value updates like policy
        self.value_loss_type = cfg.get("value_loss_type", "mse")  # "mse" or "huber"
        # On-policy rollout configuration
        self.rollout_steps = int(cfg.get("rollout_steps", 2048))
        self.num_envs = int(cfg.get("num_envs", 16))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))

        # Training horizon (steps) used by runner if integrated
        total_updates = int(cfg.get("total_updates", 0))
        if total_updates > 0:
            self.total_steps = total_updates * self.rollout_steps * self.num_envs
        else:
            self.total_steps = int(cfg.get("total_steps", 100000))

        # Discounting from experiment config
        self.gamma = float(expt["eval"].get("gamma", 0.99))

        # Reward scaling by dt (default: True for continuous-time interpretation)
        # When True: r_scaled = r * dt (continuous-time interpretation, matches other implementations)
        # When False: r_scaled = r (discrete-time interpretation)
        self.scale_reward_by_dt = cfg.get("scale_reward_by_dt", True)
        self.reward_scale = float(cfg.get("reward_scale", 1.0))

    # ===== protocol methods =====
    def to(self, device: torch.device) -> "PPO":
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        # Move action scaling tensors to device (continuous only)
        if not self.is_discrete:
            self.act_low = self.act_low.to(device)
            self.act_high = self.act_high.to(device)
            self.act_scale = self.act_scale.to(device)
            self.act_bias = self.act_bias.to(device)
        return self

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def act(self, x: Tensor, deterministic: bool = False) -> Tensor:
        if self.is_discrete:
            if deterministic:
                logits = self.actor.logits(x)
                a = logits.argmax(dim=-1, keepdim=True).float()
                return a
            a, _ = self.actor.sample(x)
            return a
        else:
            if deterministic:
                mu, _ = self.actor(x)
                a_norm = torch.tanh(mu)
            else:
                a_norm, _ = self.actor.sample(x)
            # Apply per-dimension affine transform: a_env = bias + scale * a_norm
            return self.act_bias + self.act_scale * a_norm

    @torch.no_grad()
    def sample_action_and_logp(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample action and return (action_env_space, logp) under current policy."""
        if self.is_discrete:
            logits = self.actor.logits(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().unsqueeze(-1).float()
            logp = dist.log_prob(a.long().squeeze(-1)).unsqueeze(-1)
            return a, logp
        else:
            a_norm, logp = self.actor.sample(x)
            # Apply per-dimension affine transform: a_env = bias + scale * a_norm
            a_env = self.act_bias + self.act_scale * a_norm
            return a_env, logp

    @torch.no_grad()
    def value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Run multiple epochs of PPO updates over provided on-policy batch.

        Batch keys required:
            s [N, d], a [N, m or 1], logp [N,1], adv [N,1], ret [N,1]
        Batch keys optional:
            v_old [N,1]: old value predictions (for value clipping, if use_value_clip=True)
        """
        s = batch["s"].to(self.device)
        a = batch["a"].to(self.device)
        logp_old = batch["logp"].to(self.device)
        adv = batch["adv"].to(self.device)
        ret = batch["ret"].to(self.device)

        # Normalize advantages for stability
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = s.shape[0]
        idx = torch.arange(N, device=self.device)

        pi_loss_epoch = 0.0
        vf_loss_epoch = 0.0
        ent_epoch = 0.0

        for epoch in range(self.train_iters):
            perm = idx[torch.randperm(N)]
            epoch_pi_loss = 0.0
            epoch_vf_loss = 0.0
            epoch_ent = 0.0
            num_batches = 0

            for start in range(0, N, self.minibatch_size):
                mb_idx = perm[start:start + self.minibatch_size]
                s_mb, a_mb = s[mb_idx], a[mb_idx]
                adv_mb, ret_mb = adv[mb_idx], ret[mb_idx]
                logp_old_mb = logp_old[mb_idx]

                # New log-prob
                if self.is_discrete:
                    logp = self.actor.log_prob(s_mb, a_mb)
                    # Entropy of categorical
                    logits = self.actor.logits(s_mb)
                    dist = torch.distributions.Categorical(logits=logits)
                    ent = dist.entropy().mean()
                else:
                    # Convert env-space action to normalized space for log-prob
                    # Invert affine transform: a_norm = (a_env - bias) / scale
                    a_norm = (a_mb - self.act_bias) / (self.act_scale + 1e-8)
                    logp = self.actor.log_prob(s_mb, a_norm)
                    # Approx entropy via sampling one reparam sample
                    with torch.no_grad():
                        _, lp_samp = self.actor.sample(s_mb)
                    ent = -lp_samp.mean()

                ratio = torch.exp(logp - logp_old_mb)  # [B,1]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb
                pi_loss = -(torch.min(surr1, surr2)).mean() - self.entropy_coef * ent

                # Value loss with optional clipping and loss type
                v = self.critic(s_mb)
                if self.use_value_clip and "v_old" in batch:
                    # Clipped value loss (standard PPO2)
                    v_old = batch["v_old"][mb_idx].to(self.device)
                    v_clipped = v_old + (v - v_old).clamp(-self.clip_ratio, self.clip_ratio)
                    if self.value_loss_type == "huber":
                        vf_loss_unclipped = F.smooth_l1_loss(v, ret_mb)
                        vf_loss_clipped = F.smooth_l1_loss(v_clipped, ret_mb)
                    else:  # mse
                        vf_loss_unclipped = F.mse_loss(v, ret_mb)
                        vf_loss_clipped = F.mse_loss(v_clipped, ret_mb)
                    vf_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)
                else:
                    # Unclipped value loss
                    if self.value_loss_type == "huber":
                        vf_loss = F.smooth_l1_loss(v, ret_mb)
                    else:  # mse
                        vf_loss = F.mse_loss(v, ret_mb)

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                (pi_loss + self.vf_coef * vf_loss).backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

                pi_loss_epoch += float(pi_loss.item())
                vf_loss_epoch += float(vf_loss.item())
                ent_epoch += float(ent.item())

                epoch_pi_loss += float(pi_loss.item())
                epoch_vf_loss += float(vf_loss.item())
                epoch_ent += float(ent.item())
                num_batches += 1

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_pi = epoch_pi_loss / num_batches
                avg_vf = epoch_vf_loss / num_batches
                avg_ent = epoch_ent / num_batches
                print(f"  PPO Epoch {epoch + 1}/{self.train_iters}: "
                      f"pi_loss={avg_pi:.4f}, v_loss={avg_vf:.4f}, entropy={avg_ent:.4f}")

        iters = max(1, (N // self.minibatch_size) * self.train_iters)
        return {
            "pi_loss": pi_loss_epoch / iters,
            "v_loss": vf_loss_epoch / iters,
            "entropy": ent_epoch / iters,
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str, map_location="cpu") -> None:
        sd = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    def get_training_steps(self, exp_cfg: dict) -> int:
        return self.total_steps

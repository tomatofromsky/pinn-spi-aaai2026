"""SAC: Soft Actor-Critic with entropy regularization for continuous and discrete actions."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from ..algs.base import Algorithm, Tensor
from ..agents.nets import MLP
from ..core.normalization import StateNormalization, StateActionNormalization

# Constants for log standard deviation bounds
LOG_STD_MIN, LOG_STD_MAX = -20, 2


def atanh(y, eps=1e-6):
    """Safe inverse tanh with boundary clamping."""
    y = y.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(y) - torch.log1p(-y))

class TanhGaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy. Supports "shared" and "separate" architectures."""
    def __init__(self, d, m, hidden, log_std_mode="clamp", log_std_min=-20, log_std_max=2,
                 architecture="shared", action_low=None, action_high=None,
                 normalize_inputs=False, max_x=None):
        super().__init__()
        self.m = m
        self.log_std_mode = log_std_mode  # "clamp" or "tanh"
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.architecture = architecture
        self._act_log_scale = None

        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None:
                raise ValueError("max_x must be provided when normalize_inputs=True")
            self.state_norm = StateNormalization(max_x)
        else:
            self.state_norm = None

        if architecture == "separate":
            self.backbone = MLP(d, hidden[-1], hidden[:-1]) if len(hidden) > 1 else nn.Identity()
            self.fc_mean = nn.Linear(hidden[-1] if len(hidden) > 1 else d, m)
            self.fc_logstd = nn.Linear(hidden[-1] if len(hidden) > 1 else d, m)

            if action_low is not None and action_high is not None:
                self.register_buffer(
                    "action_scale",
                    torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
                )
                self.register_buffer(
                    "action_bias",
                    torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
                )
            else:
                self.register_buffer("action_scale", torch.ones(m, dtype=torch.float32))
                self.register_buffer("action_bias", torch.zeros(m, dtype=torch.float32))
        else:
            self.backbone = MLP(d, 2 * m, hidden)
            self.action_scale = None
            self.action_bias = None

    def forward(self, x):
        if self.normalize_inputs:
            x = self.state_norm(x)

        if self.architecture == "separate":
            features = self.backbone(x) if not isinstance(self.backbone, nn.Identity) else x
            mu = self.fc_mean(features)
            log_std = self.fc_logstd(features)
        else:
            out = self.backbone(x)
            mu, log_std = torch.chunk(out, 2, dim=-1)

        if self.log_std_mode == "tanh":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        else:
            log_std = log_std.clamp(self.log_std_min, self.log_std_max)

        std = log_std.exp()
        return mu, std

    def sample(self, x):
        """
        Sample action with log-probability.

        Returns:
            - If architecture == "separate": (action_scaled, log_prob, mean_scaled)
              where actions are in environment space
            - If architecture == "shared": (u_normalized, log_prob)
              where u is in [-1, 1] normalized space
        """
        mu, std = self(x)
        eps = torch.randn_like(mu)
        z = mu + std * eps  # Pre-tanh latent variable
        u = torch.tanh(z)   # Normalized action in [-1, 1]

        # Gaussian log-probability (before tanh)
        logp = (-0.5 * (eps.pow(2) + 2 * std.log() +
                torch.log(torch.tensor(2 * torch.pi, device=x.device))))
        logp = logp.sum(-1, keepdim=True)

        if self.architecture == "separate":
            # CleanRL-style: include action_scale in Jacobian correction
            # log_prob correction: -log|det(da/du)| where a = tanh(u) * scale + bias
            # = -sum log(scale * (1 - tanh^2(u)))
            logp -= torch.log(self.action_scale * (1 - u.pow(2)) + 1e-6).sum(1, keepdim=True)

            # Scale actions to environment space
            action = u * self.action_scale + self.action_bias
            mean_scaled = torch.tanh(mu) * self.action_scale + self.action_bias

            return action, logp, mean_scaled
        else:
            # Original: tanh correction only (action scaling handled by SAC)
            logp -= torch.log(1 - u.pow(2) + 1e-6).sum(-1, keepdim=True)

            # Optional action scaling Jacobian correction (from our improvements)
            # When actions are scaled by act_scale externally, add log-det term
            if self._act_log_scale is not None:
                logp -= self._act_log_scale.sum()

            return u, logp

    def deterministic(self, x):
        """
        Return deterministic action (mean).

        Returns:
            - If architecture == "separate": mean action in environment space
            - If architecture == "shared": mean action in [-1, 1] normalized space
        """
        mu, _ = self(x)
        u_mean = torch.tanh(mu)

        if self.architecture == "separate":
            # Return scaled action
            return u_mean * self.action_scale + self.action_bias
        else:
            # Return normalized action
            return u_mean

class QNet(nn.Module):
    """Q-function network for continuous actions"""
    def __init__(self, d, m, hidden, normalize_inputs=False, max_x=None, max_u=None):
        super().__init__()
        # Optional input normalization: map raw domain to [-1, 1]
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None or max_u is None:
                raise ValueError("max_x and max_u must be provided when normalize_inputs=True")
            self.state_action_norm = StateActionNormalization(max_x, max_u)
        else:
            self.state_action_norm = None

        self.net = MLP(d + m, 1, hidden)

    def forward(self, x, u):
        # Normalize inputs if enabled
        if self.normalize_inputs:
            xu = self.state_action_norm(x, u)
        else:
            xu = torch.cat([x, u], dim=-1)
        return self.net(xu)


class DiscreteSACPolicy(nn.Module):
    """Discrete policy network with Categorical distribution for SAC"""
    def __init__(self, d, num_actions, hidden, normalize_inputs=False, max_x=None):
        super().__init__()
        self.num_actions = num_actions

        # Optional input normalization: map raw domain to [-1, 1]
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None:
                raise ValueError("max_x must be provided when normalize_inputs=True")
            self.state_norm = StateNormalization(max_x)
        else:
            self.state_norm = None

        self.net = MLP(d, num_actions, hidden)

    def forward(self, x):
        """Returns logits for Categorical distribution"""
        # Normalize inputs if enabled
        if self.normalize_inputs:
            x = self.state_norm(x)
        return self.net(x)

    def sample(self, x):
        """Sample action with log probability for entropy term"""
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # [B]
        log_prob = dist.log_prob(action)  # [B]
        return action.unsqueeze(-1), log_prob.unsqueeze(-1)  # [B, 1], [B, 1]

    def deterministic(self, x):
        """Deterministic action (argmax)"""
        logits = self.forward(x)
        return logits.argmax(dim=-1, keepdim=True)  # [B, 1]


class DiscreteQNet(nn.Module):
    """Q-function network for discrete actions - outputs Q(s, a) for all actions"""
    def __init__(self, d, num_actions, hidden, normalize_inputs=False, max_x=None):
        super().__init__()
        self.num_actions = num_actions

        # Optional input normalization: map raw domain to [-1, 1]
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None:
                raise ValueError("max_x must be provided when normalize_inputs=True")
            self.state_norm = StateNormalization(max_x)
        else:
            self.state_norm = None

        self.net = MLP(d, num_actions, hidden)

    def forward(self, x, u=None):
        """
        Compute Q-values.

        Args:
            x: States [B, d]
            u: Actions [B, 1] (optional - if provided, returns Q(x, u))

        Returns:
            If u is None: Q-values for all actions [B, num_actions]
            If u is provided: Q-values for specific actions [B, 1]
        """
        # Normalize inputs if enabled
        if self.normalize_inputs:
            x = self.state_norm(x)
        q_all = self.net(x)  # [B, num_actions]

        if u is None:
            return q_all
        else:
            # Gather Q-values for specific actions
            u = u.long().squeeze(-1)  # [B]
            q_values = q_all.gather(1, u.unsqueeze(-1))  # [B, 1]
            return q_values


class SAC(Algorithm):
    """Soft Actor-Critic implementation for continuous and discrete actions with optional twin critics"""
    def __init__(self, d, m, action_scale: float, cfg: dict, expt, device: torch.device, env):
        self.d = d  # state dimension
        self.m = m  # action dimension
        self.env = env
        self.is_discrete = env.is_discrete_action

        # Twin critics (double Q-learning) - reduces overestimation bias
        self.use_twin_critics = cfg.get("use_twin_critics", True)  # Default: ON (modern SAC)

        # SAC improvements from practical checklist (all optional for backward compatibility)
        # 1. Log-std mapping mode: "clamp" (hard, original) or "tanh" (soft, recommended)
        self.log_std_mode = cfg.get("log_std_mode", "clamp")  # Default: "clamp" (backward compatible)
        self.log_std_min = float(cfg.get("log_std_min", -20))  # Default: -20 (original)
        self.log_std_max = float(cfg.get("log_std_max", 2))    # Default: 2 (original)
        # 2. Action scaling Jacobian correction (for entropy regularization accuracy)
        self.use_action_jacobian = cfg.get("use_action_jacobian", False)  # Default: OFF (backward compatible)
        # 3. Delayed policy updates (TD3-style, for stability)
        self.policy_freq = int(cfg.get("policy_freq", 1))  # Default: 1 (update every step, backward compatible)
        self._policy_step = 0  # Counter for delayed policy updates
        # 4. Policy architecture: "shared" (original) or "separate" (CleanRL-style)
        self.policy_arch = cfg.get("policy_arch", "shared")  # Default: "shared" (backward compatible)

        # Get normalization parameters
        normalize_inputs = cfg.get("normalize_inputs", False)
        max_x = None
        max_u = None
        if normalize_inputs:
            # Get max_x from environment bounds
            state_low, state_high = env.get_state_sample_bounds()
            # Assume symmetric bounds or use the maximum absolute value
            max_x = torch.maximum(state_high.abs(), state_low.abs())

            # Get max_u from environment action bounds
            action_low, action_high = env.get_action_sample_bounds()
            max_u = torch.maximum(action_high.abs(), action_low.abs())

        # Create policy and Q-networks based on action space type
        if self.is_discrete:
            # Discrete action networks
            num_actions = env.num_discrete_actions
            self.policy = DiscreteSACPolicy(d, num_actions, cfg["hidden"],
                                          normalize_inputs=normalize_inputs, max_x=max_x).to(device)
            self.q1 = DiscreteQNet(d, num_actions, cfg["hidden"],
                                 normalize_inputs=normalize_inputs, max_x=max_x).to(device)
            self.q1_targ = DiscreteQNet(d, num_actions, cfg["hidden"],
                                       normalize_inputs=normalize_inputs, max_x=max_x).to(device)
            self.q1_targ.load_state_dict(self.q1.state_dict())

            if self.use_twin_critics:
                self.q2 = DiscreteQNet(d, num_actions, cfg["hidden"],
                                     normalize_inputs=normalize_inputs, max_x=max_x).to(device)
                self.q2_targ = DiscreteQNet(d, num_actions, cfg["hidden"],
                                           normalize_inputs=normalize_inputs, max_x=max_x).to(device)
                self.q2_targ.load_state_dict(self.q2.state_dict())
            else:
                self.q2 = None
                self.q2_targ = None
        else:
            # Continuous action networks
            # Get environment action bounds if using "separate" architecture
            action_low, action_high = None, None
            if self.policy_arch == "separate":
                action_low, action_high = env.get_action_sample_bounds()

            self.policy = TanhGaussianPolicy(
                d, m, cfg["hidden"],
                log_std_mode=self.log_std_mode,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
                architecture=self.policy_arch,
                action_low=action_low,
                action_high=action_high,
                normalize_inputs=normalize_inputs,
                max_x=max_x
            ).to(device)

            # Set action scaling Jacobian if enabled (only for "shared" architecture)
            # Note: "separate" architecture handles action scaling internally
            if self.use_action_jacobian and self.policy_arch == "shared":
                # Store log scale for Jacobian correction in log-prob
                act_scale_tensor = torch.tensor(action_scale, device=device, dtype=torch.float32)
                self.policy._act_log_scale = act_scale_tensor.log().unsqueeze(0).expand(1, m)

            self.q1 = QNet(d, m, cfg["hidden"], normalize_inputs=normalize_inputs,
                         max_x=max_x, max_u=max_u).to(device)
            self.q1_targ = QNet(d, m, cfg["hidden"], normalize_inputs=normalize_inputs,
                              max_x=max_x, max_u=max_u).to(device)
            self.q1_targ.load_state_dict(self.q1.state_dict())

            if self.use_twin_critics:
                self.q2 = QNet(d, m, cfg["hidden"], normalize_inputs=normalize_inputs,
                             max_x=max_x, max_u=max_u).to(device)
                self.q2_targ = QNet(d, m, cfg["hidden"], normalize_inputs=normalize_inputs,
                                  max_x=max_x, max_u=max_u).to(device)
                self.q2_targ.load_state_dict(self.q2.state_dict())
            else:
                self.q2 = None
                self.q2_targ = None

        # Auto-entropy tuning (learnable temperature)
        self.use_auto_entropy = cfg.get("use_auto_entropy", False)  # Default: off for backward compatibility
        if self.use_auto_entropy:
            if self.is_discrete:
                # Target entropy for discrete: -|A|
                self.target_entropy = -float(env.num_discrete_actions)
            else:
                # Target entropy for continuous: -dim(A)
                self.target_entropy = -float(m)
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(cfg.get("alpha_lr", 3e-4)))
        else:
            self.log_alpha = None
            self.target_entropy = None
            self.alpha_optimizer = None
            self.alpha = float(cfg["alpha"])

        self.gamma = float(expt["eval"]["gamma"])
        self.tau = float(cfg["tau"])
        self.act_scale = action_scale

        # Training schedule
        self.total_steps = int(cfg.get("total_steps", 50000))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.start_steps = int(cfg.get("start_steps", 1000))
        self.update_after = int(cfg.get("update_after", 1000))
        self.update_every = int(cfg.get("update_every", 1))

        # Gradient clipping (None = no clipping)
        self.max_grad_norm = cfg.get("max_grad_norm", None)  # e.g., 1.0 or 5.0

        # Reward scaling by dt (default: True for continuous-time interpretation)
        # When True: r_scaled = r * dt (continuous-time interpretation, matches other implementations)
        # When False: r_scaled = r (discrete-time interpretation)
        self.scale_reward_by_dt = cfg.get("scale_reward_by_dt", True)

        # Epsilon-greedy exploration parameters
        self.use_epsilon_greedy = cfg.get("use_epsilon_greedy", False)
        self.epsilon_start = cfg.get("epsilon_start", 1.0)
        self.epsilon_end = cfg.get("epsilon_end", 0.01)
        self.epsilon_decay_steps = cfg.get("epsilon_decay_steps", 10000)
        self.current_step = 0

        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=float(cfg["lr_actor"]))

        # Optimizer for critic(s)
        if self.use_twin_critics:
            q_params = list(self.q1.parameters()) + list(self.q2.parameters())
            self.opt_q = torch.optim.Adam(q_params, lr=float(cfg["lr_critic"]))
        else:
            self.opt_q = torch.optim.Adam(self.q1.parameters(), lr=float(cfg["lr_critic"]))

        self.device = device

    def to(self, device):
        self.device = device
        self.policy.to(device)
        self.q1.to(device)
        self.q1_targ.to(device)
        if self.use_twin_critics:
            self.q2.to(device)
            self.q2_targ.to(device)
        return self

    def train_mode(self):
        self.policy.train()
        self.q1.train()
        if self.use_twin_critics:
            self.q2.train()

    def eval_mode(self):
        self.policy.eval()
        self.q1.eval()
        if self.use_twin_critics:
            self.q2.eval()

    @torch.no_grad()
    def act(self, x: Tensor, deterministic: bool = False) -> Tensor:
        # Epsilon-greedy exploration (only during training, not evaluation)
        if self.use_epsilon_greedy and not deterministic and self.policy.training:
            epsilon = self._get_epsilon()
            if torch.rand(1).item() < epsilon:
                # Random action
                batch_size = x.shape[0]
                if self.is_discrete:
                    # Random discrete action
                    u = torch.randint(0, self.env.num_discrete_actions, (batch_size, 1), device=self.device).float()
                else:
                    # Random continuous action
                    u = 2 * torch.rand(batch_size, self.m, device=self.device) - 1  # [-1, 1]
                    u = self.act_scale * u
                return u

        # Normal SAC policy
        if deterministic:
            u = self.policy.deterministic(x)
            # For "shared" architecture, scale action
            if not self.is_discrete and self.policy_arch == "shared":
                u = self.act_scale * u
        else:
            sample_output = self.policy.sample(x)
            if self.policy_arch == "separate":
                # CleanRL-style: sample returns (action, logp, mean)
                u = sample_output[0]  # Already scaled to env space
            else:
                # Original: sample returns (action, logp)
                u = sample_output[0]
                # Scale action for continuous
                if not self.is_discrete:
                    u = self.act_scale * u

        return u

    def _get_epsilon(self) -> float:
        """Calculate current epsilon value with linear decay"""
        if self.current_step >= self.epsilon_decay_steps:
            return self.epsilon_end

        decay_ratio = self.current_step / self.epsilon_decay_steps
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_ratio
        return epsilon

    def step(self):
        """Increment step counter for epsilon decay"""
        if self.use_epsilon_greedy:
            self.current_step += 1

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        s, a, r, s2 = batch["s"], batch["a"], batch["r"], batch["s2"]
        done = batch.get("done", torch.zeros_like(r))  # Get done flag if available

        if self.is_discrete:
            return self._update_discrete(s, a, r, s2, done)
        else:
            return self._update_continuous(s, a, r, s2, done)

    def _update_continuous(self, s, a, r, s2, done=None):
        """
        Update for continuous actions with optional auto-entropy tuning and done masking.

        Supports both fixed and learnable temperature (alpha).

        IMPORTANT: Environments return costs (r ≤ 0, closer to 0 = better).
        SAC maximizes cumulative return, naturally maximizing from negative values toward 0.
        No reward negation is needed.
        """
        # Get current alpha (learned or fixed)
        alpha = self.log_alpha.exp() if self.use_auto_entropy else self.alpha

        # =====================================================
        # 1) Critic update
        # =====================================================
        with torch.no_grad():
            sample_output = self.policy.sample(s2)
            if self.policy_arch == "separate":
                # CleanRL-style: (action, logp, mean)
                a2, logp2, _ = sample_output
                # Actions already in env space, no scaling needed
            else:
                # Original: (action, logp)
                a2, logp2 = sample_output
                # Scale actions for Q-network
                a2 = a2 * self.act_scale

            # Compute target Q-values
            q1_t = self.q1_targ(s2, a2)
            if self.use_twin_critics:
                q2_t = self.q2_targ(s2, a2)
                q_t = torch.min(q1_t, q2_t)  # Double Q-learning: min(Q1, Q2)
            else:
                q_t = q1_t

            # Ensure shapes [batch_size, 1]
            if r.dim() == 1:
                r = r.unsqueeze(1)

            # Handle done mask (default to no masking if not provided for backward compatibility)
            if done is None:
                done = torch.zeros_like(r)
            if done.dim() == 1:
                done = done.unsqueeze(1)

            # Environments return costs r ≤ 0 (closer to 0 = better)
            # SAC maximizes cumulative return, which correctly maximizes toward 0 from negative values
            # No negation needed - SAC naturally handles cost-based rewards
            # Apply done mask to prevent bootstrapping from terminal states
            mask = 1.0 - done
            y = r + self.gamma * mask * (q_t - alpha * logp2)

        # Compute Q-losses
        q1_val = self.q1(s, a)
        q1_loss = F.mse_loss(q1_val, y)

        if self.use_twin_critics:
            q2_val = self.q2(s, a)
            q2_loss = F.mse_loss(q2_val, y)
            q_loss = q1_loss + q2_loss
        else:
            q_loss = q1_loss

        self.opt_q.zero_grad()
        q_loss.backward()
        if self.max_grad_norm is not None:
            if self.use_twin_critics:
                torch.nn.utils.clip_grad_norm_(
                    list(self.q1.parameters()) + list(self.q2.parameters()), self.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.opt_q.step()

        # =====================================================
        # 2) Actor update (with optional delayed updates)
        # =====================================================
        self._policy_step += 1
        do_policy_update = (self._policy_step % self.policy_freq) == 0

        pi_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)

        if do_policy_update:
            sample_output = self.policy.sample(s)
            if self.policy_arch == "separate":
                # CleanRL-style: (action, logp, mean)
                a_hat, logp, _ = sample_output
                # Actions already in env space
            else:
                # Original: (action, logp)
                a_hat, logp = sample_output
                # Scale actions for Q-network
                a_hat = a_hat * self.act_scale

            q1_pi = self.q1(s, a_hat)

            if self.use_twin_critics:
                q2_pi = self.q2(s, a_hat)
                q_val = torch.min(q1_pi, q2_pi)  # Use min(Q1, Q2) for policy improvement
            else:
                q_val = q1_pi

            pi_loss = (alpha * logp - q_val).mean()

            self.opt_pi.zero_grad()
            pi_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.opt_pi.step()

            # =====================================================
            # 3) Temperature (alpha) update (if enabled, update with policy)
            # =====================================================
            if self.use_auto_entropy:
                with torch.no_grad():
                    # Use the log_prob from policy sample (already computed)
                    # For continuous: entropy ≈ -log_prob (negative of log probability)
                    # Note: logp is already negative, so entropy = -logp
                    entropy = -logp

                # Alpha loss: minimize log_alpha * (entropy - target_entropy)
                alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # =====================================================
        # 4) Soft-update target critic(s)
        # =====================================================
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

            if self.use_twin_critics:
                for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "alpha_loss": float(alpha_loss.item()) if self.use_auto_entropy else 0.0,
            "alpha": float(alpha.item()) if self.use_auto_entropy else self.alpha
        }

    def _update_discrete(self, s, a, r, s2, done):
        """
        Update for discrete actions with optional auto-entropy tuning and done mask.

        Matches original experiment when use_auto_entropy=True.

        IMPORTANT: Environments return costs (r ≤ 0, closer to 0 = better).
        SAC maximizes cumulative return, naturally maximizing from negative values toward 0.
        No reward negation is needed.
        """
        # Get current alpha (learned or fixed)
        alpha = self.log_alpha.exp() if self.use_auto_entropy else self.alpha

        # =====================================================
        # 1) Critic update
        # =====================================================
        with torch.no_grad():
            # Get policy distribution at next state
            logits_next = self.policy(s2)  # [B, num_actions]
            log_probs_next = F.log_softmax(logits_next, dim=-1)  # [B, num_actions]
            probs_next = log_probs_next.exp()  # [B, num_actions]

            # Get target Q-values
            q1_all_next = self.q1_targ(s2)  # [B, num_actions]
            if self.use_twin_critics:
                q2_all_next = self.q2_targ(s2)  # [B, num_actions]
                q_all_next = torch.min(q1_all_next, q2_all_next)  # Double Q-learning: min(Q1, Q2)
            else:
                q_all_next = q1_all_next

            # Soft value function: V(s') = Σ_a π(a|s')[Q(s',a) - α*log π(a|s')]
            v_next = (probs_next * (q_all_next - alpha * log_probs_next)).sum(-1, keepdim=True)  # [B, 1]

            # Ensure shapes
            if r.dim() == 1:
                r = r.unsqueeze(1)
            if done.dim() == 1:
                done = done.unsqueeze(1)

            # Environments return costs r ≤ 0 (closer to 0 = better)
            # SAC maximizes cumulative return, naturally maximizing toward 0 from negative values
            # No negation needed - SAC naturally handles cost-based rewards
            # Target with done mask: y = r + γ * (1 - done) * V(s')
            y = r + self.gamma * (1 - done) * v_next  # [B, 1]

        # Current Q-values: Q(s, a)
        q1_current = self.q1(s, a)  # [B, 1]
        q1_loss = F.mse_loss(q1_current, y)

        if self.use_twin_critics:
            q2_current = self.q2(s, a)  # [B, 1]
            q2_loss = F.mse_loss(q2_current, y)
            q_loss = q1_loss + q2_loss
        else:
            q_loss = q1_loss

        self.opt_q.zero_grad()
        q_loss.backward()
        if self.max_grad_norm is not None:
            if self.use_twin_critics:
                torch.nn.utils.clip_grad_norm_(
                    list(self.q1.parameters()) + list(self.q2.parameters()), self.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.opt_q.step()

        # =====================================================
        # 2) Actor update (with optional delayed updates, fresh forward pass)
        # =====================================================
        self._policy_step += 1
        do_policy_update = (self._policy_step % self.policy_freq) == 0

        pi_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)

        if do_policy_update:
            logits = self.policy(s)  # [B, num_actions]
            log_probs = F.log_softmax(logits, dim=-1)  # [B, num_actions]
            probs = log_probs.exp()  # [B, num_actions]

            # Get Q-values (detached to avoid training critic)
            with torch.no_grad():
                q1_all = self.q1(s)  # [B, num_actions]
                if self.use_twin_critics:
                    q2_all = self.q2(s)  # [B, num_actions]
                    q_all = torch.min(q1_all, q2_all)  # Use min(Q1, Q2) for policy improvement
                else:
                    q_all = q1_all

            # Policy loss: Σ_a π(a|s)[α*log π(a|s) - Q(s,a)]
            pi_loss = ((alpha * log_probs) - q_all) * probs  # [B, num_actions]
            pi_loss = pi_loss.sum(-1).mean()

            self.opt_pi.zero_grad()
            pi_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.opt_pi.step()

            # =====================================================
            # 3) Temperature (alpha) update (if enabled, update with policy)
            # =====================================================
            if self.use_auto_entropy:
                with torch.no_grad():
                    # Entropy of current policy: H(π(s)) = -Σ π(a|s) log π(a|s)
                    entropy = -(log_probs * probs).sum(-1, keepdim=True)  # [B, 1]

                # Alpha loss: minimize log_alpha * (entropy - target_entropy)
                alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # =====================================================
        # 4) Soft-update target critic(s)
        # =====================================================
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_targ.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

            if self.use_twin_critics:
                for p, pt in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "alpha_loss": float(alpha_loss.item()) if self.use_auto_entropy else 0.0,
            "alpha": float(alpha.item()) if self.use_auto_entropy else self.alpha
        }

    def save(self, path: str):
        state_dict = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q1_targ": self.q1_targ.state_dict(),
            "use_twin_critics": self.use_twin_critics
        }

        if self.use_twin_critics:
            state_dict["q2"] = self.q2.state_dict()
            state_dict["q2_targ"] = self.q2_targ.state_dict()

        torch.save(state_dict, path)

    def load(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(sd["policy"])

        # Load q1 and q1_targ
        self.q1.load_state_dict(sd["q1"])
        self.q1_targ.load_state_dict(sd["q1_targ"])

        # Load q2 and q2_targ if twin critics are enabled
        if self.use_twin_critics and "q2" in sd:
            self.q2.load_state_dict(sd["q2"])
            self.q2_targ.load_state_dict(sd["q2_targ"])

    def get_training_steps(self, exp_cfg: dict) -> int:
        """Get total number of environment steps for SAC.

        SAC uses total_steps from algorithm config.
        Experiment config is ignored for SAC (consistent with PINN-PI/PINN-SPI).
        """
        return self.total_steps

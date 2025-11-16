"""PINN-SPI: Physics-Informed Neural Network Stochastic Policy Iteration with entropy regularization."""

from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from ..algs.base import Algorithm
from ..core.normalization import StateNormalization

Tensor = torch.Tensor


# Neural Network Building Blocks

class Block(nn.Module):
    """width → width fully-connected block (Linear → SiLU)."""
    def __init__(self, width: int):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.fc(x))


class ResidualBlock(nn.Module):
    """width → width fully-connected residual block (Linear → SiLU) + identity skip."""
    def __init__(self, width: int):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.fc(x))  # pre-activation skip


# Value Network

class ValueNet(nn.Module):
    """Value function network f : ℝᵈ → ℝ. Returns [B] (squeezed)."""
    def __init__(
        self,
        d: int,
        width: int = 100,
        depth: int = 2,
        residual: bool = False,
        normalize_inputs: bool = False,
        max_x: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        _Block = ResidualBlock if residual else Block

        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None:
                raise ValueError("max_x must be provided when normalize_inputs=True")
            self.state_norm = StateNormalization(max_x)
        else:
            self.state_norm = None

        self.input_proj = nn.Sequential(nn.Linear(d, width), nn.SiLU())
        blocks = [_Block(width) for _ in range(max(depth - 1, 0))]
        self.trunk = nn.Sequential(*blocks)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        if self.normalize_inputs:
            x = self.state_norm(x)
        h = self.input_proj(x)
        h = self.trunk(h)
        return self.out(h).squeeze(-1)


# Policy Network

class PolicyNet(nn.Module):
    """Stochastic policy π_φ(u | x). forward() returns torch.distributions.Normal."""
    def __init__(
        self,
        d: int,
        m: int,
        U_min: float | torch.Tensor,
        U_max: float | torch.Tensor,
        *,
        width: int = 100,
        depth: int = 2,
        head_width: int = 100,
        head_depth: int = 1,
        log_std_min: float = -2.0,
        log_std_max: float = 5.0,
        residual: bool = False,
        bounder: str = "scaled_tanh",
        sampling_scale: str = "basic",
        normalize_inputs: bool = False,
        max_x: Optional[torch.Tensor] = None,
    ):
        """Args:
            sampling_scale: "basic" (original), "jacobian" (corrected), or "none"
            normalize_inputs: Normalize states to [-1, 1]
            max_x: State bounds (required if normalize_inputs=True)
        """
        super().__init__()
        _Block = ResidualBlock if residual else Block

        valid_scales = ["jacobian", "basic", "none"]
        if sampling_scale not in valid_scales:
            raise ValueError(f"sampling_scale must be one of {valid_scales}, got '{sampling_scale}'")

        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None:
                raise ValueError("max_x must be provided when normalize_inputs=True")
            self.state_norm = StateNormalization(max_x)
        else:
            self.state_norm = None
        self.register_buffer("U_min", torch.as_tensor(U_min, dtype=torch.float32))
        self.register_buffer("U_max", torch.as_tensor(U_max, dtype=torch.float32))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.bounder = bounder
        self.sampling_scale = sampling_scale

        self.input_proj = nn.Sequential(nn.Linear(d, width), nn.SiLU())
        trunk_blocks = [_Block(width) for _ in range(max(depth - 1, 0))]
        self.trunk = nn.Sequential(*trunk_blocks)

        self.mean_head = self._make_head(width, m, head_width, head_depth)
        self.log_std_head = self._make_head(width, m, head_width, head_depth)

    @staticmethod
    def _make_head(in_dim, out_dim, width, depth):
        """Multi-layer head with SiLU activation."""
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Returns torch.distributions.Normal."""
        if self.normalize_inputs:
            x = self.state_norm(x)
        h = self.input_proj(x)
        h = self.trunk(h)

        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def _expand_dist(self, dist, n):
        """Expand distribution for n samples."""
        return dist.expand((n, *dist.batch_shape))

    def _squash_and_scale(self, raw):
        """Apply tanh squash and affine scaling with Jacobian correction."""
        tanh_raw = torch.tanh(raw)
        scale = 0.5 * (self.U_max - self.U_min)
        bias = 0.5 * (self.U_max + self.U_min)
        action = bias + scale * tanh_raw

        eps = 1e-6
        log_one_minus_tanh2 = torch.log1p(-tanh_raw.pow(2) + eps)
        log_scale = torch.log(scale + eps)
        log_abs_det_jac = (log_scale + log_one_minus_tanh2).sum(dim=-1)

        return action, log_abs_det_jac, tanh_raw

    def _action_and_logp_basic(self, dist_exp, *, use_rsample: bool):
        """Basic sampling without Jacobian correction (original experimental implementation)."""
        # Sample from base distribution [n, B, m]
        raw = dist_exp.rsample() if use_rsample else dist_exp.sample()

        # Original scaling pattern
        if self.bounder == "scaled_tanh":
            # Divide by U_max BEFORE tanh (changes temperature)
            bounded = torch.tanh(raw / self.U_max)
            ht_actions = self.U_max * bounded  # Scale back

            if use_rsample:
                # rsample: log_prob(raw) without Jacobian correction
                log_probs = dist_exp.log_prob(raw).sum(dim=-1)
            else:
                # sample: log_prob(bounded) - mathematically incorrect
                log_probs = dist_exp.log_prob(bounded).sum(dim=-1)

        elif self.bounder == "clamp":
            bounded = torch.clamp(raw, -self.U_max, self.U_max)
            ht_actions = self.U_max * bounded
            log_probs = dist_exp.log_prob(raw if use_rsample else bounded).sum(dim=-1)
        else:
            raise ValueError(f"Unknown bounder: {self.bounder}")

        return ht_actions.transpose(0, 1), log_probs.transpose(0, 1)

    def _action_and_logp_corrected(self, dist_exp, *, use_rsample: bool):
        """Corrected sampling with proper Jacobian correction."""
        # Sample from base distribution [n, B, m]
        raw = dist_exp.rsample() if use_rsample else dist_exp.sample()

        # Compute base log-prob (pre-transform)
        base_logp = dist_exp.log_prob(raw).sum(dim=-1)  # [n, B]

        # Apply transform with Jacobian correction
        if self.bounder == "scaled_tanh":
            action, log_abs_det_jac, _ = self._squash_and_scale(raw)
            # Corrected log-prob: log π(action) = log π(raw) + log |det J|
            log_probs = base_logp + log_abs_det_jac  # [n, B]
        elif self.bounder == "clamp":
            # Clamp is not differentiable and has no tractable Jacobian
            # Only use for evaluation, not training
            action = torch.clamp(raw, self.U_min, self.U_max)
            log_probs = base_logp  # Approximate (not technically correct)
        else:
            raise ValueError(f"Unknown bounder: {self.bounder}")

        # Transpose to [B, n, m] and [B, n]
        return action.transpose(0, 1), log_probs.transpose(0, 1)

    def _action_and_logp_none(self, dist_exp, *, use_rsample: bool):
        """
        No scaling - return raw samples from the policy network.

        This mode returns the raw samples from the Gaussian policy without
        any tanh squashing or scaling. Useful for:
        - Unbounded action spaces
        - When environment handles clipping internally
        - Debugging/analysis of raw policy outputs

        Args:
            dist_exp: Expanded base Normal distribution [n, B, m]
            use_rsample: If True, use rsample() for gradients; else sample()

        Returns:
            actions: [B, n, m] raw actions from Normal(μ, σ)
            log_probs: [B, n] log probabilities from base distribution
        """
        # Sample from base distribution [n, B, m]
        raw = dist_exp.rsample() if use_rsample else dist_exp.sample()

        # Compute log-prob (no transformation, no Jacobian)
        log_probs = dist_exp.log_prob(raw).sum(dim=-1)  # [n, B]

        # Return raw samples without any scaling
        return raw.transpose(0, 1), log_probs.transpose(0, 1)

    def _action_and_logp_from(self, dist_exp, *, use_rsample: bool):
        """
        Dispatch to sampling method based on sampling_scale configuration.

        Args:
            dist_exp: Expanded base Normal distribution [n, B, m]
            use_rsample: If True, use rsample() for gradients; else sample()

        Returns:
            actions: [B, n, m] actions (scale depends on sampling_scale)
            log_probs: [B, n] log probabilities
        """
        if self.sampling_scale == "jacobian":
            return self._action_and_logp_corrected(dist_exp, use_rsample=use_rsample)
        elif self.sampling_scale == "basic":
            return self._action_and_logp_basic(dist_exp, use_rsample=use_rsample)
        elif self.sampling_scale == "none":
            return self._action_and_logp_none(dist_exp, use_rsample=use_rsample)
        else:
            raise ValueError(f"Unknown sampling_scale: {self.sampling_scale}")

    def sample(self, x, n_samples=1):
        """Sample actions WITHOUT gradients (for evaluation)."""
        with torch.no_grad():
            dist = self.forward(x)
            dist_exp = self._expand_dist(dist, n_samples)
            return self._action_and_logp_from(dist_exp, use_rsample=False)

    def rsample(self, x, n_samples=1):
        """Sample actions WITH gradients (for policy improvement)."""
        dist = self.forward(x)
        dist_exp = self._expand_dist(dist, n_samples)
        return self._action_and_logp_from(dist_exp, use_rsample=True)


class DiscretePolicyNet(nn.Module):
    """
    Discrete stochastic policy network: π_φ(a | x) where a ∈ {0, 1, ..., num_actions-1}.

    Outputs logits for Categorical distribution over discrete actions.
    Used for environments like CartPole with discrete action spaces.
    """
    def __init__(
        self,
        d: int,
        num_actions: int,
        *,
        width: int = 100,
        depth: int = 2,
        head_width: int = 100,
        head_depth: int = 1,
        residual: bool = False,
        normalize_inputs: bool = False,
        max_x: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        _Block = ResidualBlock if residual else Block

        self.num_actions = num_actions

        # Optional input normalization: map raw domain to [-1, 1]
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            if max_x is None:
                raise ValueError("max_x must be provided when normalize_inputs=True")
            self.state_norm = StateNormalization(max_x)
        else:
            self.state_norm = None

        # Backbone network: state -> hidden representation
        layers = [nn.Linear(d, width), nn.SiLU()]
        for _ in range(depth):
            layers.append(_Block(width))
        self.backbone = nn.Sequential(*layers)

        # Head network: hidden -> logits
        head_layers = []
        for _ in range(head_depth):
            head_layers.extend([nn.Linear(width, head_width), nn.SiLU()])
        head_layers.append(nn.Linear(head_width if head_depth > 0 else width, num_actions))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        """
        Forward pass returns Categorical distribution over actions.

        Args:
            x: State tensor [B, d]

        Returns:
            dist: torch.distributions.Categorical with logits for each action
        """
        # Normalize inputs if enabled
        if self.normalize_inputs:
            x = self.state_norm(x)
        hidden = self.backbone(x)
        logits = self.head(hidden)  # [B, num_actions]
        return torch.distributions.Categorical(logits=logits)

    def act(self, x):
        """Deterministic action (argmax of logits)."""
        dist = self.forward(x)
        return dist.logits.argmax(dim=-1).unsqueeze(-1).float()  # [B, 1]

    def sample(self, x, n_samples=1):
        """
        Sample discrete actions WITHOUT gradients (for evaluation).

        Args:
            x: State tensor [B, d]
            n_samples: Number of samples per state

        Returns:
            actions: [B, n_samples] discrete action indices
            log_probs: [B, n_samples] log probabilities
        """
        dist = self.forward(x)  # [B, num_actions]

        if n_samples == 1:
            actions = dist.sample()  # [B]
            log_probs = dist.log_prob(actions)  # [B]
            return actions.unsqueeze(1).float(), log_probs.unsqueeze(1)  # [B, 1], [B, 1]
        else:
            # Sample multiple actions per state
            actions = torch.stack([dist.sample() for _ in range(n_samples)], dim=1)  # [B, n_samples]
            # Compute log_prob for each action sample separately
            # dist.log_prob expects [B] input, so we compute for each column
            log_probs = torch.stack([dist.log_prob(actions[:, i]) for i in range(n_samples)], dim=1)  # [B, n_samples]
            return actions.float(), log_probs

    def rsample(self, x, n_samples=1):
        """
        Sample discrete actions WITH gradients (Gumbel-Softmax reparameterization).

        Uses Gumbel-Softmax trick for differentiable sampling. Returns integer actions
        but maintains gradients through the soft distribution for policy improvement.

        Note: For discrete actions in PINN-SPI, the policy improvement step uses
        weighted log-probabilities, so we compute log_probs from the Categorical
        distribution (not from the Gumbel-perturbed distribution).

        Args:
            x: State tensor [B, d]
            n_samples: Number of samples per state

        Returns:
            actions: [B, n_samples] discrete action indices (as floats)
            log_probs: [B, n_samples] log probabilities from the base Categorical
        """
        dist = self.forward(x)  # Categorical distribution [B, num_actions]
        logits = dist.logits  # [B, num_actions]
        batch_size = x.shape[0]

        if n_samples == 1:
            # Single sample: Gumbel-Softmax for differentiable sampling
            # Temperature = 1.0 for standard Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            logits_with_gumbel = logits + gumbel_noise
            # Hard sampling: take argmax but keep soft gradients via straight-through
            actions_soft = F.softmax(logits_with_gumbel, dim=-1)  # [B, num_actions]
            actions_hard = actions_soft.argmax(dim=-1)  # [B]

            # Straight-through estimator: forward uses hard, backward uses soft
            actions_ste = (actions_hard.float() - actions_soft.detach()).requires_grad_() + actions_soft

            # Log probability from base Categorical (used in PINN-SPI's weighted objective)
            log_probs = dist.log_prob(actions_hard)  # [B]

            return actions_hard.unsqueeze(1).float(), log_probs.unsqueeze(1)  # [B, 1], [B, 1]
        else:
            # Multiple samples: independently sample n times
            actions_list = []
            log_probs_list = []

            for _ in range(n_samples):
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
                logits_with_gumbel = logits + gumbel_noise
                actions_soft = F.softmax(logits_with_gumbel, dim=-1)
                actions_hard = actions_soft.argmax(dim=-1)

                # Log prob from base distribution
                log_probs_list.append(dist.log_prob(actions_hard))
                actions_list.append(actions_hard)

            actions = torch.stack(actions_list, dim=1).float()  # [B, n_samples]
            log_probs = torch.stack(log_probs_list, dim=1)  # [B, n_samples]

            return actions, log_probs


# ============================================================================
# Helper Functions (EXACTLY from original)
# ============================================================================

def grad(x, v_func):
    """Compute gradient of value function w.r.t. states."""
    x = x.clone().requires_grad_(True)
    v_out = v_func(x)
    return torch.autograd.grad(v_out.sum(), x, create_graph=True)[0]


def grad_and_hess_diag(x, v_func):
    """Compute gradient and diagonal of Hessian (for Laplacian)."""
    x = x.clone().requires_grad_(True)
    v_out = v_func(x)
    grad_v = torch.autograd.grad(v_out.sum(), x, create_graph=True)[0]
    hess_diag = []
    for i in range(x.shape[1]):
        g_i = grad_v[:, i].sum()
        h_ii = torch.autograd.grad(g_i, x, retain_graph=True)[0][:, i]
        hess_diag.append(h_ii)
    hess_diag = torch.stack(hess_diag, dim=1)
    return grad_v, hess_diag


def trace_sigma_d2v(x, u, v_func, sigma_val: float, d_x: int):
    """
    Compute trace of (Σ · Hessian(V)) for diffusion term in HJB equation.

    EXACTLY matching original implementation.

    Args:
        x: States [B, d]
        u: Actions [B, m]
        v_func: Value function network
        sigma_val: Diffusion coefficient (scalar, should be 0.01)
        d_x: State dimension

    Returns:
        trace: [B] trace term for each state
    """
    # Compute diffusion matrix: Σ = σ * I
    S = sigma_val * torch.eye(d_x, device=x.device).unsqueeze(0).expand(x.shape[0], d_x, d_x)

    # Compute Σ @ Σ^T (for diagonal diffusion, this is σ² * I)
    if S.dim() > 2:
        Sigma = S @ S.transpose(-1, -2)
    elif S.dim() == 0:
        Sigma = S.unsqueeze(0).unsqueeze(0)

    # Compute Hessian diagonal
    _, hess_diag = grad_and_hess_diag(x, v_func)

    # Extract diagonal of Σ
    Sigma_diag = Sigma.diagonal(dim1=-2, dim2=-1)

    # Compute trace: sum(Σ_diag * Hess_diag)
    return (Sigma_diag * hess_diag).sum(dim=-1)  # [B]


# ============================================================================
# Main Algorithm
# ============================================================================

class PINNSPI(Algorithm):
    """
    Physics-Informed Neural Network Stochastic Policy Iteration.

    IMPORTANT: This implementation EXACTLY matches the original experimental script.
    """
    def __init__(self, d: int, m: int, cfg: dict, device: torch.device, env, expt):
        # Get normalization parameters
        normalize_inputs = cfg.get("normalize_inputs", False)
        max_x = None
        if normalize_inputs:
            # Get max_x from environment bounds
            state_low, state_high = env.get_state_sample_bounds()
            # Assume symmetric bounds or use the maximum absolute value
            max_x = torch.maximum(state_high.abs(), state_low.abs())

        # Networks (EXACTLY matching original architecture)
        self.v = ValueNet(
            d,
            width=cfg.get("nn_v_width", 100),
            depth=cfg.get("nn_v_depth", 2),
            residual=cfg.get("nn_v_residual", False),
            normalize_inputs=normalize_inputs,
            max_x=max_x,
        ).to(device)

        # Create policy network (discrete or continuous based on environment)
        if env.is_discrete_action:
            # Discrete action policy (e.g., CartPole)
            self.policy = DiscretePolicyNet(
                d,
                num_actions=env.num_discrete_actions,
                width=cfg.get("nn_p_width", 100),
                depth=cfg.get("nn_p_depth", 2),
                head_width=cfg.get("nn_p_head_width", 100),
                head_depth=cfg.get("nn_p_head_depth", 1),
                residual=cfg.get("nn_p_residual", False),
                normalize_inputs=normalize_inputs,
                max_x=max_x,
            ).to(device)
        else:
            # Continuous action policy (e.g., LQR)
            action_low, action_high = env.get_action_sample_bounds()
            U_min = action_low[0].item() if action_low.numel() == 1 else action_low
            U_max = action_high[0].item() if action_high.numel() == 1 else action_high

            self.policy = PolicyNet(
                d,
                m,
                U_min=U_min,
                U_max=U_max,
                width=cfg.get("nn_p_width", 100),
                depth=cfg.get("nn_p_depth", 2),
                head_width=cfg.get("nn_p_head_width", 100),
                head_depth=cfg.get("nn_p_head_depth", 1),
                residual=cfg.get("nn_p_residual", False),
                bounder=cfg.get("bounder", "scaled_tanh"),
                sampling_scale=cfg.get("sampling_scale", "basic"),
                normalize_inputs=normalize_inputs,
                max_x=max_x,
            ).to(device)

        # Optimizers
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=float(cfg["lr_v"]))
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=float(cfg["lr_pi"]))

        # Algorithm hyperparameters
        # dt = expt.get("dt", 0.02)
        # import math
        # default_gamma = math.exp(-float(cfg["rho"]) * dt)
        # self.rho = default_gamma  # Discount rate (it is same as gamma at other algorhtims)
        # self.rho = float(cfg["rho"])  # Discount rate
        self.rho = float(expt["eval"]["rho"])  # Discount rate
        self.lam = float(cfg["lam"])  # Entropy weight λ
        self.env = env
        self.device = device

        # Get sampling bounds from environment (works for LQR and CartPole)
        state_low, state_high = env.get_state_sample_bounds()
        self.low_x = state_low
        self.high_x = state_high
        self.sigma = env.spec.sigma

        # Training schedule
        self.outer_iters = cfg.get("outer_iters", 50)
        self.pi_improve_iters = cfg.get("pi_improve_iters", 1000)
        self.pi_improve_resample = cfg.get("pi_improve_resample", 100)
        self.v_eval_iters = cfg.get("v_eval_iters", 1000)
        self.v_eval_resample = cfg.get("v_eval_resample", 100)
        self.N_x = cfg.get("N_x", 100)
        self.N_u = cfg.get("N_u", 100)

        # Reward scaling by dt (default: True for continuous-time interpretation)
        # When True: r_scaled = r * dt (continuous-time interpretation, matches other implementations)
        # When False: r_scaled = r (discrete-time interpretation)
        # NOTE: Only applies to evaluation, NOT to HJB training loop (which uses instantaneous rates)
        self.scale_reward_by_dt = cfg.get("scale_reward_by_dt", True)
        self.dt = env.dt  # Store dt for reward scaling

    def to(self, device):
        self.device = device
        self.v.to(device)
        self.policy.to(device)
        return self

    def train_mode(self):
        self.v.train()
        self.policy.train()

    def eval_mode(self):
        self.v.eval()
        self.policy.eval()

    def act(self, x: Tensor, deterministic: bool = False) -> Tensor:
        """Sample action from policy."""
        actions, _ = self.policy.sample(x, n_samples=1)
        return actions.squeeze(1)  # [B, 1, m] -> [B, m]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one outer iteration of policy iteration.
        EXACTLY matching original training loop.
        """
        # Policy Improvement
        pi_loss = self._policy_improvement()

        # Policy Evaluation
        v_loss = self._policy_evaluation()

        return {"pi_loss": pi_loss, "v_loss": v_loss}

    def _policy_improvement(self) -> float:
        """Policy Improvement (EXACTLY from original)."""
        self.policy.train()
        d = self.env.d
        m = self.env.m

        total_loss = 0.0

        for it in range(self.pi_improve_iters):
            self.opt_pi.zero_grad()  # CRITICAL: zero_grad BEFORE resampling (as in original)

            # Resample batch periodically
            if it % self.pi_improve_resample == 0:
                # Sample states using per-dimension bounds (works for LQR and CartPole)
                x_coll = self.low_x + (self.high_x - self.low_x) * torch.rand(self.N_x, d, device=self.device)
                # CRITICAL: x_flat created HERE during resampling (as in original)
                x_flat = x_coll.unsqueeze(1).expand(-1, self.N_u, -1).reshape(-1, d)

            # Sample actions from current policy
            u_flat, log_pi = self.policy.rsample(x_coll, n_samples=self.N_u)  # [N_x, N_u, m], [N_x, N_u]
            u_flat = u_flat.reshape(-1, m)  # CRITICAL: use u_flat not u_flat_reshape

            # Compute advantage using value function (detached for policy training)
            # NOTE: We need gradients to compute trace, but detach the final result
            grad_v = grad(x_flat, self.v)  # grad_v shape = [N_x * N_u, d]

            # Compute instantaneous Hamiltonian: b·∇V + (1/2)tr(σσ^T·D²V) + r
            # This matches the paper algorithm's policy improvement objective
            # NOTE: No dt scaling in training loop (HJB PDE uses instantaneous reward rate)
            h_inst = (
                (self.env.b(x_flat, u_flat) * grad_v).sum(dim=1)
                + 0.5 * trace_sigma_d2v(x_flat, u_flat, self.v, self.sigma, d)
                + self.env.r(x_flat, u_flat)
            )
            h_inst = h_inst.reshape(self.N_x, self.N_u).detach()  # Detach to avoid training V

            # Compute target weights via softmax (entropy regularization)
            target_weights = torch.softmax(h_inst / self.lam, dim=1)  # [N_x, N_u]

            # Policy loss: maximize weighted log probabilities
            pi_loss = -(target_weights * log_pi).sum() / self.N_x

            pi_loss.backward()
            self.opt_pi.step()

            # CRITICAL: Print every 10 steps (as in original)
            if (it + 1) % 10 == 0:
                print(f"Policy Improvement Loss {it + 1}: {pi_loss.item():.4e}")

            total_loss += pi_loss.item()

        avg_loss = total_loss / self.pi_improve_iters
        return avg_loss

    def _policy_evaluation(self) -> float:
        """Policy Evaluation (EXACTLY from original)."""
        self.v.train()
        d = self.env.d
        m = self.env.m

        total_loss = 0.0

        for it in range(self.v_eval_iters):
            # Resample batch periodically
            if it % self.v_eval_resample == 0:
                # Sample states using per-dimension bounds (works for LQR and CartPole)
                x_flat = self.low_x + (self.high_x - self.low_x) * torch.rand(self.N_x, d, device=self.device)

                # Sample actions from current policy (detached)
                with torch.no_grad():
                    u_flat, log_pi = self.policy.sample(x_flat, n_samples=self.N_u)  # [N_x, N_u, m], [N_x, N_u]

                # CRITICAL: Reshape happens HERE during resampling (as in original)
                x_flat = x_flat.unsqueeze(1).expand(-1, self.N_u, -1).reshape(-1, d)
                u_flat = u_flat.reshape(-1, m)  # CRITICAL: use u_flat not u_flat_reshape

            self.opt_v.zero_grad()

            # Compute HJB terms
            # NOTE: No dt scaling in training loop (HJB PDE uses instantaneous reward rate)
            h_inst = (
                (self.env.b(x_flat, u_flat) * grad(x_flat, self.v)).sum(dim=1)
                + 0.5 * trace_sigma_d2v(x_flat, u_flat, self.v, self.sigma, d)
                + self.env.r(x_flat, u_flat)
            )
            h_inst = h_inst.view(self.N_x, self.N_u)  # [N_x, N_u]

            # Subtract entropy term: λ·log(π)
            h_inst -= self.lam * log_pi

            # Average over action samples
            h_inst = h_inst.mean(dim=1)  # [N_x]

            # Compute value at collocation points
            x_coll = x_flat.reshape(self.N_x, self.N_u, d)[:, 0, :]
            v_x = self.v(x_coll)  # [N_x] (already squeezed by ValueNet)

            # PDE residual: ρV - H = 0
            # v_loss = ((self.rho * v_x - h_inst) ** 2).mean()
            v_loss = ((self.rho * v_x - h_inst) ** 2).mean()

            v_loss.backward()
            self.opt_v.step()

            # CRITICAL: Print every 10 steps (as in original)
            if (it + 1) % 10 == 0:
                print(f"Policy Evaluation Loss {it + 1}: {v_loss.item():.4e}")

            total_loss += v_loss.item()

        avg_loss = total_loss / self.v_eval_iters
        return avg_loss

    def save(self, path: str):
        torch.save({
            "v": self.v.state_dict(),
            "policy": self.policy.state_dict()
        }, path)

    def load(self, path: str, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        self.v.load_state_dict(sd["v"])
        self.policy.load_state_dict(sd["policy"])

    def get_eval_params(self, env, eval_cfg: Dict[str, any]) -> Dict[str, any]:
        """Get PINN-SPI specific evaluation parameters.

        PINN-SPI uses:
        - Discounted returns: gamma = exp(-rho * dt)
        - No state clipping during evaluation (matches original)

        Config can override these defaults via eval_cfg.
        """
        import math

        # Default: Compute gamma from rho and dt (original PINN-SPI behavior)
        # gamma = exp(-rho * dt)
        dt = eval_cfg.get("dt", 0.02)
        default_gamma = math.exp(-self.rho * dt)

        # Allow config to override (for extensibility)
        gamma = eval_cfg.get("gamma", default_gamma)
        clip_state = eval_cfg.get("clip_state", False)  # Original doesn't clip

        return {
            "gamma": gamma,
            "clip_state": clip_state,
        }

    def get_training_steps(self, exp_cfg: Dict[str, any]) -> int:
        """Get total number of policy iterations for PINN-SPI.

        PINN-SPI uses outer_iters from algorithm config.
        Experiment config is ignored for PINN-SPI.
        """
        return self.outer_iters

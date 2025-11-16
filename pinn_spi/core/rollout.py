"""Rollout and evaluation functions for SDE environments."""

import math
import torch
from typing import Dict

@torch.no_grad()
def euler_maruyama_step(env, x, u, clip_state=True, dynamics_mode="b"):
    """Single step Euler-Maruyama SDE integration.

    Args:
        env: Environment with dynamics
        x: Current state [B, d]
        u: Action [B, m]
        clip_state: Clip state to domain bounds
        dynamics_mode: "b" (PyTorch physics) or "gym" (Gymnasium physics, batch_size=1 only)
    """
    batch_size = x.shape[0]

    # Noise term
    noise = env.sigma(x, u) @ torch.randn(batch_size, env.d, 1, device=x.device)
    noise_term = noise.squeeze(-1) * math.sqrt(env.dt)

    # Deterministic dynamics
    if dynamics_mode == "gym":
        if not hasattr(env, 'gym_env'):
            raise ValueError(f"dynamics_mode='gym' requires environment with gym_env attribute. "
                           f"Environment {type(env).__name__} does not have gym_env.")

        if batch_size != 1:
            raise ValueError(f"dynamics_mode='gym' only supports batch_size=1, got batch_size={batch_size}. "
                           f"Use dynamics_mode='b' for batched operations.")

        # Action conversion
        gym_env = env.gym_env
        action_np = u.squeeze(0).cpu().numpy()

        import gymnasium as gym
        if isinstance(gym_env.action_space, gym.spaces.Discrete):
            action_gym = int(action_np.item())
        elif isinstance(gym_env.action_space, gym.spaces.Box):
            action_gym = action_np.reshape(-1).clip(gym_env.action_space.low, gym_env.action_space.high)
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(gym_env.action_space)}")

        obs_next, _, _, _, _ = gym_env.step(action_gym)
        x2 = torch.tensor(obs_next, device=x.device, dtype=x.dtype).unsqueeze(0) + noise_term

    elif dynamics_mode == "b":
        drift = env.b(x, u)
        x2 = x + drift * env.dt + noise_term

    else:
        raise ValueError(f"Unknown dynamics_mode: {dynamics_mode}. Must be 'b' or 'gym'.")

    x2 = env.clip_state(x2) if clip_state else x2
    env.sync_state(x2)

    return x2

def evaluate_policy(
    env,
    algo,
    num_traj,
    traj_len,
    x0=None,
    gamma=None,
    clip_state=True,
    project_l2_ball=False,
    l2_radius=0.1,
    dynamics_mode="b",
    *,
    deterministic: bool = True,
    scale_reward_by_dt: bool = False,
    reward_scale: float = 1.0,
    use_env_reset: bool = True,
    seed_offset: int = 10000,
):
    """Vectorized policy evaluation (all trajectories in parallel).

    Args:
        env: Environment with dynamics
        algo: Algorithm to evaluate
        num_traj: Number of trajectories
        traj_len: Trajectory length
        x0: Initial state (None = sampled from environment)
        gamma: Discount factor (None = no discounting)
        clip_state: Clip states to domain
        project_l2_ball: Project initial states onto L2 ball
        l2_radius: L2 ball radius
        dynamics_mode: "b" (PyTorch) or "gym" (Gymnasium)
        deterministic: Use deterministic actions
        scale_reward_by_dt: Scale rewards by dt
        reward_scale: Base reward scale
        use_env_reset: Reset environment for initial states
        seed_offset: Seed offset for resets

    Returns:
        dict: {G: returns [num_traj], X: states [traj_len+1, num_traj, d]}
    """
    algo.eval_mode()

    if x0 is None:
        # Sample initial states
        if use_env_reset and hasattr(env, 'gym_env') and hasattr(env.gym_env, 'reset'):
            initial_states = []
            for i in range(num_traj):
                try:
                    obs = env.gym_env.reset(seed=seed_offset + i)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                except TypeError:
                    obs = env.gym_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                if isinstance(obs, torch.Tensor):
                    obs_tensor = obs.detach().clone().to(dtype=torch.float32, device=algo.device)
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algo.device)
                initial_states.append(obs_tensor)

            x = torch.stack(initial_states, dim=0)

        elif use_env_reset and hasattr(env, 'reset'):
            try:
                obs = env.reset(batch_size=num_traj, seed=seed_offset)
                if isinstance(obs, tuple):
                    obs = obs[0]
            except TypeError:
                try:
                    obs = env.reset(num_traj)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                except TypeError:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
            if isinstance(obs, torch.Tensor):
                x = obs.detach().clone().to(dtype=torch.float32, device=algo.device)
            else:
                x = torch.tensor(obs, dtype=torch.float32, device=algo.device)

            if x.shape[0] != num_traj:
                initial_states = []
                for i in range(num_traj):
                    try:
                        obs = env.reset(batch_size=1, seed=seed_offset + i)
                        if isinstance(obs, tuple):
                            obs = obs[0]
                    except TypeError:
                        obs = env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]

                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.detach().clone().to(dtype=torch.float32, device=algo.device).squeeze(0)
                    else:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algo.device).squeeze(0)
                    initial_states.append(obs_tensor)
                x = torch.stack(initial_states, dim=0)
        elif hasattr(env, 'sample_initial_state'):
            x = env.sample_initial_state(num_traj, device=algo.device)
        else:
            x = torch.zeros(num_traj, env.d, device=algo.device)
    else:
        x = x0.to(algo.device)
    if project_l2_ball:
        norms = torch.norm(x, dim=1, keepdim=True)
        mask = norms > l2_radius
        x = torch.where(mask, x * (l2_radius / norms), x)

    scale = reward_scale
    if scale_reward_by_dt:
        scale *= float(getattr(env, "dt", 1.0))

    G = torch.zeros(num_traj, device=algo.device)
    X = torch.zeros(traj_len + 1, num_traj, env.d, device=algo.device)
    X[0] = x

    for t in range(traj_len):
        u = algo.act(X[t], deterministic=deterministic)
        r = env.r(X[t], u) * scale

        G += (r if gamma is None else (gamma**t) * r)
        X[t + 1] = euler_maruyama_step(env, X[t], u, clip_state=clip_state, dynamics_mode=dynamics_mode)

    return dict(G=G, X=X)


def rollout_with_actions(
    env,
    algo,
    num_traj,
    traj_len,
    x0=None,
    clip_state=True,
    dynamics_mode="b",
    *,
    deterministic: bool = True,
    use_env_reset: bool = True,
    seed_offset: int = 10000,
):
    """Roll out trajectories with actions.

    Args:
        env: Environment
        algo: Algorithm
        num_traj: Number of trajectories
        traj_len: Trajectory length
        x0: Initial state (None = sampled)
        clip_state: Clip states to domain
        dynamics_mode: "b" or "gym"
        deterministic: Use deterministic actions
        use_env_reset: Reset environment for initial states
        seed_offset: Seed offset

    Returns:
        dict: {X: states [traj_len+1, num_traj, d], U: actions [traj_len, num_traj, m]}
    """
    algo.eval_mode()

    if x0 is None:
        if use_env_reset and hasattr(env, 'gym_env') and hasattr(env.gym_env, 'reset'):
            initial_states = []
            for i in range(num_traj):
                try:
                    obs = env.gym_env.reset(seed=seed_offset + i)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                except TypeError:
                    obs = env.gym_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]

                if isinstance(obs, torch.Tensor):
                    obs_tensor = obs.detach().clone().to(dtype=torch.float32, device=algo.device)
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algo.device)
                initial_states.append(obs_tensor)

            x = torch.stack(initial_states, dim=0)
        elif use_env_reset and hasattr(env, 'reset'):
            try:
                obs = env.reset(batch_size=num_traj, seed=seed_offset)
                if isinstance(obs, tuple):
                    obs = obs[0]
            except TypeError:
                try:
                    obs = env.reset(num_traj)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                except TypeError:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]

            if isinstance(obs, torch.Tensor):
                x = obs.detach().clone().to(dtype=torch.float32, device=algo.device)
            else:
                x = torch.tensor(obs, dtype=torch.float32, device=algo.device)

            if x.shape[0] != num_traj:
                initial_states = []
                for i in range(num_traj):
                    try:
                        obs = env.reset(batch_size=1, seed=seed_offset + i)
                        if isinstance(obs, tuple):
                            obs = obs[0]
                    except TypeError:
                        obs = env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]

                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.detach().clone().to(dtype=torch.float32, device=algo.device).squeeze(0)
                    else:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=algo.device).squeeze(0)
                    initial_states.append(obs_tensor)
                x = torch.stack(initial_states, dim=0)
        elif hasattr(env, 'sample_initial_state'):
            x = env.sample_initial_state(num_traj, device=algo.device)
        else:
            x = torch.zeros(num_traj, env.d, device=algo.device)
    else:
        x = x0.to(algo.device)

    X = torch.zeros(traj_len + 1, num_traj, env.d, device=algo.device)
    U = torch.zeros(traj_len, num_traj, env.m, device=algo.device)
    X[0] = x

    for t in range(traj_len):
        u = algo.act(X[t], deterministic=deterministic)
        U[t] = u
        X[t + 1] = euler_maruyama_step(env, X[t], u, clip_state=clip_state, dynamics_mode=dynamics_mode)

    return dict(X=X, U=U)


def evaluate_policy_sequential(
    env,
    algo,
    num_episodes: int,
    max_steps: int,
    gamma=None,
    clip_state=True,
    dynamics_mode="b",
    *,
    deterministic: bool = True,
    scale_reward_by_dt: bool = False,
    reward_scale: float = 1.0,
    use_env_reset: bool = True,
    seed_offset: int = 10000,
):
    """Sequential policy evaluation (one episode at a time, respects termination).

    Args:
        env: Environment
        algo: Algorithm
        num_episodes: Number of episodes
        max_steps: Max steps per episode
        gamma: Discount factor (None = no discounting)
        clip_state: Clip states to domain
        dynamics_mode: "b" or "gym"
        deterministic: Use deterministic actions
        scale_reward_by_dt: Scale rewards by dt
        reward_scale: Base reward scale
        use_env_reset: Reset environment between episodes
        seed_offset: Seed offset

    Returns:
        dict: {G: returns [num_episodes], episode_lengths: [num_episodes], X: list of trajectories}
    """
    algo.eval_mode()

    scale = reward_scale
    if scale_reward_by_dt:
        scale *= float(getattr(env, "dt", 1.0))

    returns = []
    episode_lengths = []
    trajectories = []

    for ep in range(num_episodes):
        if use_env_reset and hasattr(env, 'gym_env') and hasattr(env.gym_env, 'reset'):
            try:
                obs = env.gym_env.reset(seed=seed_offset + ep)
                if isinstance(obs, tuple):
                    obs = obs[0]
            except TypeError:
                obs = env.gym_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
            if isinstance(obs, torch.Tensor):
                x = obs.detach().clone().to(dtype=torch.float32, device=algo.device).unsqueeze(0)
            else:
                x = torch.tensor(obs, dtype=torch.float32, device=algo.device).unsqueeze(0)

        elif use_env_reset and hasattr(env, 'reset'):
            try:
                obs = env.reset(batch_size=1, seed=seed_offset + ep)
                if isinstance(obs, tuple):
                    obs = obs[0]
            except TypeError:
                try:
                    obs = env.reset(1)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                except TypeError:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
            if isinstance(obs, torch.Tensor):
                x = obs.detach().clone().to(dtype=torch.float32, device=algo.device)
            else:
                x = torch.tensor(obs, dtype=torch.float32, device=algo.device)

            if x.dim() == 1:
                x = x.unsqueeze(0)
        elif hasattr(env, 'sample_initial_state'):
            x = env.sample_initial_state(1, device=algo.device)
        else:
            x = torch.zeros(1, env.d, device=algo.device)
        done = False
        G = 0.0
        disc = 1.0
        episode_states = [x.clone()]

        for t in range(max_steps):
            u = algo.act(x, deterministic=deterministic)
            r = env.r(x, u) * scale
            if hasattr(env, 'gym_env'):
                done = (r.item() == 0.0)
            elif hasattr(env, 'terminate_radius'):
                if env.terminate_radius is not None:
                    done = (torch.norm(x) > env.terminate_radius).item()
            elif hasattr(env, 'horizon'):
                if env.horizon is not None:
                    done = (t + 1 >= env.horizon)
            G += disc * r.item()
            if gamma is not None:
                disc *= gamma

            if not done:
                x = euler_maruyama_step(env, x, u, clip_state=clip_state, dynamics_mode=dynamics_mode)
                episode_states.append(x.clone())

            if done:
                break

        returns.append(G)
        episode_lengths.append(t + 1)
        trajectories.append(torch.cat(episode_states, dim=0))  # [T+1, d]

    return dict( G=torch.tensor(returns, device=algo.device), episode_lengths=torch.tensor(episode_lengths, device=algo.device), X=trajectories)

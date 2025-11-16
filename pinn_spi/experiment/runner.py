from __future__ import annotations

import os
import json
import time
import yaml
import torch
from dataclasses import dataclass
from ..utils.seeding import seed_everything
from ..utils.logger import MetricLogger
from ..utils.plots import plot_training_progress
from ..core.buffers import Replay
from ..core.rollout import euler_maruyama_step
from ..envs.lqr import load_lqr_from_yaml
from .config import EvaluationConfig, TrainingConfig, RewardScaleConfig
from .evaluation import Evaluator

try:
    from ..envs.cartpole import load_cartpole_from_yaml
    CARTPOLE_AVAILABLE = True
except ImportError:
    CARTPOLE_AVAILABLE = False

try:
    from ..envs.pendulum import load_pendulum_from_yaml
    PENDULUM_AVAILABLE = True
except ImportError:
    PENDULUM_AVAILABLE = False

try:
    from ..envs.halfcheetah import load_halfcheetah_from_yaml
    HALFCHEETAH_AVAILABLE = True
except ImportError:
    HALFCHEETAH_AVAILABLE = False


from ..algs.sac import SAC
from ..algs.pinn_spi import PINNSPI
try:
    from ..algs.PPO import PPO
    PPO_AVAILABLE = True
except Exception:
    PPO_AVAILABLE = False


@dataclass
class RunHandles:
    """Container for experiment components"""

    env: object
    algo: object
    results_dir: str
    logger: MetricLogger
    replay: Replay | None


def load_yaml(path: str):
    """Load YAML configuration file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(
    env_cfg_path: str, dt: float, device: torch.device, verbose: bool = False
):
    """Build environment from configuration"""
    env_cfg = load_yaml(env_cfg_path)
    if env_cfg["name"] == "lqr":
        return load_lqr_from_yaml(env_cfg, dt, device, verbose=verbose)
    elif env_cfg["name"] == "cartpole":
        if not CARTPOLE_AVAILABLE:
            raise ImportError("CartPole environment requires gymnasium. Install with: pip install gymnasium")
        return load_cartpole_from_yaml(env_cfg, dt, device, verbose=verbose)
    elif env_cfg["name"] == "pendulum":
        if not PENDULUM_AVAILABLE:
            raise ImportError("Pendulum environment requires gymnasium. Install with: pip install gymnasium")
        return load_pendulum_from_yaml(env_cfg, dt, device, verbose=verbose)
    elif env_cfg["name"] == "halfcheetah":
        if not HALFCHEETAH_AVAILABLE:
            raise ImportError("Pendulum environment requires gymnasium. Install with: pip install gymnasium")
        return load_halfcheetah_from_yaml(env_cfg, dt, device, verbose=verbose)
    else:
        raise NotImplementedError(f"Environment {env_cfg['name']} not implemented")


def build_algo(algo_cfg_path: str, env, expt, device: torch.device):
    """Build algorithm from configuration"""
    cfg = load_yaml(algo_cfg_path)
    if cfg["name"] == "sac":
        # Get action scale from environment bounds (works for LQR and CartPole)
        action_low, action_high = env.get_action_sample_bounds()
        scale = action_high[0].item() if action_high.numel() == 1 else action_high.max().item()
        return SAC(env.d, env.m, scale, cfg["params"], expt, device, env)
    elif cfg["name"] == "pinnpi":
        # PINN-PI support has been removed in this branch
        raise NotImplementedError("PINN-PI has been removed; use 'pinnspi' or 'sac'.")
    elif cfg["name"] == "pinnspi":
        # Only check gym_env for environments that have it (CartPole, HalfCheetah)
        # LQR and Pendulum don't have gym_env attribute
        if hasattr(env, 'gym_env') and hasattr(env.gym_env, 'spec'):
            print("env.gym_env.name", env.gym_env.spec.id)
            # Train dynamics model for HalfCheetah if specified
            if env.gym_env.spec.id == 'HalfCheetah-v5' and hasattr(env, 'train_dynamics') and env.train_dynamics:
                dynamic_params = env.spec.train_dynamic_params
                env.run_train_dynamics(dynamic_params)
        return PINNSPI(env.d, env.m, cfg["params"], device, env, expt)
    elif cfg["name"] == "ppo":
        if not PPO_AVAILABLE:
            raise ImportError("PPO module not available")
        return PPO(env.d, env.m, cfg["params"], expt, device, env)
    else:
        raise NotImplementedError(f"Algorithm {cfg['name']} not implemented")


def prepare_run(
    exp_cfg_path: str, env_cfg_path: str, algo_cfg_path: str, tag: str = ""
):
    """Prepare experiment run with all components"""
    exp = load_yaml(exp_cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(exp["seed"])

    # Check for verbose flag
    verbose = exp.get("verbose", False)
    if verbose:
        print(f"🔧 Verbose mode enabled")
        print(f"📁 Experiment config: {exp_cfg_path}")
        print(f"🌍 Environment config: {env_cfg_path}")
        print(f"🧠 Algorithm config: {algo_cfg_path}")
        print(f"🏷️  Tag: {tag}")
        print(f"🎯 Device: {device}")
        print(f"🎲 Seed: {exp['seed']}")

    # Time step from experiment configuration
    dt = float(exp["eval"]["dt"])

    # Compute gamma from rho (discount rate)
    # gamma = exp(-rho * dt) converts continuous discount rate to discrete discount factor
    # All algorithms now use rho-based discounting for consistency
    import math
    rho = float(exp["eval"]["rho"])
    gamma = math.exp(-rho * dt)
    exp["eval"]["gamma"] = gamma

    if verbose:
        print(f"📊 Discount: rho={rho}, dt={dt}, gamma={gamma:.6f}")

    env = build_env(env_cfg_path, dt, device, verbose=verbose)
    algo = build_algo(algo_cfg_path, env, exp, device)

    # Create results directory
    stamp = time.strftime("%Y%m%d-%H%M%S")
    algo_name = cfg_name(algo_cfg_path)
    results_dir = os.path.join(exp["results_root"], f"{stamp}_{tag or algo_name}")
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging
    logger = MetricLogger(os.path.join(results_dir, "metrics.jsonl"))

    # Set up replay buffer for off-policy algorithms (extensible approach)
    # SAC uses replay buffer, PINN-PI/PINN-SPI use collocation (no buffer needed)
    algo_cfg = load_yaml(algo_cfg_path)
    if algo_cfg["name"] == "sac":
        buffer_size = algo_cfg["params"].get("buffer_size", 1_000_000)
        replay = Replay(maxlen=buffer_size, device=device)
    else:
        replay = None

    # Save configuration manifest
    with open(os.path.join(results_dir, "config_manifest.json"), "w") as f:
        json.dump(
            {"experiment": exp_cfg_path, "env": env_cfg_path, "algo": algo_cfg_path},
            f,
            indent=2,
        )

    return RunHandles(env, algo, results_dir, logger, replay), exp


def cfg_name(path: str) -> str:
    """Extract configuration name from path"""
    return os.path.splitext(os.path.basename(path))[0]


def train_loop(handles: RunHandles, exp_cfg: dict):
    """Main training loop for all algorithms.

    This function provides a unified training interface that works with any algorithm
    implementing the Algorithm protocol. Algorithm-specific behavior is handled through
    protocol methods (get_training_steps, get_eval_params) for maximum extensibility.
    """
    env, algo, logger, replay = (
        handles.env,
        handles.algo,
        handles.logger,
        handles.replay,
    )

    # Initialize configuration objects
    train_config = TrainingConfig.from_dict(exp_cfg)
    eval_config = EvaluationConfig.from_dict(exp_cfg)
    reward_config = RewardScaleConfig.from_algo_and_env(algo, env)

    # Get total training steps from algorithm
    total = algo.get_training_steps(exp_cfg)
    train_config.total_steps = total

    # Create evaluator
    evaluator = Evaluator(env, algo, eval_config, reward_config)

    # Helper functions
    def progress_marker(step: int) -> str:
        return f"[{min(step, total)}/{total}]"

    # Convenience aliases
    eval_every = eval_config.every_steps
    save_every = train_config.checkpoint_every
    print_interval = train_config.print_interval
    dynamics_mode = eval_config.dynamics_mode
    reward_scale_factor = reward_config.effective_scale

    # Initialize state
    x = torch.zeros(1, env.d, device=algo.device)

    # PPO-specific initialization
    is_ppo = PPO_AVAILABLE and algo.__class__.__name__ == 'PPO'
    if is_ppo:
        num_envs = getattr(algo, 'num_envs', 16)
        rollout_steps = getattr(algo, 'rollout_steps', 2048)
        gae_lambda = getattr(algo, 'gae_lambda', 0.95)
        gamma = eval_config.gamma

        reset_sampler = getattr(env, "sample_initial_state", None)
        if reset_sampler is not None:
            def _sample_initial(n: int):
                return reset_sampler(n, algo.device)
        else:
            state_low, state_high = env.get_state_sample_bounds()
            def _sample_initial(n: int):
                return state_low + (state_high - state_low) * torch.rand(n, env.d, device=algo.device)

        x_vec = _sample_initial(num_envs)
        buf_s, buf_a, buf_logp, buf_r, buf_v, buf_done = [], [], [], [], [], []
        steps_collected = 0
        ppo_step_counts = torch.zeros(num_envs, device=algo.device, dtype=torch.long)
        terminate_radius = getattr(env, "terminate_radius", None)
        horizon = getattr(env, "horizon", None)

    # Initial evaluation
    results = evaluator.evaluate()
    metrics = evaluator.log_results(logger, results, step=0)
    G, std_ret0 = metrics["avg_return"], metrics["std_return"]

    step_type = "Step"
    training_time = logger.get_training_time()
    wall_time = logger.get_wall_time()
    print(
        f"{progress_marker(0)} {step_type} 0: avg_return = {G:.3f} ± {std_ret0:.3f}, "
        f"training_time = {training_time:.1f}s, wall_time = {wall_time:.1f}s"
    )

    env_steps = 0
    for t in range(1, total + 1):
        logger.start_training_timer()

        if replay is None:
            if is_ppo:
                # PPO on-policy collection
                with torch.no_grad():
                    s_t = x_vec
                    a_t, logp_t = algo.sample_action_and_logp(s_t)
                    v_t = algo.value(s_t)
                    r_t = env.r(s_t, a_t) * reward_scale_factor
                    x_vec = euler_maruyama_step(env, s_t, a_t, clip_state=True, dynamics_mode=dynamics_mode)
                    ppo_step_counts += 1

                buf_s.append(s_t)
                buf_a.append(a_t)
                buf_logp.append(logp_t)
                buf_r.append(r_t)
                buf_v.append(v_t)
                done_mask = torch.zeros(num_envs, device=algo.device, dtype=torch.bool)
                if terminate_radius is not None:
                    done_mask |= torch.linalg.norm(x_vec, dim=1) > float(terminate_radius)
                if horizon is not None:
                    done_mask |= ppo_step_counts >= int(horizon)
                buf_done.append(done_mask.float().unsqueeze(-1))
                if done_mask.any():
                    x_vec[done_mask] = _sample_initial(int(done_mask.sum().item()))
                    ppo_step_counts[done_mask] = 0

                steps_collected += num_envs
                env_steps += num_envs

                logs = {}
                # When trajectory segment is full, compute GAE and update
                if len(buf_s) >= rollout_steps:
                    with torch.no_grad():
                        v_next = algo.value(x_vec)  # [num_envs,1]

                    # Stack to [T, N, ...]
                    S = torch.stack(buf_s)              # [T,N,d]
                    A = torch.stack(buf_a)              # [T,N,m or 1]
                    LOGP = torch.stack(buf_logp)        # [T,N,1]
                    R = torch.stack(buf_r)              # [T,N]
                    V = torch.stack(buf_v).squeeze(-1)  # [T,N]
                    Vn = v_next.squeeze(-1)             # [N]
                    DONE = torch.stack(buf_done).squeeze(-1)  # [T,N]

                    # Compute GAE advantages per env
                    Tlen, Nenv = R.shape
                    adv = torch.zeros(Tlen, Nenv, device=algo.device)
                    lastgaelam = torch.zeros(Nenv, device=algo.device)
                    for t_rev in reversed(range(Tlen)):
                        next_v = Vn if t_rev == Tlen - 1 else V[t_rev + 1]
                        nonterm = 1.0 - DONE[t_rev]
                        delta = R[t_rev] + gamma * next_v * nonterm - V[t_rev]
                        lastgaelam = delta + gamma * gae_lambda * lastgaelam * nonterm
                        adv[t_rev] = lastgaelam
                    ret = adv + V

                    # Flatten to [T*N, ...]
                    batch = {
                        's': S.reshape(-1, env.d),
                        'a': A.reshape(-1, env.m),
                        'logp': LOGP.reshape(-1, 1),
                        'adv': adv.reshape(-1, 1),
                        'ret': ret.reshape(-1, 1),
                        'done': DONE.reshape(-1, 1),
                    }

                    logs = algo.update(batch)

                    # Clear buffers
                    buf_s.clear(); buf_a.clear(); buf_logp.clear(); buf_r.clear(); buf_v.clear(); buf_done.clear()
            else:
                # PINN-SPI collocation-based update
                logs = algo.update({"batch_size": 256})
                env_steps += 1
        else:
            # SAC online learning
            with torch.no_grad():
                if t < algo.start_steps:
                    action_low, action_high = env.get_action_sample_bounds()
                    u = action_low + (action_high - action_low) * torch.rand(1, env.m, device=algo.device)
                else:
                    u = algo.act(x)

                x2 = euler_maruyama_step(env, x, u, dynamics_mode=dynamics_mode)
                r = env.r(x, u) * reward_scale_factor
                done = torch.tensor(0.0, device=algo.device)

                replay.add(x.squeeze(0), u.squeeze(0), r, x2.squeeze(0), done)
                x = x2.clone()
                env_steps += 1

                if hasattr(algo, "step"):
                    algo.step()

            logs = {}
            if len(replay) >= max(algo.update_after, algo.batch_size) and t % algo.update_every == 0:
                batch = replay.sample(algo.batch_size)
                logs = algo.update(batch)

        training_duration = logger.end_training_timer()

        # Log metrics
        for k, v in logs.items():
            logger.log_scalar(k, v, step=env_steps)
        logger.log_scalar("timing/step_training_time", training_duration, step=env_steps)
        logger.log_scalar("timing/total_training_time", logger.get_training_time(), step=env_steps)
        logger.log_scalar("timing/wall_time", logger.get_wall_time(), step=env_steps)

        # Print losses
        effective_print_interval = print_interval if print_interval else (100 if (replay is not None or is_ppo) else 1)
        if logs and env_steps % effective_print_interval == 0:
            loss_str = ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
            print(f"  {progress_marker(env_steps)} {step_type} {env_steps}: {loss_str}")

        # Evaluation
        if env_steps > 0 and env_steps % eval_every == 0:
            results = evaluator.evaluate()
            metrics = evaluator.log_results(logger, results, step=env_steps)
            G, std_eval = metrics["avg_return"], metrics["std_return"]
            training_time = logger.get_training_time()
            wall_time = logger.get_wall_time()
            print(
                f"{progress_marker(env_steps)} {step_type} {env_steps}: avg_return = {G:.3f} ± {std_eval:.3f}, "
                f"training_time = {training_time:.1f}s, wall_time = {wall_time:.1f}s"
            )

        # Checkpoint
        if save_every and env_steps > 0 and env_steps % save_every == 0:
            ckpt_path = os.path.join(handles.results_dir, f"checkpoint_{env_steps}.pth")
            algo.save(ckpt_path)

            try:
                plots_dir = os.path.join(handles.results_dir, "plots")
                metrics_path = os.path.join(handles.results_dir, "metrics.jsonl")
                plot_training_progress(metrics_path, plots_dir)
                print(f"  📊 Plots saved to {plots_dir}/")
            except Exception as e:
                print(f"  Warning: Could not generate plots: {e}")

        if env_steps >= total:
            break

    # Final save
    final_path = os.path.join(handles.results_dir, "final_model.pth")
    algo.save(final_path)

    print("\n📊 Generating final plots...")
    try:
        plots_dir = os.path.join(handles.results_dir, "plots")
        metrics_path = os.path.join(handles.results_dir, "metrics.jsonl")
        plot_training_progress(metrics_path, plots_dir)
        print(f"✅ Plots saved to {plots_dir}/")
    except Exception as e:
        print(f"⚠️  Warning: {e}")

    logger.flush()
    logger.close()
    return handles.results_dir

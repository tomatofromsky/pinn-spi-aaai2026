import argparse
import os
import sys
import torch

# Ensure repo root is importable when running this script directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pinn_spi.experiment.runner import prepare_run
from pinn_spi.core.rollout import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--exp", default="configs/experiment/default.yaml",
                       help="Experiment configuration file")
    parser.add_argument("--env", default="configs/envs/lqr5d.yaml",
                       help="Environment configuration file")
    parser.add_argument("--algo", default="configs/algs/sac.yaml",
                       help="Algorithm configuration file")
    parser.add_argument("--ckpt", required=False,
                       help="Path to model checkpoint")
    parser.add_argument("--num_traj", type=int, default=100,
                       help="Number of evaluation trajectories")

    args = parser.parse_args()

    print(f"Evaluating with:")
    print(f"  Environment: {args.env}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Checkpoint: {args.ckpt}")

    handles, exp = prepare_run(args.exp, args.env, args.algo, tag="eval")

    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        handles.algo.load(args.ckpt, map_location="cpu")

    # Evaluate policy
    eval_cfg = exp["eval"]
    deterministic_eval = bool(eval_cfg.get("deterministic", False))
    reward_scale_base = float(getattr(handles.algo, "reward_scale", 1.0))
    scale_reward_dt = bool(getattr(handles.algo, "scale_reward_by_dt", False))

    out = evaluate_policy(
        handles.env,
        handles.algo,
        args.num_traj,
        eval_cfg["traj_len"],
        gamma=eval_cfg.get("gamma"),
        clip_state=eval_cfg.get("clip_state", True),
        project_l2_ball=eval_cfg.get("project_l2_ball", False),
        l2_radius=eval_cfg.get("l2_radius", 0.1),
        dynamics_mode=eval_cfg.get("dynamics_mode", "b"),
        deterministic=deterministic_eval,
        scale_reward_by_dt=scale_reward_dt,
        reward_scale=reward_scale_base,
    )

    returns = out["G"]
    print(f"\nEvaluation Results:")
    mean_ret = returns.mean().item()
    std_ret = returns.std(unbiased=False).item() if returns.numel() > 1 else 0.0
    print(f"  Mean ± Std return: {mean_ret:.3f} ± {std_ret:.3f}")
    print(f"  Min return: {returns.min().item():.3f}")
    print(f"  Max return: {returns.max().item():.3f}")

if __name__ == "__main__":
    main()

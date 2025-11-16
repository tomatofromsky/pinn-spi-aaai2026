#!/usr/bin/env python3
"""
Script to visualize trajectory norms (states and actions) from trained models.

Usage:
    python scripts/plot_trajectory_norms.py --ckpt path/to/checkpoint.pt \
        --env configs/envs/pendulum.yaml \
        --exp configs/experiment/pendulum_expt.yaml \
        --algo configs/algs/pinnspi_pendulum.yaml \
        --out plots/trajectory_norms.png
"""

import argparse
import os
import sys
import torch
import numpy as np

# Ensure repo root is importable when running this script directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pinn_spi.experiment.runner import prepare_run
from pinn_spi.core.rollout import rollout_with_actions
from pinn_spi.utils.plots import plot_trajectory_norms, plot_state_components


def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectory norms from trained model"
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment configuration file"
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Environment configuration file"
    )
    parser.add_argument(
        "--algo",
        required=True,
        help="Algorithm configuration file"
    )
    parser.add_argument(
        "--ckpt",
        required=False,
        help="Path to model checkpoint (if not provided, uses random policy)"
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=10,
        help="Number of trajectories to roll out (default: 10)"
    )
    parser.add_argument(
        "--traj_len",
        type=int,
        default=None,
        help="Length of each trajectory (default: from experiment config)"
    )
    parser.add_argument(
        "--out",
        default="trajectory_norms.png",
        help="Output image path for norm plot (default: trajectory_norms.png)"
    )
    parser.add_argument(
        "--out_components",
        default=None,
        help="Output image path for state components plot (optional)"
    )
    parser.add_argument(
        "--plot_individual",
        action="store_true",
        help="Plot individual trajectory norms (not just mean/std)"
    )
    parser.add_argument(
        "--max_individual",
        type=int,
        default=5,
        help="Maximum number of individual trajectories to plot (default: 5)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (default: stochastic for PINN-SPI, deterministic for SAC/PPO)"
    )
    parser.add_argument(
        "--state_labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels for state dimensions (e.g., --state_labels theta omega)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Trajectory Analysis",
        help="Plot title"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Plotting Trajectory Norms")
    print("=" * 60)
    print(f"Environment:  {args.env}")
    print(f"Experiment:   {args.exp}")
    print(f"Algorithm:    {args.algo}")
    print(f"Checkpoint:   {args.ckpt if args.ckpt else 'None (random policy)'}")
    print(f"Trajectories: {args.num_traj}")
    print("=" * 60)

    # Prepare experiment
    handles, exp = prepare_run(args.exp, args.env, args.algo, tag="plot_norms")

    # Load checkpoint if provided
    if args.ckpt:
        print(f"\nLoading checkpoint: {args.ckpt}")
        handles.algo.load(args.ckpt, map_location="cpu")
        print("✓ Checkpoint loaded successfully")
    else:
        print("\nNo checkpoint provided - using random/initialized policy")

    # Get trajectory length
    eval_cfg = exp["eval"]
    traj_len = args.traj_len if args.traj_len is not None else eval_cfg["traj_len"]
    print(f"\nRolling out {args.num_traj} trajectories of length {traj_len}...")

    # Determine if actions should be deterministic
    # Default: deterministic for model-free RL (SAC, PPO), stochastic for PINN-SPI
    if args.deterministic:
        deterministic = True
    else:
        # Check algorithm type from class name
        algo_class_name = handles.algo.__class__.__name__.lower()
        deterministic = "sac" in algo_class_name or "ppo" in algo_class_name

        # Fallback: try to get from config
        if not deterministic and "algo" in exp and "name" in exp["algo"]:
            algo_name = exp["algo"]["name"].lower()
            deterministic = algo_name in ["sac", "ppo"]

    print(f"Action mode: {'deterministic' if deterministic else 'stochastic'}")

    # Rollout with actions
    result = rollout_with_actions(
        handles.env,
        handles.algo,
        args.num_traj,
        traj_len,
        clip_state=eval_cfg.get("clip_state", True),
        dynamics_mode=eval_cfg.get("dynamics_mode", "b"),
        deterministic=deterministic,
        use_env_reset=True,
        seed_offset=10000,
    )

    X = result["X"]  # [traj_len+1, num_traj, d]
    U = result["U"]  # [traj_len, num_traj, m]

    print(f"✓ Rollout complete")
    print(f"  State shape: {X.shape}")
    print(f"  Action shape: {U.shape}")

    # Convert to numpy
    X_np = X.cpu().numpy()
    U_np = U.cpu().numpy()

    # Create output directory if needed
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Plot trajectory norms
    print(f"\nPlotting trajectory norms to {args.out}...")
    plot_trajectory_norms(
        X_np,
        actions=U_np,
        out_png=args.out,
        title=args.title,
        plot_individual=args.plot_individual,
        max_individual=args.max_individual,
        state_labels=args.state_labels,
    )
    print(f"✓ Saved to {args.out}")

    # Plot state components if requested
    if args.out_components:
        print(f"\nPlotting state components to {args.out_components}...")
        out_comp_dir = os.path.dirname(args.out_components)
        if out_comp_dir and not os.path.exists(out_comp_dir):
            os.makedirs(out_comp_dir)

        plot_state_components(
            X_np,
            out_png=args.out_components,
            title=f"{args.title} - State Components",
            state_labels=args.state_labels,
            plot_individual=args.plot_individual,
            max_individual=args.max_individual,
        )
        print(f"✓ Saved to {args.out_components}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Trajectory Statistics")
    print("=" * 60)

    state_norms = np.linalg.norm(X_np, axis=2)  # [T+1, num_traj]
    action_norms = np.linalg.norm(U_np, axis=2)  # [T, num_traj]

    print(f"State norms ||x||:")
    print(f"  Mean:   {state_norms.mean():.4f}")
    print(f"  Std:    {state_norms.std():.4f}")
    print(f"  Max:    {state_norms.max():.4f}")
    print(f"  Final:  {state_norms[-1].mean():.4f} ± {state_norms[-1].std():.4f}")

    print(f"\nAction norms ||u||:")
    print(f"  Mean:   {action_norms.mean():.4f}")
    print(f"  Std:    {action_norms.std():.4f}")
    print(f"  Max:    {action_norms.max():.4f}")

    print("=" * 60)
    print("✓ Done!")


if __name__ == "__main__":
    main()

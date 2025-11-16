#!/usr/bin/env python3
"""
Training Script for PINN-PI Research Framework

This script provides a command-line interface for running experiments
comparing PINN-PI (Physics-Informed Neural Network Policy Iteration)
against SAC (Soft Actor-Critic) on continuous control tasks.

Usage:
    python scripts/train.py --exp configs/experiment/lqr_5d_pinnpi.yaml \
                           --env configs/envs/lqr5d.yaml \
                           --algo configs/algs/lqr5d_pinnpi.yaml \
                           --tag "5d_experiment"

The script handles:
- Configuration loading and validation
- Experiment setup with proper seeding
- Training loop execution (PINN-PI or SAC)
- Results saving and checkpoint management
"""

import argparse
import os
import sys

# Ensure repo root is importable when running this script directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pinn_spi.experiment.runner import prepare_run, train_loop


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train PINN-SPI or SAC on continuous control tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration file arguments
    parser.add_argument("--exp", default="configs/experiment/default.yaml",
                       help="Experiment configuration file")
    parser.add_argument("--env", default="configs/envs/lqr5d.yaml",
                       help="Environment configuration file")
    parser.add_argument("--algo", default="configs/algs/sac.yaml",
                       help="Algorithm configuration file")
    parser.add_argument("--tag", default="",
                       help="Tag for experiment run (appended to results directory)")

    args = parser.parse_args()

    # Display experiment configuration
    print(f"🚀 Starting training with:")
    print(f"   Experiment: {args.exp}")
    print(f"   Environment: {args.env}")
    print(f"   Algorithm: {args.algo}")
    print(f"   Tag: {args.tag or 'none'}")

    # Setup and run experiment
    handles, exp = prepare_run(args.exp, args.env, args.algo, tag=args.tag)
    results_dir = train_loop(handles, exp)

    print(f"✅ Training completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

"""
Evaluation orchestration for training loop.

Handles evaluation execution and result logging, abstracting away the complexity
of different evaluation modes (sequential vs vectorized).
"""

import time
import torch
from typing import Dict, Any
from ..core.rollout import evaluate_policy, evaluate_policy_sequential
from .config import EvaluationConfig, RewardScaleConfig


class Evaluator:
    """Handles policy evaluation during training."""

    def __init__(
        self,
        env,
        algo,
        eval_config: EvaluationConfig,
        reward_config: RewardScaleConfig,
    ):
        self.env = env
        self.algo = algo
        self.eval_config = eval_config
        self.reward_config = reward_config

    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation and return results.

        Returns:
            dict with keys: G (returns), X (trajectories), duration (eval time)
        """
        eval_start = time.time()

        if self.eval_config.evaluation_type == "sequential":
            results = self._evaluate_sequential()
        elif self.eval_config.evaluation_type == "vectorized":
            results = self._evaluate_vectorized()
        else:
            raise ValueError(
                f"Invalid evaluation_type: {self.eval_config.evaluation_type}. "
                "Must be 'sequential' or 'vectorized'."
            )

        eval_duration = time.time() - eval_start
        results["duration"] = eval_duration

        return results

    def _evaluate_sequential(self) -> Dict[str, Any]:
        """Run sequential evaluation (one episode at a time)."""
        params = self.eval_config.get_sequential_params()

        return evaluate_policy_sequential(
            self.env,
            self.algo,
            num_episodes=self.eval_config.num_traj,
            max_steps=self.eval_config.traj_len,
            deterministic=self.eval_config.deterministic,
            scale_reward_by_dt=self.reward_config.scale_by_dt,
            reward_scale=self.reward_config.base_scale,
            **params,
        )

    def _evaluate_vectorized(self) -> Dict[str, Any]:
        """Run vectorized evaluation (all trajectories in parallel)."""
        params = self.eval_config.to_params_dict()

        return evaluate_policy(
            self.env,
            self.algo,
            num_traj=self.eval_config.num_traj,
            traj_len=self.eval_config.traj_len,
            deterministic=self.eval_config.deterministic,
            scale_reward_by_dt=self.reward_config.scale_by_dt,
            reward_scale=self.reward_config.base_scale,
            **params,
        )

    def log_results(self, logger, results: Dict[str, Any], step: int) -> Dict[str, float]:
        """
        Log evaluation results to logger.

        Returns:
            dict with logged metrics (for printing)
        """
        G = results["G"].mean().item()
        std_ret = float(results["G"].std(unbiased=False).item()) if results["G"].numel() > 1 else 0.0

        logger.log_scalar("eval/avg_return", G, step=step)
        logger.log_scalar("eval/std_return", std_ret, step=step)
        logger.log_scalar("timing/eval_time", results["duration"], step=step)

        return {
            "avg_return": G,
            "std_return": std_ret,
            "eval_time": results["duration"],
        }

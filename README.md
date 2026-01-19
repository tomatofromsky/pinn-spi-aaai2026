# Physics-Informed Approach for Exploratory Hamilton–Jacobi–Bellman Equations via Policy Iterations

This repository implements a mesh-free, physics-informed policy iteration (PINN‑SPI) framework for solving entropy‑regularized stochastic control problems, and provides a common runner to compare against SAC and PPO baselines on LQR, Pendulum, and CartPole tasks.

The implementation follows the paper “Physics-informed approach for exploratory Hamilton–Jacobi–Bellman equations via policy iterations”. It alternates between soft policy evaluation and improvement using automatic differentiation and neural approximation, without spatial discretization.


**Method Overview**
- Soft PI: alternate between (i) soft policy evaluation by solving a linear elliptic PDE and (ii) policy improvement via a soft‑max update.
- Mesh‑free realization: embed the HJB residual into a PINN loss; use AD for gradients/Hessian traces; sample collocation points in state–action space.
- Entropy regularization: add KL‑type term with temperature λ to encourage exploration and stabilize learning.

—

**Requirements**
- Python >= 3.8
- Core deps: `torch`, `numpy`, `matplotlib`, `pyyaml`, `tqdm`
- Optional envs: `gymnasium` (for CartPole, Pendulum)
- Package/lock: `pyproject.toml`, `uv.lock` (apps/experiments should commit lock for reproducibility)

Install (choose one):
- Using uv
  - `uv sync` (CI: `uv sync --frozen`)
- Using pip
  - Core: `pip install -e .`
  - With gym envs: `pip install -e .[gym]`

**Project Structure**
- `pinn_spi/` — package (algorithms, environments, runner, core utils)
  - `algs/` — `pinn_spi` (PINN‑SPI), `sac` (continuous/discrete), `ppo` (continuous/discrete)
  - `envs/` — LQR (continuous), Pendulum (continuous), CartPole (discrete)
  - `experiment/` — training infrastructure
    - `runner.py` — main training loop orchestration
    - `config.py` — configuration dataclasses (EvaluationConfig, TrainingConfig, RewardScaleConfig)
    - `evaluation.py` — evaluation orchestration (Evaluator class)
  - `core/` — rollout (Euler–Maruyama with two dynamics modes), evaluate (sequential/vectorized), replay buffer
  - `utils/` — logging (JSONL with timing), plots
- `configs/` — YAML configs for experiments, algorithms, environments
- `scripts/` — CLI scripts: `train.py`, `eval.py`, `compare.py`
- `results/` — metrics/checkpoints/plots (git‑ignored by default)

**Quick Start**
- PINN‑SPI on LQR 5D
  - `python scripts/train.py --exp configs/experiment/lqr_5d_pinnspi.yaml --env configs/envs/lqr5d.yaml --algo configs/algs/lqr5d_pinnspi.yaml --tag lqr5d_pinnspi`
- SAC on LQR 5D
  - `python scripts/train.py --exp configs/experiment/lqr_5d_sac.yaml --env configs/envs/lqr5d.yaml --algo configs/algs/lqr5d_sac.yaml --tag lqr5d_sac`
- PPO on LQR 5D
  - `python scripts/train.py --exp configs/experiment/lqr_5d_sac.yaml --env configs/envs/lqr5d.yaml --algo configs/algs/ppo.yaml --tag lqr5d_ppo`

Notes
- Algorithm `name` in YAML should be `pinnspi`, `sac`, or `ppo` (PINN‑PI path has been removed in this branch).
- For CartPole/Pendulum, install `gymnasium`.

PPO notes
- On‑policy collection with vectorized rollout: params `rollout_steps`, `num_envs`, `gae_lambda`, `train_iters`, `minibatch_size`, `total_steps`.
- Continuous actions are tanh‑Gaussian and scaled by environment bounds automatically; discrete uses Categorical.
- Logging/evaluation/ckpt are recorded in environment steps for PPO/SAC; iterations for PINN‑SPI.

**Evaluation**
- Evaluate a checkpoint:
  - `python scripts/eval.py --exp <exp.yaml> --env <env.yaml> --algo <algo.yaml> --ckpt results/<run>/final_model.pth --num_traj 100`
- During training, periodic eval logs to `results/<run>/metrics.jsonl` and plots are auto‑generated at checkpoints.

**Comparison**
- Compare two runs (time‑based x‑axis for fair comparison):
  - `python scripts/compare.py --run1 results/<run1> --run2 results/<run2> --key "eval/avg_return" --x_axis training_time --out comparison.png`

**Configuration Highlights**
- Discounting: `gamma = exp(-rho * dt)` computed from experiment `rho` and `dt`.
- PINN‑SPI training (typical): outer_iters, per‑phase iters, resample periods, collocation sizes `N_x`, `N_u`, temperature `lam`.
- PPO training (typical): `hidden`, `lr_actor`, `lr_critic`, `clip_ratio`, `entropy_coef`, `vf_coef`, `max_grad_norm`,
  `rollout_steps`, `num_envs`, `gae_lambda`, `train_iters`, `minibatch_size`, `total_steps`.
- LQR matrices are loaded from `data/`; bounds and noise set in `configs/envs/*.yaml`.
- **Evaluation modes** (in experiment config `eval` section):
  - `evaluation_type`: "sequential" (proper episodes with termination) or "vectorized" (parallel fixed-length trajectories)
  - `dynamics_mode`: "b" (PyTorch SDE, default) or "gym" (Gymnasium physics, CartPole only)
- **Environment-specific**:
  - CartPole: supports both dynamics modes ("gym" and "b")
  - Pendulum/LQR: only support "b" mode (pure PyTorch implementations)

**Reproducibility**
- Commit both `pyproject.toml` and `uv.lock` for consistent resolution across machines.
- In CI, prefer `uv sync --frozen`.


**License**
MIT License - see LICENSE file for details.

**Citation**
If you use this code in your research, please cite:

@software{pinn_spi,
  title = {Physics-Informed Approach for Exploratory Hamilton–Jacobi–Bellman Equations via Policy Iterations},
  author = {Kim, Yeongjong and Kim, Yeoneung and Kim, Minseok and Cho, Namkyeong},
  year = {2025},
  url = {[https://github.com/...](https://github.com/tomatofromsky/pinn-spi-aaai2026)}
}

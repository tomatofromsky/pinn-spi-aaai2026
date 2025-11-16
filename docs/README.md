# PINN-SPI Documentation

This directory contains comprehensive documentation for the PINN-SPI research framework comparing PINN-SPI, SAC, and PPO algorithms.

## Documentation Structure

### Core Documentation
- **[metrics.md](metrics.md)** - Detailed explanation of all evaluation and training metrics
- **[scripts.md](scripts.md)** - Comprehensive usage guide for train.py, eval.py, and compare.py
- **[algorithms.md](algorithms.md)** - Algorithm-specific implementation details *(planned)*
- **[experiments.md](experiments.md)** - Experimental setup and configuration guide *(planned)*

### Quick Reference

#### Key Metrics Summary

| Metric | Algorithm | Description | Interpretation |
|--------|-----------|-------------|----------------|
| `eval/avg_return` | All | Average discounted return across trajectories | Higher = better (less negative for LQR) |
| `pde_mse` | PINN-SPI | HJB PDE residual MSE during training | Lower = better PDE satisfaction |
| `actor_loss` | SAC/PPO | Policy gradient loss | Lower = better policy |
| `critic_loss` | SAC/PPO | Value function temporal difference error | Lower = better value estimation |

#### Timing Metrics Summary

| Metric | Description | Usage |
|--------|-------------|-------|
| `timing/step_training_time` | Training time per step (excludes evaluation) | Step-by-step efficiency analysis |
| `timing/total_training_time` | Cumulative training time (excludes evaluation) | Fair algorithm comparison |
| `timing/wall_time` | Total elapsed time (includes everything) | Real-world performance assessment |
| `timing/eval_time` | Evaluation time per step | Evaluation overhead analysis |

#### Evaluation Frequency Setup

For fair algorithm comparison:

| Experiment | PINN-SPI | SAC/PPO | Updates per Evaluation |
|------------|----------|---------|----------------------|
| 5D LQR | `every_steps: 1` | `every_steps: 100` | 100 updates |
| 10D LQR | `every_steps: 1` | `every_steps: 200` | 200 updates |
| 20D LQR | `every_steps: 1` | `every_steps: 300` | 300 updates |

#### Configuration Files

**Algorithm Configs**:
- `configs/aaai2026/sigma*/lqr{5d,10d,20d}_spi.yaml` - PINN-SPI hyperparameters
- `configs/aaai2026/sigma*/lqr{5d,10d,20d}_sac.yaml` - SAC hyperparameters
- `configs/aaai2026/sigma*/cartpole_sac.yaml`, `pendulum_sac.yaml` - Environment-specific SAC configs

**Experiment Configs**:
- `configs/aaai2026/sigma*/lqr{5d,10d,20d}_expt.yaml` - LQR experiments
- `configs/aaai2026/sigma*/cartpole_expt.yaml` - CartPole experiments (with `_dynamics_b` variants)
- `configs/aaai2026/sigma*/pendulum_expt.yaml` - Pendulum experiments

**Environment Configs**:
- `configs/aaai2026/sigma*/lqr{5d,10d,20d}.yaml` - LQR system specifications
- `configs/aaai2026/sigma*/cartpole.yaml` - CartPole specifications
- `configs/aaai2026/sigma*/pendulum.yaml` - Pendulum specifications

**Evaluation Configuration Options** (in experiment config `eval` section):
- `evaluation_type`: "sequential" or "vectorized"
  - **sequential**: Proper episodes with termination (for CartPole/Pendulum)
  - **vectorized**: Parallel fixed-length trajectories (for LQR)
- `dynamics_mode`: "b" or "gym"
  - **"b"**: PyTorch SDE physics (default, all environments)
  - **"gym"**: Gymnasium physics (CartPole only)

**Code Architecture** (refactored for maintainability):
- `pinn_spi/experiment/config.py`: Configuration dataclasses
- `pinn_spi/experiment/evaluation.py`: Evaluation orchestration
- `pinn_spi/experiment/runner.py`: Training loop (uses config objects)

### Usage Examples

#### Running Experiments
```bash
# PINN-SPI on 5D LQR
python scripts/train.py --exp configs/aaai2026/sigma0p1/lqr5d_expt.yaml \
                       --env configs/aaai2026/sigma0p1/lqr5d.yaml \
                       --algo configs/aaai2026/sigma0p1/lqr5d_spi.yaml \
                       --tag "lqr5d_spi"

# SAC on 5D LQR
python scripts/train.py --exp configs/aaai2026/sigma0p1/lqr5d_expt.yaml \
                       --env configs/aaai2026/sigma0p1/lqr5d.yaml \
                       --algo configs/aaai2026/sigma0p1/lqr5d_sac.yaml \
                       --tag "lqr5d_sac"

# SAC on Pendulum (sequential evaluation, PyTorch SDE physics)
python scripts/train.py --exp configs/aaai2026/sigma0p1/pendulum_expt.yaml \
                       --env configs/aaai2026/sigma0p1/pendulum.yaml \
                       --algo configs/aaai2026/sigma0p1/pendulum_sac.yaml \
                       --tag "pendulum_sac"
```

#### Analyzing Results
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics from experiment
with open('results/{timestamp}_experiment/metrics.jsonl', 'r') as f:
    metrics = [json.loads(line) for line in f]

df = pd.DataFrame(metrics)

# Sample metric format (now includes timestamps and timing info):
# {
#   "step": 100,
#   "name": "eval/avg_return",
#   "value": -558.525,
#   "timestamp": 1758964394.87,
#   "wall_time": 23.4,
#   "training_time": 12.1
# }

# Performance analysis
avg_returns = df[df['name'] == 'eval/avg_return']
training_times = df[df['name'] == 'timing/total_training_time']

# Plot performance vs training time (fair comparison)
plt.figure(figsize=(10, 6))
plt.plot(training_times['value'], avg_returns['value'])
plt.xlabel('Training Time (seconds)')
plt.ylabel('Average Return')
plt.title('Performance vs Computational Budget')
```

### Mathematical Background

#### LQR Dynamics
```
dx = (Ax + Bu)dt + σdW
```

#### Cost Function
```
J = E[∫₀^T e^(-ρt) (x'Qx + u'Ru) dt]
```

#### HJB PDE (PINN-SPI)
```
ρV(x) - (1/2)σ²ΔV(x) - E_u~π[b(x,u)·∇V(x) + r(x,u)] - λH(π(·|x)) = 0
```
Where H(π) is the entropy term for exploration regularization.

### Contributing

When adding new metrics or algorithms:
1. Update `metrics.md` with detailed explanations
2. Add configuration examples
3. Include interpretation guidelines
4. Provide mathematical formulations where applicable

### References

See `metrics.md` for comprehensive references to relevant literature and theoretical foundations.
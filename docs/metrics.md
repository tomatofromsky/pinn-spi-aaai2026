# Metrics Documentation

This document provides detailed explanations of all metrics used in the PINN-PI research framework for evaluating and comparing reinforcement learning algorithms.

## Table of Contents

1. [Core Evaluation Metrics](#core-evaluation-metrics)
2. [Training Metrics](#training-metrics)
3. [Metric Computation Details](#metric-computation-details)
4. [Interpretation Guidelines](#interpretation-guidelines)
5. [Comparison Between Algorithms](#comparison-between-algorithms)

---

## Core Evaluation Metrics

### `eval/avg_return`

**Definition**: Average discounted cumulative reward over multiple policy evaluation trajectories.

**Mathematical Formula**:
```
avg_return = (1/N) * Σ[i=1 to N] G_i

where G_i = Σ[t=0 to T-1] γ^t * r(x_t^i, u_t^i)
```

**Components**:
- `N`: Number of evaluation trajectories (`num_traj` in config)
- `T`: Trajectory length (`traj_len` in config)
- `γ`: Discount factor (`exp(-ρ * dt)` where `ρ` is from config)
- `r(x_t, u_t)`: Instantaneous reward at time step t
- `x_t^i, u_t^i`: State and action at time t in trajectory i

**Computation Process**:
1. **Trajectory Rollout**: Generate N independent trajectories starting from x₀
2. **Reward Collection**: Compute instantaneous rewards at each time step
3. **Discounting**: Apply discount factor γᵗ to rewards
4. **Accumulation**: Sum discounted rewards per trajectory to get returns G_i
5. **Averaging**: Compute mean across all trajectories

**Code Implementation**:
```python
# In evaluate_policy function
G = torch.zeros(num_traj, device=algo.device)
for t in range(traj_len):
    u = algo.act(X[t], deterministic=False)
    r = env.r(X[t], u)
    G += (r if gamma is None else (gamma**t) * r)

# In runner
avg_return = out["G"].mean().item()
```

**Typical Values**:
- **LQR Systems**: Usually negative (cost-to-go), range: [-1000, 0]
- **Higher values = better performance** (less negative cost)
- **Convergence**: Should stabilize as learning progresses

---

## Training Metrics

### `pde_mse` (PINN-PI only)

**Definition**: Mean squared error of the HJB PDE residual during value function training.

**Mathematical Formula**:
```
pde_mse = (1/B) * Σ[i=1 to B] ||residual_i||²

where residual_i = ρV(x_i) - (1/2)σ²ΔV(x_i) - b(x_i,u*_i)·∇V(x_i) - r(x_i,u*_i)
```

**Components**:
- `ρ`: Discount rate (exponent parameter)
- `V(x)`: Value function approximation
- `ΔV(x)`: Laplacian of value function (trace of Hessian)
- `∇V(x)`: Gradient of value function
- `b(x,u)`: Drift dynamics
- `u*`: Greedy action from current policy
- `r(x,u)`: Instantaneous reward

**Interpretation**:
- **Lower values = better**: PDE is more accurately satisfied
- **Training progress**: Should decrease during inner epochs
- **Convergence indicator**: Stabilization indicates policy convergence

---

## Timing Metrics

### `timing/step_training_time`

**Definition**: Time spent on training computations for the current step/iteration, excluding evaluation.

**Components**:
- **PINN-PI**: Time for one full `inner_steps` epochs of value function training
- **SAC**: Time for environment interaction + gradient updates (when applicable)
- **Excludes**: Policy evaluation time for fair algorithm comparison

### `timing/total_training_time`

**Definition**: Cumulative training time since experiment start, excluding all evaluation periods.

**Usage**:
- **Fair comparison**: Compare algorithms based on actual learning computation time
- **Efficiency analysis**: Training performance vs. computational budget
- **Scalability**: How training time scales with problem complexity

### `timing/wall_time`

**Definition**: Total elapsed time since experiment start (includes everything).

**Components**:
- Training time
- Evaluation time
- Logging overhead
- Checkpoint saving

### `timing/eval_time`

**Definition**: Time spent on policy evaluation for the current step.

**Components**:
- Trajectory generation (`num_traj` × `traj_len` rollouts)
- Reward computation and discounting
- Statistical aggregation

### Timestamp Information

**`timestamp`**: Unix timestamp when metric was logged
**`wall_time`**: Seconds since experiment start
**`training_time`**: Cumulative training seconds (auto-tracked)

---

### Algorithm-Specific Training Metrics

#### SAC Training Metrics

| Metric | Description | Formula | Interpretation |
|--------|-------------|---------|----------------|
| `actor_loss` | Policy gradient loss | `-E[Q(s,π(s)) - α*log π(a\|s)]` | Lower = better policy |
| `critic_loss` | Q-function TD error | `E[(Q(s,a) - target)²]` | Lower = better value estimation |
| `entropy` | Policy entropy | `-E[log π(a\|s)]` | Higher = more exploration |

---

## Metric Computation Details

### Discount Factor Calculation

**Formula**: `γ = exp(-ρ * dt)`

**Parameters**:
- `ρ`: Continuous discount rate (from environment config)
- `dt`: Time step size (from evaluation config)

**Example** (LQR 5D):
```
ρ = 0.5 (from config)
dt = 0.02 (from config)
γ = exp(-0.5 * 0.02) = exp(-0.01) ≈ 0.99
```

### Initial State Configuration

**Options**:
- `x0_weight = 0.0`: Start from origin (zero state)
- `x0_weight = 0.1`: Start from `0.1 * ones(d)`
- Custom initial states can be provided

**Algorithm Differences**:
- **PINN-PI**: Typically uses `x0_weight = 0.0`
- **SAC**: Often uses `x0_weight = 0.1` for exploration

### Trajectory Generation

**Process**:
1. **Initialization**: Set x₀ according to `x0_weight`
2. **Policy Execution**: Sample actions from current policy
3. **Dynamics**: Euler-Maruyama integration of SDE
4. **Reward Computation**: Calculate instantaneous rewards
5. **State Clipping**: Enforce domain bounds if specified

---

## Interpretation Guidelines

### Learning Curves

#### Healthy Learning Patterns

**PINN-PI**:
- `eval/avg_return`: Gradual improvement over policy iterations
- `pde_mse`: Decreasing trend within each outer iteration
- **Convergence**: Both metrics stabilize after sufficient iterations

**SAC**:
- `eval/avg_return`: Noisy but improving trend over environment steps
- Training losses: Fluctuating but generally decreasing
- **Convergence**: Performance plateaus with continued exploration

#### Warning Signs

- **Divergence**: `avg_return` decreasing significantly
- **Instability**: Extreme oscillations in metrics
- **Plateau**: No improvement for extended periods
- **Numerical Issues**: NaN or infinite values

### Performance Comparison

#### Fair Comparison Setup

**Evaluation Frequency**:
- **PINN-PI**: Every `outer_iters` (after `inner_steps` value updates)
- **SAC**: Every `inner_steps` environment steps
- **Result**: Both evaluated after equivalent amounts of learning

**Example** (5D LQR):
```
PINN-PI: every_steps=1, inner_steps=100 → eval every 100 updates
SAC: every_steps=100 → eval every 100 updates
```

#### Statistical Significance

**Trajectory Count**: Use sufficient `num_traj` (typically 30) for reliable estimates
**Multiple Seeds**: Run experiments with different random seeds
**Confidence Intervals**: Report mean ± standard deviation across runs

---

## Comparison Between Algorithms

### Expected Performance Characteristics

#### PINN-PI
- **Advantages**: Sample efficient, principled optimization, theoretical guarantees
- **Learning Pattern**: Smooth convergence, fewer evaluations needed
- **Computational**: Higher cost per iteration, lower total wall-clock time

#### SAC
- **Advantages**: Model-free, robust to dynamics, wide applicability
- **Learning Pattern**: Noisy exploration, many evaluations needed
- **Computational**: Lower cost per step, higher total wall-clock time

### Experimental Validation

#### Convergence Analysis
- Plot `eval/avg_return` vs. update count for both algorithms
- Compare final performance after equivalent computational budget
- Analyze sample efficiency (performance vs. environment interactions)

#### Robustness Testing
- Test with different initial conditions
- Vary hyperparameters within reasonable ranges
- Evaluate on multiple random seeds

---

## Implementation Notes

### Metric Logging

**File Format**: JSONL (JSON Lines) for streaming metrics
**Location**: `{results_dir}/metrics.jsonl`
**Structure**:
```json
{"metric": "eval/avg_return", "value": -558.525, "step": 100}
{"metric": "pde_mse", "value": 1.23e-4, "step": 100}
```

### Visualization

**Recommended Tools**:
- **TensorBoard**: For real-time monitoring
- **Matplotlib**: For publication-quality plots
- **Pandas**: For metric analysis and statistics

**Key Plots**:
1. Learning curves (avg_return vs. updates)
2. Training metrics vs. time
3. Comparison across algorithms/configurations
4. Statistical analysis (mean ± std across seeds)

---

## References

1. **HJB PDE Theory**: Evans, L.C. "Partial Differential Equations"
2. **SAC Algorithm**: Haarnoja, T. et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
3. **LQR Control**: Anderson, B.D.O. & Moore, J.B. "Optimal Control: Linear Quadratic Methods"
4. **Stochastic Calculus**: Øksendal, B. "Stochastic Differential Equations"
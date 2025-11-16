# Script Usage Documentation

This document provides detailed usage instructions for the main scripts in the PINN-PI research framework.

## Table of Contents

1. [Training Scripts](#training-scripts)
2. [Evaluation Scripts](#evaluation-scripts)
3. [Comparison Scripts](#comparison-scripts)
4. [Common Workflows](#common-workflows)
5. [Configuration Files](#configuration-files)
6. [Troubleshooting](#troubleshooting)

---

## Training Scripts

### `scripts/train.py`

**Purpose**: Train PINN-PI or SAC algorithms on continuous control tasks with comprehensive metric logging.

#### Basic Usage

```bash
python scripts/train.py --exp <experiment_config> --env <env_config> --algo <algo_config> [--tag <tag>]
```

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--exp` | Experiment configuration file | `configs/experiment/lqr_5d_pinnpi.yaml` |
| `--env` | Environment configuration file | `configs/envs/lqr5d.yaml` |
| `--algo` | Algorithm configuration file | `configs/algs/pinnpi_lqr_5d.yaml` |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tag` | `""` | Tag appended to results directory name |

#### Example Commands

**PINN-PI on 5D LQR:**
```bash
python scripts/train.py --exp configs/experiment/lqr_5d_pinnpi.yaml \
                       --env configs/envs/lqr5d.yaml \
                       --algo configs/algs/pinnpi_lqr_5d.yaml \
                       --tag "5d_pinnpi_run1"
```

**SAC on 10D LQR:**
```bash
python scripts/train.py --exp configs/experiment/lqr_10d_sac.yaml \
                       --env configs/envs/lqr10d.yaml \
                       --algo configs/algs/sac_lqr_10d.yaml \
                       --tag "10d_sac_baseline"
```

#### Output Structure

Training creates a timestamped results directory:
```
results/
├── 20241027-143052_5d_pinnpi_run1/
│   ├── metrics.jsonl          # Time-series metrics
│   ├── config_manifest.json   # Configuration file paths
│   ├── checkpoint_*.pth       # Periodic checkpoints (if enabled)
│   └── final_model.pth        # Final trained model
```

#### Key Features

- **Automatic Device Detection**: Uses CUDA if available, falls back to CPU
- **Comprehensive Logging**: Tracks performance, training time, wall time, and algorithm-specific metrics
- **Checkpoint Management**: Saves periodic and final model checkpoints
- **Configuration Tracking**: Records all configuration files used
- **Timing Analysis**: Separates training time from evaluation time for fair comparison

---

## Evaluation Scripts

### `scripts/eval.py`

**Purpose**: Evaluate trained models on specified tasks with detailed performance statistics.

#### Basic Usage

```bash
python scripts/eval.py --exp <experiment_config> --env <env_config> --algo <algo_config> [--ckpt <checkpoint>] [--num_traj <N>]
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--exp` | Yes | `configs/experiment/default.yaml` | Experiment configuration |
| `--env` | Yes | `configs/envs/lqr5d.yaml` | Environment configuration |
| `--algo` | Yes | `configs/algs/sac.yaml` | Algorithm configuration |
| `--ckpt` | No | `None` | Path to model checkpoint |
| `--num_traj` | No | `100` | Number of evaluation trajectories |

#### Example Commands

**Evaluate final model:**
```bash
python scripts/eval.py --exp configs/experiment/lqr_5d_pinnpi.yaml \
                      --env configs/envs/lqr5d.yaml \
                      --algo configs/algs/pinnpi_lqr_5d.yaml \
                      --ckpt results/20241027-143052_5d_pinnpi_run1/final_model.pth \
                      --num_traj 200
```

**Evaluate specific checkpoint:**
```bash
python scripts/eval.py --exp configs/experiment/lqr_10d_sac.yaml \
                      --env configs/envs/lqr10d.yaml \
                      --algo configs/algs/sac_lqr_10d.yaml \
                      --ckpt results/20241027-150000_10d_sac/checkpoint_5000.pth
```

**Evaluate without checkpoint (random policy):**
```bash
python scripts/eval.py --exp configs/experiment/lqr_5d_pinnpi.yaml \
                      --env configs/envs/lqr5d.yaml \
                      --algo configs/algs/pinnpi_lqr_5d.yaml \
                      --num_traj 50
```

#### Output Format

```
Evaluating with:
  Environment: configs/envs/lqr5d.yaml
  Algorithm: configs/algs/pinnpi_lqr_5d.yaml
  Checkpoint: results/20241027-143052_5d_pinnpi_run1/final_model.pth

Loading checkpoint: results/20241027-143052_5d_pinnpi_run1/final_model.pth

Evaluation Results:
  Mean return: -234.567
  Std return: 12.345
  Min return: -267.890
  Max return: -201.234
```

---

## Comparison Scripts

### `scripts/compare.py`

**Purpose**: Compare performance between two experiment runs with visualization and statistics.

#### Basic Usage

```bash
python scripts/compare.py --run1 <results_dir1> --run2 <results_dir2> [options]
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--run1` | Yes | - | First experiment results directory |
| `--run2` | Yes | - | Second experiment results directory |
| `--key` | No | `eval/avg_return` | Metric to compare |
| `--out` | No | `comparison.png` | Output plot filename |
| `--label1` | No | Directory name | Label for first run |
| `--label2` | No | Directory name | Label for second run |

#### Example Commands

**Basic comparison:**
```bash
python scripts/compare.py --run1 results/20241027-143052_5d_pinnpi_run1 \
                         --run2 results/20241027-150000_5d_sac_run1 \
                         --out pinnpi_vs_sac_5d.png
```

**Custom labels and metric:**
```bash
python scripts/compare.py --run1 results/20241027-143052_5d_pinnpi_run1 \
                         --run2 results/20241027-150000_5d_sac_run1 \
                         --key "pde_mse" \
                         --label1 "PINN-PI" \
                         --label2 "SAC" \
                         --out pde_convergence.png
```

**Training time comparison:**
```bash
python scripts/compare.py --run1 results/20241027-143052_5d_pinnpi_run1 \
                         --run2 results/20241027-150000_5d_sac_run1 \
                         --key "timing/total_training_time" \
                         --out training_time_comparison.png
```

#### Output

The script produces:
1. **Comparison plot**: Visual comparison of the specified metric over time
2. **Statistical summary**: Final value, mean, and maximum for each run

```
Comparing runs:
  Run 1: results/20241027-143052_5d_pinnpi_run1 -> PINN-PI
  Run 2: results/20241027-150000_5d_sac_run1 -> SAC
  Metric: eval/avg_return

Comparison plot saved to: pinnpi_vs_sac_5d.png

PINN-PI:
  Final value: -234.567
  Mean value: -345.678
  Max value: -201.234

SAC:
  Final value: -256.789
  Mean value: -389.012
  Max value: -234.567
```

---

## Common Workflows

### 1. Full Algorithm Comparison

```bash
# Step 1: Train PINN-PI
python scripts/train.py --exp configs/experiment/lqr_5d_pinnpi.yaml \
                       --env configs/envs/lqr5d.yaml \
                       --algo configs/algs/pinnpi_lqr_5d.yaml \
                       --tag "comparison_pinnpi"

# Step 2: Train SAC
python scripts/train.py --exp configs/experiment/lqr_5d_sac.yaml \
                       --env configs/envs/lqr5d.yaml \
                       --algo configs/algs/sac_lqr_5d.yaml \
                       --tag "comparison_sac"

# Step 3: Compare results
python scripts/compare.py --run1 results/*_comparison_pinnpi \
                         --run2 results/*_comparison_sac \
                         --label1 "PINN-PI" \
                         --label2 "SAC" \
                         --out algorithm_comparison.png
```

### 2. Hyperparameter Sweep

```bash
# Train with different configurations
for lr in 0.001 0.0001; do
    python scripts/train.py --exp configs/experiment/lqr_5d_pinnpi.yaml \
                           --env configs/envs/lqr5d.yaml \
                           --algo configs/algs/pinnpi_lqr_5d.yaml \
                           --tag "lr_${lr}"
done

# Compare results
python scripts/compare.py --run1 results/*_lr_0.001 \
                         --run2 results/*_lr_0.0001 \
                         --label1 "LR=0.001" \
                         --label2 "LR=0.0001"
```

### 3. Model Evaluation Pipeline

```bash
# Train model
python scripts/train.py --exp configs/experiment/lqr_10d_pinnpi.yaml \
                       --env configs/envs/lqr10d.yaml \
                       --algo configs/algs/pinnpi_lqr_10d.yaml \
                       --tag "final_model"

# Evaluate at different checkpoints
RESULTS_DIR="results/*_final_model"
for ckpt in checkpoint_1000.pth checkpoint_5000.pth final_model.pth; do
    echo "Evaluating $ckpt"
    python scripts/eval.py --exp configs/experiment/lqr_10d_pinnpi.yaml \
                          --env configs/envs/lqr10d.yaml \
                          --algo configs/algs/pinnpi_lqr_10d.yaml \
                          --ckpt $RESULTS_DIR/$ckpt \
                          --num_traj 500
done
```

### 4. Advanced Time-Based Comparison

```python
# Create comprehensive time-based plots
from pinn_pi.utils.plots import create_comprehensive_plots

experiment_paths = {
    "pinnpi": "results/20241027-143052_5d_pinnpi_run1/metrics.jsonl",
    "sac": "results/20241027-150000_5d_sac_run1/metrics.jsonl"
}

create_comprehensive_plots(
    experiment_paths,
    "comparison_plots",
    metrics=["eval/avg_return", "timing/total_training_time"]
)
```

---

## Configuration Files

### Configuration Structure

The framework uses three types of configuration files:

#### 1. Experiment Configuration (`configs/experiment/`)
Controls training parameters, evaluation settings, and logging.

**Key parameters:**
- `total_steps` / `outer_iters`: Training duration
- `eval.every_steps`: Evaluation frequency
- `eval.num_traj`: Number of evaluation trajectories
- `eval.traj_len`: Trajectory length
- `seed`: Random seed for reproducibility

#### 2. Environment Configuration (`configs/envs/`)
Defines the control system and task parameters.

**For LQR systems:**
- System matrices (A, B, Q, R)
- Noise parameters (σ)
- Discount rate (ρ)
- State/control bounds

#### 3. Algorithm Configuration (`configs/algs/`)
Specifies algorithm hyperparameters.

**PINN-PI parameters:**
- Network architecture
- Learning rates
- Inner/outer iteration counts
- Collocation points

**SAC parameters:**
- Network architectures
- Learning rates
- Replay buffer size
- Exploration parameters

### Pre-configured Experiments

| Task | PINN-PI Config | SAC Config | Environment |
|------|----------------|------------|-------------|
| 5D LQR | `configs/experiment/lqr_5d_pinnpi.yaml` | `configs/experiment/lqr_5d_sac.yaml` | `configs/envs/lqr5d.yaml` |
| 10D LQR | `configs/experiment/lqr_10d_pinnpi.yaml` | `configs/experiment/lqr_10d_sac.yaml` | `configs/envs/lqr10d.yaml` |
| 20D LQR | `configs/experiment/lqr_20d_pinnpi.yaml` | `configs/experiment/lqr_20d_sac.yaml` | `configs/envs/lqr20d.yaml` |

---

## Troubleshooting

### Common Issues

#### 1. Configuration File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/...'
```
**Solution**: Ensure you're running scripts from the project root directory and configuration files exist.

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch size in algorithm configuration
- Use `--device cpu` if available
- Reduce network sizes

#### 3. No Data for Metric
```
Warning: No data found for key 'eval/avg_return' in metrics.jsonl
```
**Solution**:
- Check metric name spelling
- Ensure evaluation was performed (check `every_steps` setting)
- Verify experiment completed successfully

#### 4. Model Loading Error
```
RuntimeError: Error(s) in loading state_dict
```
**Solution**:
- Ensure algorithm configuration matches the saved model
- Check that checkpoint file exists and is not corrupted
- Verify device compatibility (CPU vs GPU)

### Performance Tips

1. **Fair Algorithm Comparison**: Use training time metrics rather than step count
2. **Statistical Significance**: Run multiple seeds and compare distributions
3. **Memory Management**: Use checkpointing for long runs
4. **Efficient Evaluation**: Adjust `num_traj` based on required precision

### Debug Mode

Enable verbose output in experiment configurations:
```yaml
verbose: true
```

This provides detailed information about:
- Configuration loading
- Device selection
- Training progress
- Timing breakdown

---

## Advanced Usage

### Custom Metrics Analysis

```python
from pinn_pi.utils.plots import load_series
import matplotlib.pyplot as plt

# Load specific metrics
x, y = load_series("results/experiment/metrics.jsonl", "pde_mse", x_axis="training_time")

# Custom analysis
convergence_point = x[y < 1e-4][0] if any(y < 1e-4) else None
print(f"PDE converged after {convergence_point:.1f}s training")
```

### Batch Processing

```bash
# Process multiple experiments
for exp in configs/experiment/lqr_*d_pinnpi.yaml; do
    env=$(echo $exp | sed 's/experiment/envs/' | sed 's/pinnpi//')
    algo=$(echo $exp | sed 's/experiment/algs/')

    python scripts/train.py --exp $exp --env $env --algo $algo
done
```

For more detailed information about metrics and theoretical background, see [metrics.md](metrics.md).
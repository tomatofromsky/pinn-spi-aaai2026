import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

def load_series(jsonl_path: str, key: str, x_axis: str = "step") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time series data from JSONL log file

    Args:
        jsonl_path: Path to JSONL metrics file
        key: Metric name to extract (e.g., 'eval/avg_return')
        x_axis: X-axis variable ('step', 'training_time', 'wall_time', 'timestamp')
    """
    xs, ys = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["name"] == key:
                # Use the specified x-axis variable
                if x_axis == "step":
                    x_val = record["step"]
                elif x_axis == "training_time":
                    x_val = record.get("training_time", 0)
                elif x_axis == "wall_time":
                    x_val = record.get("wall_time", 0)
                elif x_axis == "timestamp":
                    x_val = record.get("timestamp", 0)
                else:
                    raise ValueError(f"Unknown x_axis: {x_axis}")

                xs.append(x_val)
                ys.append(record["value"])
    return np.array(xs), np.array(ys)

def load_series_legacy(jsonl_path: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function for backward compatibility"""
    return load_series(jsonl_path, key, x_axis="step")

def plot_learning_curve(jsonl_path: str, key: str, out_png: str, title: str = None,
                       x_axis: str = "step", std_key: str = None):
    """
    Plot learning curve from JSONL log

    Args:
        jsonl_path: Path to metrics file
        key: Metric to plot (e.g., 'eval/avg_return')
        out_png: Output image path
        title: Plot title
        x_axis: X-axis variable ('step', 'training_time', 'wall_time')
    """
    x, y = load_series(jsonl_path, key, x_axis)
    if len(x) == 0:
        print(f"Warning: No data found for key '{key}' in {jsonl_path}")
        return

    std = None
    if std_key:
        xs_std, ys_std = load_series(jsonl_path, std_key, x_axis)
        if len(xs_std) > 0:
            std_map = {sx: sy for sx, sy in zip(xs_std, ys_std)}
            std = np.array([std_map.get(xi, np.nan) for xi in x])
            mask = ~np.isnan(std)
            if mask.any():
                x, y, std = x[mask], y[mask], std[mask]
            else:
                std = None

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=2)
    if std is not None and len(std) == len(x):
        upper = y + std
        lower = y - std
        plt.fill_between(x, lower, upper, color='C0', alpha=0.2)

    # Set appropriate x-axis label
    x_labels = {
        "step": "Step/Iteration",
        "training_time": "Training Time (seconds)",
        "wall_time": "Wall Time (seconds)",
        "timestamp": "Time"
    }
    plt.xlabel(x_labels.get(x_axis, x_axis))
    plt.ylabel(key)

    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_comparison(paths_and_labels: List[Tuple[str, str]], key: str, out_png: str,
                   x_axis: str = "step", title: str = None):
    """
    Plot comparison of multiple runs

    Args:
        paths_and_labels: List of (jsonl_path, label) tuples
        key: Metric to plot
        out_png: Output image path
        x_axis: X-axis variable ('step', 'training_time', 'wall_time')
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    for jsonl_path, label in paths_and_labels:
        x, y = load_series(jsonl_path, key, x_axis)
        if len(x) > 0:
            plt.plot(x, y, linewidth=2, label=label, marker='o', markersize=3)

    # Set appropriate x-axis label
    x_labels = {
        "step": "Step/Iteration",
        "training_time": "Training Time (seconds)",
        "wall_time": "Wall Time (seconds)",
        "timestamp": "Time"
    }
    plt.xlabel(x_labels.get(x_axis, x_axis))
    plt.ylabel(key)

    if title:
        plt.title(title)
    else:
        plt.title(f"{key} Comparison")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_algorithm_comparison(pinnpi_path: str, sac_path: str, metric: str,
                            out_png: str, x_axis: str = "training_time"):
    """
    Compare PINN-PI vs SAC performance with time-based x-axis

    Args:
        pinnpi_path: Path to PINN-PI metrics file
        sac_path: Path to SAC metrics file
        metric: Metric to compare (e.g., 'eval/avg_return')
        out_png: Output image path
        x_axis: Time variable for fair comparison ('training_time', 'wall_time')
    """
    plt.figure(figsize=(10, 6))

    # Load data for both algorithms
    x_pinnpi, y_pinnpi = load_series(pinnpi_path, metric, x_axis)
    x_sac, y_sac = load_series(sac_path, metric, x_axis)

    if len(x_pinnpi) > 0:
        plt.plot(x_pinnpi, y_pinnpi, linewidth=2, label='PINN-PI', marker='o', markersize=4)
    if len(x_sac) > 0:
        plt.plot(x_sac, y_sac, linewidth=2, label='SAC', marker='s', markersize=4)

    # Formatting
    x_labels = {
        "training_time": "Training Time (seconds)",
        "wall_time": "Wall Time (seconds)",
        "step": "Step/Iteration"
    }
    plt.xlabel(x_labels.get(x_axis, x_axis))
    plt.ylabel(metric)
    plt.title(f"{metric} - Algorithm Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_efficiency_analysis(paths_and_labels: List[Tuple[str, str]], metric: str,
                           out_png: str, budget_limit: Optional[float] = None):
    """
    Analyze algorithm efficiency with computational budget constraints

    Args:
        paths_and_labels: List of (jsonl_path, label) tuples
        metric: Performance metric to analyze
        out_png: Output image path
        budget_limit: Maximum training time to consider (seconds)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for jsonl_path, label in paths_and_labels:
        # Training time efficiency
        x_train, y_train = load_series(jsonl_path, metric, "training_time")
        if len(x_train) > 0:
            if budget_limit:
                mask = x_train <= budget_limit
                x_train, y_train = x_train[mask], y_train[mask]
            ax1.plot(x_train, y_train, linewidth=2, label=label, marker='o', markersize=3)

        # Wall time comparison
        x_wall, y_wall = load_series(jsonl_path, metric, "wall_time")
        if len(x_wall) > 0:
            if budget_limit:
                mask = x_wall <= budget_limit * 2  # Allow more wall time
                x_wall, y_wall = x_wall[mask], y_wall[mask]
            ax2.plot(x_wall, y_wall, linewidth=2, label=label, marker='s', markersize=3)

    # Format subplots
    ax1.set_xlabel("Training Time (seconds)")
    ax1.set_ylabel(metric)
    ax1.set_title("Performance vs Training Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Wall Time (seconds)")
    ax2.set_ylabel(metric)
    ax2.set_title("Performance vs Wall Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def create_comprehensive_plots(experiment_paths: dict, output_dir: str,
                             metrics: List[str] = None):
    """
    Create comprehensive comparison plots for algorithm analysis

    Args:
        experiment_paths: Dict with 'pinnpi' and 'sac' keys pointing to results
        output_dir: Directory to save plots
        metrics: List of metrics to plot (default: common evaluation metrics)
    """
    if metrics is None:
        metrics = ["eval/avg_return"]

    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        # Individual algorithm comparison
        plot_algorithm_comparison(
            experiment_paths["pinnpi"],
            experiment_paths["sac"],
            metric,
            os.path.join(output_dir, f"{metric.replace('/', '_')}_comparison.png"),
            x_axis="training_time"
        )

        # Efficiency analysis
        paths_labels = [
            (experiment_paths["pinnpi"], "PINN-PI"),
            (experiment_paths["sac"], "SAC")
        ]
        plot_efficiency_analysis(
            paths_labels,
            metric,
            os.path.join(output_dir, f"{metric.replace('/', '_')}_efficiency.png")
        )

def plot_training_progress(metrics_path: str, output_dir: str,
                          main_metrics: List[str] = None,
                          timing_metrics: bool = True):
    """
    Generate training progress plots from metrics JSONL file.

    Creates plots for main performance metrics and optional timing analysis.
    This function is called during training at checkpoint saves and at the end.

    Args:
        metrics_path: Path to metrics.jsonl file
        output_dir: Directory to save plots
        main_metrics: List of main metrics to plot (default: ['eval/avg_return'])
        timing_metrics: Whether to plot timing metrics (default: True)
    """
    if main_metrics is None:
        main_metrics = ["eval/avg_return"]

    os.makedirs(output_dir, exist_ok=True)

    # Plot main performance metrics with step-based x-axis
    for metric in main_metrics:
        try:
            std_metric = None
            if metric == "eval/avg_return":
                std_metric = "eval/std_return"

            plot_learning_curve(
                metrics_path,
                metric,
                os.path.join(output_dir, f"{metric.replace('/', '_')}_vs_step.png"),
                title=f"{metric} vs Step/Iteration",
                x_axis="step",
                std_key=std_metric,
            )

            # Also create training-time based plot for efficiency analysis
            plot_learning_curve(
                metrics_path,
                metric,
                os.path.join(output_dir, f"{metric.replace('/', '_')}_vs_time.png"),
                title=f"{metric} vs Training Time",
                x_axis="training_time",
                std_key=std_metric,
            )
        except Exception as e:
            print(f"Warning: Could not plot {metric}: {e}")

    # Plot timing analysis if requested
    if timing_metrics:
        try:
            # Create multi-panel timing analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Training time per step
            try:
                x, y = load_series(metrics_path, "timing/step_training_time", "step")
                if len(x) > 0:
                    axes[0, 0].plot(x, y, linewidth=1, alpha=0.7)
                    axes[0, 0].set_xlabel("Step/Iteration")
                    axes[0, 0].set_ylabel("Training Time (s)")
                    axes[0, 0].set_title("Training Time per Step")
                    axes[0, 0].grid(True, alpha=0.3)
            except:
                axes[0, 0].text(0.5, 0.5, "No data", ha='center', va='center')

            # Plot 2: Cumulative training time
            try:
                x, y = load_series(metrics_path, "timing/total_training_time", "step")
                if len(x) > 0:
                    axes[0, 1].plot(x, y, linewidth=2)
                    axes[0, 1].set_xlabel("Step/Iteration")
                    axes[0, 1].set_ylabel("Cumulative Training Time (s)")
                    axes[0, 1].set_title("Total Training Time")
                    axes[0, 1].grid(True, alpha=0.3)
            except:
                axes[0, 1].text(0.5, 0.5, "No data", ha='center', va='center')

            # Plot 3: Wall time
            try:
                x, y = load_series(metrics_path, "timing/wall_time", "step")
                if len(x) > 0:
                    axes[1, 0].plot(x, y, linewidth=2, color='orange')
                    axes[1, 0].set_xlabel("Step/Iteration")
                    axes[1, 0].set_ylabel("Wall Time (s)")
                    axes[1, 0].set_title("Total Wall Time (includes eval)")
                    axes[1, 0].grid(True, alpha=0.3)
            except:
                axes[1, 0].text(0.5, 0.5, "No data", ha='center', va='center')

            # Plot 4: Evaluation time
            try:
                x, y = load_series(metrics_path, "timing/eval_time", "step")
                if len(x) > 0:
                    axes[1, 1].plot(x, y, linewidth=1, alpha=0.7, color='green')
                    axes[1, 1].set_xlabel("Step/Iteration")
                    axes[1, 1].set_ylabel("Evaluation Time (s)")
                    axes[1, 1].set_title("Evaluation Time per Checkpoint")
                    axes[1, 1].grid(True, alpha=0.3)
            except:
                axes[1, 1].text(0.5, 0.5, "No data", ha='center', va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "timing_analysis.png"), dpi=150)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not create timing analysis plot: {e}")

def plot_trajectory_norms(
    trajectories: np.ndarray,
    actions: Optional[np.ndarray] = None,
    out_png: str = None,
    title: str = "Trajectory Norms",
    plot_individual: bool = False,
    max_individual: int = 5,
    state_labels: Optional[List[str]] = None,
):
    """
    Plot norm of state (and optionally action) trajectories over time.

    Args:
        trajectories: State trajectories [traj_len+1, num_traj, d] or [traj_len+1, d]
        actions: Action trajectories [traj_len, num_traj, m] or [traj_len, m] (optional)
        out_png: Output image path (if None, displays plot)
        title: Plot title
        plot_individual: Whether to plot individual trajectory norms (default: False)
        max_individual: Maximum number of individual trajectories to plot
        state_labels: Optional labels for state dimensions
    """
    # Handle single trajectory case
    if trajectories.ndim == 2:
        trajectories = trajectories[:, None, :]  # [T+1, 1, d]
    if actions is not None and actions.ndim == 2:
        actions = actions[:, None, :]  # [T, 1, m]

    traj_len_plus_1, num_traj, d = trajectories.shape
    traj_len = traj_len_plus_1 - 1

    # Compute state norms: ||x_t|| for each trajectory
    state_norms = np.linalg.norm(trajectories, axis=2)  # [T+1, num_traj]

    # Compute action norms if provided
    action_norms = None
    if actions is not None:
        action_norms = np.linalg.norm(actions, axis=2)  # [T, num_traj]

    # Time steps
    time_steps = np.arange(traj_len_plus_1)
    action_time_steps = np.arange(traj_len) if actions is not None else None

    # Create figure
    n_plots = 2 if actions is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Plot state norms
    ax = axes[0]

    # Plot individual trajectories (if requested)
    if plot_individual and num_traj > 1:
        n_plot = min(max_individual, num_traj)
        for i in range(n_plot):
            ax.plot(time_steps, state_norms[:, i], alpha=0.3, linewidth=1, color='C0')

    # Plot mean and std
    if num_traj > 1:
        mean_state_norm = np.mean(state_norms, axis=1)
        std_state_norm = np.std(state_norms, axis=1)

        ax.plot(time_steps, mean_state_norm, linewidth=2, color='C0', label='Mean ||x||')
        ax.fill_between(
            time_steps,
            mean_state_norm - std_state_norm,
            mean_state_norm + std_state_norm,
            color='C0',
            alpha=0.2,
            label='±1 std'
        )
    else:
        # Single trajectory
        ax.plot(time_steps, state_norms[:, 0], linewidth=2, color='C0', label='||x||')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('State Norm ||x||')
    ax.set_title(f'{title} - State Norms')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot action norms (if provided)
    if actions is not None:
        ax = axes[1]

        # Plot individual trajectories (if requested)
        if plot_individual and num_traj > 1:
            n_plot = min(max_individual, num_traj)
            for i in range(n_plot):
                ax.plot(action_time_steps, action_norms[:, i], alpha=0.3, linewidth=1, color='C1')

        # Plot mean and std
        if num_traj > 1:
            mean_action_norm = np.mean(action_norms, axis=1)
            std_action_norm = np.std(action_norms, axis=1)

            ax.plot(action_time_steps, mean_action_norm, linewidth=2, color='C1', label='Mean ||u||')
            ax.fill_between(
                action_time_steps,
                mean_action_norm - std_action_norm,
                mean_action_norm + std_action_norm,
                color='C1',
                alpha=0.2,
                label='±1 std'
            )
        else:
            # Single trajectory
            ax.plot(action_time_steps, action_norms[:, 0], linewidth=2, color='C1', label='||u||')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Norm ||u||')
        ax.set_title(f'{title} - Action Norms')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_state_components(
    trajectories: np.ndarray,
    out_png: str = None,
    title: str = "State Components",
    state_labels: Optional[List[str]] = None,
    plot_individual: bool = False,
    max_individual: int = 3,
):
    """
    Plot individual state components over time.

    Args:
        trajectories: State trajectories [traj_len+1, num_traj, d] or [traj_len+1, d]
        out_png: Output image path (if None, displays plot)
        title: Plot title
        state_labels: Optional labels for state dimensions (e.g., ['x', 'v'])
        plot_individual: Whether to plot individual trajectories
        max_individual: Maximum number of individual trajectories to plot
    """
    # Handle single trajectory case
    if trajectories.ndim == 2:
        trajectories = trajectories[:, None, :]  # [T+1, 1, d]

    traj_len_plus_1, num_traj, d = trajectories.shape
    time_steps = np.arange(traj_len_plus_1)

    # Default labels
    if state_labels is None:
        state_labels = [f'x_{i}' for i in range(d)]

    # Determine grid layout
    n_cols = min(3, d)
    n_rows = (d + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if d == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for dim in range(d):
        ax = axes[dim]

        # Plot individual trajectories (if requested)
        if plot_individual and num_traj > 1:
            n_plot = min(max_individual, num_traj)
            for i in range(n_plot):
                ax.plot(time_steps, trajectories[:, i, dim], alpha=0.3, linewidth=1)

        # Plot mean and std
        if num_traj > 1:
            mean_comp = np.mean(trajectories[:, :, dim], axis=1)
            std_comp = np.std(trajectories[:, :, dim], axis=1)

            ax.plot(time_steps, mean_comp, linewidth=2, label=f'Mean {state_labels[dim]}')
            ax.fill_between(
                time_steps,
                mean_comp - std_comp,
                mean_comp + std_comp,
                alpha=0.2,
                label='±1 std'
            )
        else:
            # Single trajectory
            ax.plot(time_steps, trajectories[:, 0, dim], linewidth=2, label=state_labels[dim])

        ax.set_xlabel('Time Step')
        ax.set_ylabel(state_labels[dim])
        ax.set_title(f'State Component: {state_labels[dim]}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots
    for i in range(d, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

import argparse
import os
import sys

# Ensure repo root is importable when running this script directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pinn_spi.utils.plots import plot_comparison, load_series

def main():
    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument("--run1", required=True,
                       help="Directory containing first run's metrics.jsonl")
    parser.add_argument("--run2", required=True,
                       help="Directory containing second run's metrics.jsonl")
    parser.add_argument("--key", default="eval/avg_return",
                       help="Metric key to compare")
    parser.add_argument("--out", default="comparison.png",
                       help="Output plot file")
    parser.add_argument("--label1", default=None,
                       help="Label for first run")
    parser.add_argument("--label2", default=None,
                       help="Label for second run")
    parser.add_argument("--x_axis", default="step",
                       choices=["step", "training_time", "wall_time", "timestamp"],
                       help="X-axis variable for fair algorithm comparison")

    args = parser.parse_args()

    # Default labels from directory names
    label1 = args.label1 or os.path.basename(args.run1)
    label2 = args.label2 or os.path.basename(args.run2)

    # Prepare data for plotting
    runs_data = [
        (os.path.join(args.run1, "metrics.jsonl"), label1),
        (os.path.join(args.run2, "metrics.jsonl"), label2)
    ]

    print(f"Comparing runs:")
    print(f"  Run 1: {args.run1} -> {label1}")
    print(f"  Run 2: {args.run2} -> {label2}")
    print(f"  Metric: {args.key}")
    print(f"  X-axis: {args.x_axis}")

    # Create comparison plot
    plot_comparison(runs_data, args.key, args.out, x_axis=args.x_axis)
    print(f"Comparison plot saved to: {args.out}")

    # Print statistics
    for jsonl_path, label in runs_data:
        if os.path.exists(jsonl_path):
            x, y = load_series(jsonl_path, args.key, x_axis=args.x_axis)
            if len(y) > 0:
                print(f"\n{label}:")
                print(f"  Final value: {y[-1]:.3f}")
                print(f"  Mean value: {y.mean():.3f}")
                print(f"  Max value: {y.max():.3f}")
                if args.x_axis != "step":
                    print(f"  Final {args.x_axis}: {x[-1]:.3f}")
            else:
                print(f"\n{label}: No data found for key '{args.key}'")
        else:
            print(f"\n{label}: metrics.jsonl not found")

if __name__ == "__main__":
    main()

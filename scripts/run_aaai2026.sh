#!/bin/bash
#
# run_aaai2026.sh - Run experiments with AAAI2026 configurations
#
# This script provides a convenient interface to run experiments with various
# combinations of environments, algorithms, noise levels, and other parameters.
#
# Usage:
#   ./scripts/run_aaai2026.sh [OPTIONS]
#
# Options:
#   --env ENV            Environment: lqr5d, lqr10d, lqr20d, cartpole, pendulum, or all
#   --sigma SIGMA        Noise level: sigma0, sigma0p01, sigma0p03, sigma0p1, or all (default: all)
#   --bounds BOUNDS      LQR bounds: max10 (10,10), maxx1 (1,10), max1 (1,1), or all (default: max10)
#   --traj TRAJ          Trajectory length: 100, 200, or both (default: 100)
#   --algo ALGO          Algorithm: sac, spi, ppo, or all (default: all)
#   --norm NORM          Normalization: true, false, or both (default: true)
#   --eval EVAL          Evaluation type: sequential, vectorized, or both (default: sequential for cartpole/pendulum)
#   --dynamics DYN       Dynamics mode: gym, b, or both (default: gym for cartpole/pendulum)
#   --tag TAG            Experiment tag suffix (default: auto-generated)
#   --dry-run            Print commands without executing
#   --help               Show this help message
#
# Examples:
#   # Run LQR 10D with SAC, all noise levels, normalized inputs
#   ./scripts/run_aaai2026.sh --env lqr10d --algo sac --norm true
#
#   # Run all LQR environments with all bounds, PINN-SPI, sigma0p1
#   ./scripts/run_aaai2026.sh --env lqr --sigma sigma0p1 --algo spi --bounds all
#
#   # Run CartPole with PPO, both normalized and non-normalized
#   ./scripts/run_aaai2026.sh --env cartpole --algo ppo --norm both
#
#   # Run CartPole with SAC, both eval types, both dynamics modes
#   ./scripts/run_aaai2026.sh --env cartpole --algo sac --eval both --dynamics both
#
#   # Run full ablation: all envs, all algos, all noise, both norm
#   ./scripts/run_aaai2026.sh --env all --algo all --sigma all --norm both
#

set -e  # Exit on error

# ===================================
# Default values
# ===================================
ENV="all"
SIGMA="all"
BOUNDS="max10"
TRAJ="100"
ALGO="all"
NORM="true"
EVAL="auto"  # "auto" means sequential for cartpole/pendulum, vectorized for LQR
DYNAMICS="auto"  # "auto" means gym for cartpole/pendulum, b for LQR
TAG=""
DRY_RUN=false

# ===================================
# Parse command-line arguments
# ===================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --sigma)
            SIGMA="$2"
            shift 2
            ;;
        --bounds)
            BOUNDS="$2"
            shift 2
            ;;
        --traj)
            TRAJ="$2"
            shift 2
            ;;
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --norm)
            NORM="$2"
            shift 2
            ;;
        --eval)
            EVAL="$2"
            shift 2
            ;;
        --dynamics)
            DYNAMICS="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            head -n 40 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ===================================
# Expand wildcards to lists
# ===================================

# Environments
case $ENV in
    all)
        ENVS=("lqr5d" "lqr10d" "lqr20d" "cartpole" "pendulum")
        ;;
    lqr)
        ENVS=("lqr5d" "lqr10d" "lqr20d")
        ;;
    *)
        ENVS=("$ENV")
        ;;
esac

# Sigma levels
case $SIGMA in
    all)
        SIGMAS=("sigma0" "sigma0p01" "sigma0p03" "sigma0p1")
        ;;
    *)
        SIGMAS=("$SIGMA")
        ;;
esac

# LQR bounds configurations
case $BOUNDS in
    all)
        BOUNDS_LIST=("max10" "maxx1" "max1")
        ;;
    *)
        BOUNDS_LIST=("$BOUNDS")
        ;;
esac

# Trajectory lengths
case $TRAJ in
    both)
        TRAJS=("100" "200")
        ;;
    *)
        TRAJS=("$TRAJ")
        ;;
esac

# Algorithms
case $ALGO in
    all)
        ALGOS=("sac" "spi" "ppo")
        ;;
    *)
        ALGOS=("$ALGO")
        ;;
esac

# Normalization
case $NORM in
    both)
        NORMS=("true" "false")
        ;;
    *)
        NORMS=("$NORM")
        ;;
esac

# Evaluation types
case $EVAL in
    both)
        EVALS=("sequential" "vectorized")
        ;;
    auto)
        EVALS=("auto")  # Will be determined per environment
        ;;
    *)
        EVALS=("$EVAL")
        ;;
esac

# Dynamics modes
case $DYNAMICS in
    both)
        DYNAMICS_MODES=("gym" "b")
        ;;
    auto)
        DYNAMICS_MODES=("auto")  # Will be determined per environment
        ;;
    *)
        DYNAMICS_MODES=("$DYNAMICS")
        ;;
esac

# ===================================
# Helper functions
# ===================================

# Get environment config file name
get_env_config() {
    local env=$1
    local sigma=$2
    local bounds=$3

    case $env in
        lqr*)
            case $bounds in
                max10)
                    # Standard bounds: max_x=10, max_u=10
                    echo "${env}_max10_${sigma}.yaml"
                    ;;
                maxx1)
                    # Tight state bounds: max_x=1, max_u=10
                    echo "${env}_max10_maxx1_${sigma}.yaml"
                    ;;
                max1)
                    # Small uniform bounds: max_x=1, max_u=1
                    echo "${env}_max1_maxx1_${sigma}.yaml"
                    ;;
            esac
            ;;
        cartpole|pendulum)
            # Non-LQR environments don't have bounds variants
            echo "${env}.yaml"
            ;;
    esac
}

# Get experiment config file name
get_expt_config() {
    local env=$1
    local traj=$2
    local algo=$3
    local eval_type=$4
    local dynamics_mode=$5

    # Determine defaults based on environment
    if [[ $eval_type == "auto" ]]; then
        if [[ $env == "cartpole" || $env == "pendulum" ]]; then
            eval_type="sequential"
        else
            eval_type="vectorized"
        fi
    fi

    if [[ $dynamics_mode == "auto" ]]; then
        if [[ $env == "cartpole" || $env == "pendulum" ]]; then
            dynamics_mode="gym"
        else
            dynamics_mode="b"
        fi
    fi

    # Build base filename
    local base=""

    # Special case: CartPole + SPI uses _spi suffix for checkpoint_every: 100
    if [[ $env == "cartpole" && $algo == "spi" ]]; then
        case $traj in
            100) base="${env}_expt_spi" ;;
            200) base="${env}_expt_traj200_spi" ;;
        esac
    else
        case $traj in
            100) base="${env}_expt" ;;
            200) base="${env}_expt_traj200" ;;
        esac
    fi

    # Add evaluation type suffix if not default
    if [[ $env == "cartpole" || $env == "pendulum" ]]; then
        if [[ $eval_type == "vectorized" ]]; then
            base="${base}_vectorized"
        fi
    fi

    # Add dynamics mode suffix if not default for this environment
    if [[ $env == "cartpole" || $env == "pendulum" ]]; then
        if [[ $dynamics_mode == "b" ]]; then
            base="${base}_dynamics_b"
        fi
    fi

    echo "${base}.yaml"
}

# Get algorithm config file name
get_algo_config() {
    local env=$1
    local algo=$2
    local norm=$3

    case $norm in
        true)
            echo "${env}_${algo}.yaml"
            ;;
        false)
            echo "${env}_${algo}_nonorm.yaml"
            ;;
    esac
}

# Build experiment tag
build_tag() {
    local env=$1
    local sigma=$2
    local bounds=$3
    local traj=$4
    local algo=$5
    local norm=$6
    local eval_type=$7
    local dynamics_mode=$8
    local custom_tag=$9

    # Remove 'sigma' prefix and convert to short form
    local sigma_short="${sigma#sigma}"
    # Convert 0p03 to 0p03 format (already in correct format)
    local tag="${env}-${sigma_short}"

    # Add bounds info for LQR
    if [[ $env == lqr* ]]; then
        case $bounds in
            max10) tag="${tag}-std" ;;
            maxx1) tag="${tag}-tight" ;;
            max1) tag="${tag}-small" ;;
        esac
    fi

    # Add trajectory length if not default
    if [[ $traj == "200" ]]; then
        tag="${tag}-traj200"
    fi

    # Add algorithm
    tag="${tag}-${algo}"

    # Add normalization status
    case $norm in
        true) tag="${tag}-norm" ;;
        false) tag="${tag}-nonorm" ;;
    esac

    # Add evaluation type if not default for this env
    if [[ $env == "cartpole" || $env == "pendulum" ]]; then
        if [[ $eval_type == "vectorized" ]]; then
            tag="${tag}-vec"
        fi
    fi

    # Add dynamics mode if not default for this env
    if [[ $env == "cartpole" || $env == "pendulum" ]]; then
        if [[ $dynamics_mode == "b" ]]; then
            tag="${tag}-dynB"
        fi
    fi

    # Add custom tag if provided
    if [[ -n $custom_tag ]]; then
        tag="${tag}-${custom_tag}"
    fi

    echo "$tag"
}

# ===================================
# Main execution loop
# ===================================

echo "=========================================="
echo "AAAI2026 Experiment Runner"
echo "=========================================="
echo "Configuration:"
echo "  Environments: ${ENVS[*]}"
echo "  Sigma levels: ${SIGMAS[*]}"
echo "  LQR bounds: ${BOUNDS_LIST[*]}"
echo "  Trajectory lengths: ${TRAJS[*]}"
echo "  Algorithms: ${ALGOS[*]}"
echo "  Normalization: ${NORMS[*]}"
echo "  Evaluation types: ${EVALS[*]}"
echo "  Dynamics modes: ${DYNAMICS_MODES[*]}"
if [[ $DRY_RUN == true ]]; then
    echo "  Mode: DRY RUN (commands will be printed, not executed)"
fi
echo "=========================================="
echo

experiment_count=0

for env in "${ENVS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        # For non-LQR environments, only use the first bounds config (it's ignored anyway)
        if [[ $env == "cartpole" || $env == "pendulum" ]]; then
            bounds_to_try=("${BOUNDS_LIST[0]}")
        else
            bounds_to_try=("${BOUNDS_LIST[@]}")
        fi

        for bounds in "${bounds_to_try[@]}"; do
            for traj in "${TRAJS[@]}"; do
                for algo in "${ALGOS[@]}"; do
                    for norm in "${NORMS[@]}"; do
                        for eval_type in "${EVALS[@]}"; do
                            for dynamics_mode in "${DYNAMICS_MODES[@]}"; do
                                # Build file paths
                                env_config="configs/aaai2026/${sigma}/$(get_env_config $env $sigma $bounds)"
                                expt_config="configs/aaai2026/${sigma}/$(get_expt_config $env $traj $algo $eval_type $dynamics_mode)"
                                algo_config="configs/aaai2026/${sigma}/$(get_algo_config $env $algo $norm)"

                                # Check if all config files exist
                                missing_files=()
                                [[ ! -f $env_config ]] && missing_files+=("$env_config")
                                [[ ! -f $expt_config ]] && missing_files+=("$expt_config")
                                [[ ! -f $algo_config ]] && missing_files+=("$algo_config")

                                if [[ ${#missing_files[@]} -gt 0 ]]; then
                                    echo "⊗ Skipping: Missing config files:"
                                    for f in "${missing_files[@]}"; do
                                        echo "    - $f"
                                    done
                                    echo
                                    continue
                                fi

                                # Build experiment tag
                                exp_tag=$(build_tag $env $sigma $bounds $traj $algo $norm $eval_type $dynamics_mode "$TAG")

                                # Build command
                                cmd="python scripts/train.py \\"
                                cmd="$cmd\n  --env $env_config \\"
                                cmd="$cmd\n  --exp $expt_config \\"
                                cmd="$cmd\n  --algo $algo_config \\"
                                cmd="$cmd\n  --tag \"$exp_tag\""

                                experiment_count=$((experiment_count + 1))
                                echo "[$experiment_count] Running: $exp_tag"
                                echo -e "$cmd"
                                echo

                                if [[ $DRY_RUN == false ]]; then
                                    # Execute command
                                    python scripts/train.py \
                                        --env "$env_config" \
                                        --exp "$expt_config" \
                                        --algo "$algo_config" \
                                        --tag "$exp_tag"

                                    echo "✓ Completed: $exp_tag"
                                    echo
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "=========================================="
if [[ $DRY_RUN == true ]]; then
    echo "Total experiments to run: $experiment_count"
    echo "(DRY RUN - no experiments were actually executed)"
else
    echo "All experiments completed: $experiment_count total"
fi
echo "=========================================="

#!/bin/bash

# Script to test SAC configurations on various environments
# Usage: ./scripts/test_sac_configs.sh [INDEX...] [--env ENV] [--sigma SIGMA] [--tag TAG]
#   INDEX: SAC configuration number(s) (0-34), or "all" to run all configs
#          Multiple indices can be specified: 0 1 3 5 7
#   --env: Environment to test (cartpole, pendulum, lqr5d, lqr10d, lqr20d)
#   --sigma: Noise level (sigma0, sigma0p01, sigma0p1)
#   --tag: Optional custom tag for the experiment

set -e  # Exit on error

# Default configurations
DEFAULT_ENV="cartpole"
DEFAULT_SIGMA="sigma0p1"
SAC_DIR="configs/sac"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help message
show_help() {
    echo "Usage: $0 [INDEX...] [--env ENV] [--sigma SIGMA] [--tag TAG]"
    echo ""
    echo "Test SAC configurations on various environments"
    echo ""
    echo "Arguments:"
    echo "  INDEX...      SAC configuration index (0-34), or 'all' to run all configs"
    echo "                Multiple indices can be specified: 0 1 3 5 7"
    echo "  --env ENV     Environment to test (default: cartpole)"
    echo "                Options: cartpole, pendulum, lqr5d, lqr10d, lqr20d"
    echo "  --sigma SIGMA Noise level (default: sigma0p1)"
    echo "                Options: sigma0, sigma0p01, sigma0p1"
    echo "  --tag TAG     Optional custom experiment tag (prepended to generated tag)"
    echo ""
    echo "Tag format: [CUSTOM_]ENV_SIGMA_sac_IDX_DESC"
    echo "  Example: cartpole_sigma0p1_sac_0_baseline"
    echo "           cartpole_sigma0p1_sac_1_large"
    echo "           lqr5d_sigma0_sac_24_all_features"
    echo "           mytest_pendulum_sigma0p01_sac_18_action_jacobian"
    echo ""
    echo "Available SAC configurations:"
    echo "  SAC_0:  Baseline (env-specific config from aaai2026 directory)"
    echo ""
    echo "Classic configs (1-17):"
    echo "  SAC_1:  Large networks [512,512]"
    echo "  SAC_2:  Extra large networks [1024,1024]"
    echo "  SAC_3:  Deep 3-layer networks [256,256,256]"
    echo "  SAC_4:  Deep 4-layer networks [128,128,128,128]"
    echo "  SAC_5:  Balanced learning rates (1e-4/1e-4)"
    echo "  SAC_6:  Higher actor LR (1e-4/3e-4)"
    echo "  SAC_7:  Fixed alpha (no auto-tuning)"
    echo "  SAC_8:  High tau (0.005)"
    echo "  SAC_9:  Low tau (0.001)"
    echo "  SAC_10: Single critic (no twin)"
    echo "  SAC_11: Large batch (512)"
    echo "  SAC_12: Small batch (128)"
    echo "  SAC_13: Small buffer (100K)"
    echo "  SAC_14: Update every 4 steps"
    echo "  SAC_15: Long warmup (1000 steps)"
    echo "  SAC_16: Epsilon-greedy exploration"
    echo "  SAC_17: High alpha LR (5e-4)"
    echo ""
    echo "Advanced features (18-27: New implementation options):"
    echo "  SAC_18: Action Jacobian"
    echo "  SAC_19: Tanh log std mode"
    echo "  SAC_20: Delayed policy updates (freq=2)"
    echo "  SAC_21: Jacobian + Tanh log std"
    echo "  SAC_22: Jacobian + Delayed policy"
    echo "  SAC_23: Tanh log std + Delayed policy"
    echo "  SAC_24: All new features"
    echo "  SAC_25: Jacobian + Large network"
    echo "  SAC_26: Tanh log std + Large network"
    echo "  SAC_27: All features + Large network"
    echo ""
    echo "Separate policy architecture (28-34):"
    echo "  SAC_28: Separate policy (baseline)"
    echo "  SAC_29: Separate policy + Large network"
    echo "  SAC_30: Separate policy + Action Jacobian"
    echo "  SAC_31: Separate policy + Tanh log std"
    echo "  SAC_32: Separate policy + All features"
    echo "  SAC_33: Separate policy + Extra large network"
    echo "  SAC_34: Separate policy + All features + Large network"
    echo ""
    echo "Examples:"
    echo "  $0 0                              # Run baseline SAC on cartpole with sigma0p1"
    echo "  $0 0 1 --env pendulum             # Run baseline and SAC_1 on pendulum"
    echo "  $0 1 --env pendulum               # Run SAC_1 on pendulum"
    echo "  $0 1 --sigma sigma0               # Run SAC_1 with no noise"
    echo "  $0 0 --env lqr5d --sigma sigma0p01  # Run baseline on lqr5d with sigma=0.01"
    echo "  $0 0 1 3 5 7 --env cartpole       # Run baseline + selected configs on cartpole"
    echo "  $0 18 19 20 --env pendulum        # Run advanced feature tests on pendulum"
    echo "  $0 24 27 --env lqr10d --sigma sigma0  # Run all features on lqr10d deterministic"
    echo "  $0 28 29 --env lqr5d              # Test separate policy architecture"
    echo "  $0 32 34 --env pendulum --sigma sigma0  # Test separate + all features"
    echo "  $0 5 --tag test_lr                # Run SAC_5 with custom tag"
    echo "  $0 all --env lqr5d                # Run all 35 configs (0-34) on lqr5d"
    echo ""
}

# Parse arguments
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Collect indices and parse arguments
INDICES=()
ENV="$DEFAULT_ENV"
SIGMA="$DEFAULT_SIGMA"
CUSTOM_TAG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --env|-e)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: --env requires an argument${NC}"
                show_help
                exit 1
            fi
            ENV="$2"
            shift 2
            ;;
        --sigma|-s)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: --sigma requires an argument${NC}"
                show_help
                exit 1
            fi
            SIGMA="$2"
            shift 2
            ;;
        --tag)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: --tag requires an argument${NC}"
                show_help
                exit 1
            fi
            CUSTOM_TAG="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # Assume it's an index
            INDICES+=("$1")
            shift
            ;;
    esac
done

# Validate that we have at least one index
if [ ${#INDICES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No indices specified${NC}"
    show_help
    exit 1
fi

# Validate environment
case "$ENV" in
    cartpole|pendulum|lqr5d|lqr10d|lqr20d)
        # Valid environment
        ;;
    *)
        echo -e "${RED}Error: Invalid environment '$ENV'. Must be: cartpole, pendulum, lqr5d, lqr10d, lqr20d${NC}"
        show_help
        exit 1
        ;;
esac

# Validate sigma
case "$SIGMA" in
    sigma0|sigma0p01|sigma0p1)
        # Valid sigma
        ;;
    *)
        echo -e "${RED}Error: Invalid sigma '$SIGMA'. Must be: sigma0, sigma0p01, sigma0p1${NC}"
        show_help
        exit 1
        ;;
esac

# Construct config paths based on environment and sigma
BASE_DIR="configs/aaai2026/${SIGMA}"

# Handle environment config filename (LQR environments include sigma in filename)
if [[ "$ENV" == lqr* ]]; then
    ENV_CONFIG="${BASE_DIR}/${ENV}_max10_${SIGMA}.yaml"
else
    ENV_CONFIG="${BASE_DIR}/${ENV}.yaml"
fi

EXP_CONFIG="${BASE_DIR}/${ENV}_expt.yaml"

# If first index is "all", expand to all indices
if [ "${INDICES[0]}" = "all" ]; then
    if [ ${#INDICES[@]} -gt 1 ]; then
        echo -e "${RED}Error: Cannot specify 'all' with other indices${NC}"
        exit 1
    fi
    INDICES=($(seq 0 34))
else
    # Validate each index is a number between 0 and 34
    for idx in "${INDICES[@]}"; do
        if ! [[ "$idx" =~ ^[0-9]+$ ]] || [ "$idx" -lt 0 ] || [ "$idx" -gt 34 ]; then
            echo -e "${RED}Error: Invalid index '$idx'. Must be 0-34 or 'all'${NC}"
            show_help
            exit 1
        fi
    done
fi

# Function to run a single SAC configuration
run_sac_config() {
    local idx=$1
    local sac_config=""
    local desc=""

    # Handle index 0 (baseline from aaai2026 directory)
    if [ "$idx" -eq 0 ]; then
        sac_config="${BASE_DIR}/${ENV}_sac.yaml"
        desc="baseline"

        if [ ! -f "$sac_config" ]; then
            echo -e "${RED}Error: Baseline SAC configuration not found: $sac_config${NC}"
            return 1
        fi
    else
        # Handle index 1-34 (configs from configs/sac directory)
        local sac_file="${SAC_DIR}/SAC_${idx}_*.yaml"
        sac_config=$(ls ${sac_file} 2>/dev/null | head -n 1)

        if [ -z "$sac_config" ]; then
            echo -e "${RED}Error: SAC configuration ${idx} not found${NC}"
            return 1
        fi

        # Extract description from filename
        desc=$(basename "$sac_config" .yaml | cut -d'_' -f3-)
    fi

    # Build tag with environment and sigma
    local tag="${ENV}_${SIGMA}_sac_${idx}_${desc}"
    if [ -n "$CUSTOM_TAG" ]; then
        tag="${CUSTOM_TAG}_${tag}"
    fi

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running SAC Configuration ${idx}: ${desc}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Environment: $ENV_CONFIG"
    echo "Experiment:  $EXP_CONFIG"
    echo "Algorithm:   $sac_config"
    echo "Tag:         $tag"
    echo ""

    # Check if config files exist
    if [ ! -f "$ENV_CONFIG" ]; then
        echo -e "${RED}Error: Environment config not found: $ENV_CONFIG${NC}"
        return 1
    fi

    if [ ! -f "$EXP_CONFIG" ]; then
        echo -e "${RED}Error: Experiment config not found: $EXP_CONFIG${NC}"
        return 1
    fi

    # Run training
    if python -u scripts/train.py \
        --env "$ENV_CONFIG" \
        --exp "$EXP_CONFIG" \
        --algo "$sac_config" \
        --tag "$tag"; then
        echo -e "${GREEN}Success: SAC_${idx} completed${NC}"
        return 0
    else
        echo -e "${RED}Error: SAC_${idx} failed${NC}"
        return 1
    fi
}

# Main execution
START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_CONFIGS=()

# Display which configurations will be run
NUM_CONFIGS=${#INDICES[@]}
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Test Configuration${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "Environment: $ENV"
echo "Sigma:       $SIGMA"
if [ $NUM_CONFIGS -eq 35 ]; then
    echo "SAC configs: All (0-34)"
elif [ $NUM_CONFIGS -eq 1 ]; then
    echo "SAC configs: ${INDICES[0]}"
else
    echo "SAC configs: ${INDICES[@]}"
fi
echo ""

# Run each configuration
for idx in "${INDICES[@]}"; do
    if run_sac_config $idx; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CONFIGS+=($idx)
    fi
    echo ""
done

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $NUM_CONFIGS -gt 1 ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Total configurations: $NUM_CONFIGS"
    echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
    echo -e "${RED}Failed: $FAIL_COUNT${NC}"
    echo "Duration: ${DURATION}s"

    if [ $FAIL_COUNT -gt 0 ]; then
        echo ""
        echo -e "${RED}Failed configurations:${NC}"
        for failed_idx in "${FAILED_CONFIGS[@]}"; do
            echo "  - SAC_${failed_idx}"
        done
        exit 1
    else
        echo -e "${GREEN}All configurations completed successfully!${NC}"
        exit 0
    fi
else
    # Single configuration - simpler message
    if [ $SUCCESS_COUNT -eq 1 ]; then
        echo -e "${GREEN}Configuration SAC_${INDICES[0]} completed successfully! (${DURATION}s)${NC}"
        exit 0
    else
        echo -e "${RED}Configuration SAC_${INDICES[0]} failed! (${DURATION}s)${NC}"
        exit 1
    fi
fi

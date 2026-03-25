#!/usr/bin/env bash
# Run 4 model variants × N_RUNS seeds each.
# Results saved under: ../results/20_03_26_value_subspace_experiments/
#
# Variants:
#   1. Full plasticity  + no meta-RL inputs   (freeze=none,         prevAR=0)
#   2. Readout only     + no meta-RL inputs   (freeze=readout_only, prevAR=0)
#   3. Full plasticity  + meta-RL inputs      (freeze=none,         prevAR=1)
#   4. Readout only     + meta-RL inputs      (freeze=readout_only, prevAR=1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_meta_ac_abcdef_partial_multirun.py"
DATE_TAG="${DATE_TAG:-$(date +%d_%m_%y)}"
RESULTS_SUBDIR_DEFAULT="${DATE_TAG}_multirev_value_subspace_experiments"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results/$RESULTS_SUBDIR_DEFAULT}"
N_RUNS=${N_RUNS:-10}
OVERWRITE=${OVERWRITE:-1}
TASK=${TASK:-reversal_abcdef_multitimestep_partial_multirev}

# Locate mamba/conda python regardless of whether shell init has run
MAMBA="${HOME}/miniforge3/bin/mamba"
if [ ! -x "$MAMBA" ]; then
    MAMBA="$(command -v mamba 2>/dev/null || command -v conda 2>/dev/null)"
fi

run_variant() {
    local freeze_mode="$1"
    local use_prev_ar="$2"
    local label="$3"
    echo ""
    echo "========================================================"
    echo "Variant: $label"
    echo "  freeze_mode=$freeze_mode  use_prev_action_reward=$use_prev_ar"
    echo "========================================================"
    local overwrite_flag=""
    if [ "$OVERWRITE" -eq 1 ]; then
        overwrite_flag="--overwrite"
    fi
    "$MAMBA" run -n cog_nn python3 "$RUNNER" \
        --freeze_mode "$freeze_mode" \
        --use_prev_action_reward "$use_prev_ar" \
        --n_runs "$N_RUNS" \
        --task "$TASK" \
        --results_dir "$RESULTS_DIR" \
        $overwrite_flag
}

# 1. Full plasticity + no meta-RL
run_variant none 0 "Full plasticity + no prev-AR"

# 2. Readout only + no meta-RL
run_variant readout_only 0 "Readout only + no prev-AR"

# 3. Full plasticity + meta-RL
run_variant none 1 "Full plasticity + meta-RL"

# 4. Readout only + meta-RL
run_variant readout_only 1 "Readout only + meta-RL"

echo ""
echo "All variants complete. Results in: $RESULTS_DIR"

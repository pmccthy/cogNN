#!/usr/bin/env bash
# Single-reversal ACE plasticity-switch experiment — model-size sweep.
# Runs the same 1 000-trial-per-phase job for hidden sizes: 2, 4, 8, 16, 32, 64, 128.
# Each size is run as a separate variant inside the shared results directory.
# Stimulus set: A (H→L), C (50% always), E (L→H).
# Phase structure: 1 training reversal (plasticity ON) + 1 reference reversal (plasticity OFF).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

TRIALS_PER_PHASE=1000
RESULTS_DIR="results/21_04_26_singlerev_ace_plasticity_model_sizes"

# Generate task data for all seeds if not already present
for SEED in 42 43 44 45 46 47 48 49 50 51; do
    python3 -m cog_nn.tasks.generate_singlerev_ace_plasticity \
        --trials_per_phase "$TRIALS_PER_PHASE" \
        --seed "$SEED"
done

# Sweep over hidden sizes (powers of 2 from 2 to 128)
for HIDDEN_SIZE in 2 4 8 16 32 64 128; do
    echo "========================================"
    echo "Running hidden_size=${HIDDEN_SIZE}"
    echo "========================================"
    python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
        --n_runs            10 \
        --n_train_reversals 1 \
        --n_test_reversals  1 \
        --trials_per_phase  "$TRIALS_PER_PHASE" \
        --hidden_size       "$HIDDEN_SIZE" \
        --task_pkl          "task_data/reversal_ace_singlerev_plast_tpp${TRIALS_PER_PHASE}_seed42.pkl" \
        --plot_stims        0,2,4 \
        --results_dir       "$RESULTS_DIR"
done

echo "All model-size variants complete."
echo "Results saved to: $RESULTS_DIR"

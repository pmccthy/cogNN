#!/usr/bin/env bash
# Single-reversal ACE plasticity-switch — no-context, 1000 trials/phase, model size sweep.
#
# Runs the no-context variant across hidden unit sizes: 2, 4, 8, 16, 32, 64, 128.
# Each size gets its own subdirectory inside the main results folder (the variant
# name already encodes the readout size, which scales with hidden size at 50%).
#
# Results layout:
#   results/22_04_26_ace_tpp1000_model_sweep/
#     abef_plswitch_nocontext_train1rev_test1rev_tpp1000_partialro1_v2/   (hidden=2)
#     abef_plswitch_nocontext_train1rev_test1rev_tpp1000_partialro2_v2/   (hidden=4)
#     ...
#     abef_plswitch_nocontext_train1rev_test1rev_tpp1000_partialro64_v2/  (hidden=128)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

RESULTS_DIR="results/22_04_26_ace_tpp1000_model_sweep"
TASK_PKL="task_data/reversal_ace_singlerev_plast_tpp1000_seed42.pkl"

# Generate task data for seed 42 with 1000 trials per phase if not already present
python3 -m cog_nn.tasks.generate_singlerev_ace_plasticity \
    --trials_per_phase 1000 \
    --seed 42

for HIDDEN in 2 4 8 16 32 64 128; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  hidden_size = ${HIDDEN}"
    echo "════════════════════════════════════════"
    python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
        --n_runs            10 \
        --n_train_reversals 1 \
        --n_test_reversals  1 \
        --trials_per_phase  1000 \
        --hidden_size       "$HIDDEN" \
        --task_pkl          "$TASK_PKL" \
        --plot_stims        0,2,4 \
        --no_context \
        --results_dir       "$RESULTS_DIR"
done

echo ""
echo "All sizes done. Results in: $RESULTS_DIR"

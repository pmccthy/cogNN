#!/usr/bin/env bash
# No-reversal ACE plasticity-switch — model size sweep, both punishment variants.
#
# Runs hidden sizes 2, 4, 8, 16, 32, 64, 128 × 10 seeds for:
#   Variant A  lick_no_reward = -1.0  (punishment)   → results/22_04_26_norev_ace_punish_model_sweep/
#   Variant B  lick_no_reward =  0.0  (no punishment) → results/22_04_26_norev_ace_nopunish_model_sweep/
#
# Task: stable contingencies throughout (A=100%, C=50%, E=0%), no reversal.
# 2000 training trials (phases 0+1), plasticity OFF from trial 2000.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

TASK_PKL="task_data/ace_norev_plast_tpp1000_seed42.pkl"

# Generate task data if not already present
python3 -m cog_nn.tasks.generate_norev_ace_plasticity \
    --trials_per_phase 1000 \
    --seed 42

for PUNISH in -1.0 0.0; do
    if [ "$PUNISH" = "-1.0" ]; then
        RESULTS_DIR="results/22_04_26_norev_ace_punish_model_sweep"
        echo ""
        echo "████████████████████████████████████████"
        echo "  Variant: punishment = -1  (lick_no_reward=-1.0)"
        echo "  Results: $RESULTS_DIR"
        echo "████████████████████████████████████████"
    else
        RESULTS_DIR="results/22_04_26_norev_ace_nopunish_model_sweep"
        echo ""
        echo "████████████████████████████████████████"
        echo "  Variant: no punishment  (lick_no_reward=0.0)"
        echo "  Results: $RESULTS_DIR"
        echo "████████████████████████████████████████"
    fi

    for HIDDEN in 2 4 8 16 32 64 128; do
        echo ""
        echo "  ── hidden_size = ${HIDDEN} ──"
        python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
            --n_runs            10 \
            --n_train_reversals 1 \
            --n_test_reversals  1 \
            --trials_per_phase  1000 \
            --hidden_size       "$HIDDEN" \
            --task_pkl          "$TASK_PKL" \
            --plot_stims        0,2,4 \
            --lick_no_reward    "$PUNISH" \
            --no_context \
            --results_dir       "$RESULTS_DIR"
    done

    echo ""
    echo "  Done. Results in: $RESULTS_DIR"
done

echo ""
echo "Both variants complete."
echo "  Punishment (-1): results/22_04_26_norev_ace_punish_model_sweep/"
echo "  No punishment (0): results/22_04_26_norev_ace_nopunish_model_sweep/"

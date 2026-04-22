#!/usr/bin/env bash
# Single-reversal ACE plasticity-switch experiment.
# Stimulus set: A (H→L), C (50% always), E (L→H) — no B/D/F.
# Phase structure: 1 training reversal (plasticity ON) + 1 reference reversal (plasticity OFF).
# 500 trials per phase, pre-generated task pkl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Generate task data for all seeds if not already present
for SEED in 42 43 44 45 46 47 48 49 50 51; do
    python3 -m cog_nn.tasks.generate_singlerev_ace_plasticity \
        --trials_per_phase 500 \
        --seed "$SEED"
done

python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs 10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  500 \
    --task_pkl          task_data/reversal_ace_singlerev_plast_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --results_dir       results/16_04_26_singlerev_ace_plasticity

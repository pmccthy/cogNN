#!/usr/bin/env bash
# Single-reversal ACE plasticity-switch experiments — v2 script.
#
# Runs both the context and no-context variants using
# run_ace_plasticity_multirun_v2.py, which adds:
#   - within_trial_actions recording (binary lick/no-lick per timestep)
#   - trial_stim_lick_counts (lick count in stimulus window)
#   - time-resolved value decoder (pre-ITI + stim + reward window, per timepoint)
#   - clearer plot labels stating which timesteps each metric uses
#
# Task data is the same as used for the 16_04_26 runs (seed 42 pkl only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Generate task data for seed 42 if not already present
python3 -m cog_nn.tasks.generate_singlerev_ace_plasticity \
    --trials_per_phase 500 \
    --seed 42

# ── Context signal (model receives one-hot [pre, post] context) ───────────────
python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
    --n_runs            10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  500 \
    --task_pkl          task_data/reversal_ace_singlerev_plast_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --results_dir       results/21_04_26_singlerev_ace_plasticity_v2

# ── No-context (model receives state only, must infer reversal from rewards) ──
python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
    --n_runs            10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  500 \
    --task_pkl          task_data/reversal_ace_singlerev_plast_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --no_context \
    --results_dir       results/21_04_26_singlerev_ace_plasticity_v2

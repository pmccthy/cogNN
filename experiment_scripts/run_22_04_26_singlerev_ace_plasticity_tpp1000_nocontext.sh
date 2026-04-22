#!/usr/bin/env bash
# Single-reversal ACE plasticity-switch — no-context variant, 1000 trials/phase.
#
# Mirrors the 16_04_26_singlerev_ace_plasticity job but with:
#   - 1000 trials per phase (vs 500)
#   - no_context flag only
#   - uses run_ace_plasticity_multirun_v2.py (fixed within_trial_states,
#     adds within_trial_actions, trial_stim_lick_counts, time-resolved decoder)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Generate task data for seed 42 with 1000 trials per phase if not already present
python3 -m cog_nn.tasks.generate_singlerev_ace_plasticity \
    --trials_per_phase 1000 \
    --seed 42

# ── No-context (model receives state only, must infer reversal from rewards) ──
python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
    --n_runs            10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  1000 \
    --task_pkl          task_data/reversal_ace_singlerev_plast_tpp1000_seed42.pkl \
    --plot_stims        0,2,4 \
    --no_context \
    --results_dir       results/22_04_26_singlerev_ace_plasticity_tpp1000

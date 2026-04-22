#!/usr/bin/env bash
# Single-reversal ACE plasticity-switch experiment with 4 recurrent units.
# Runs both context and no-context variants for direct comparison.
# readout_fraction=1.0 so all 4 units project to actor/critic.

set -euo pipefail

# Task data already generated; generate any missing seeds
for SEED in 42 43 44 45 46 47 48 49 50 51; do
    python3 -m cog_nn.tasks.generate_singlerev_ace_plasticity \
        --trials_per_phase 500 \
        --seed "$SEED"
done

# ── With context signal ────────────────────────────────────────────────────────
python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs 10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  500 \
    --hidden_size       4 \
    --readout_fraction  1.0 \
    --task_pkl          task_data/reversal_ace_singlerev_plast_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --results_dir       results/17_04_26_singlerev_ace_4units

# ── Without context signal ────────────────────────────────────────────────────
python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs 10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  500 \
    --hidden_size       4 \
    --readout_fraction  1.0 \
    --task_pkl          task_data/reversal_ace_singlerev_plast_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --no_context \
    --results_dir       results/17_04_26_singlerev_ace_4units

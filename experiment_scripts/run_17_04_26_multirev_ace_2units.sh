#!/usr/bin/env bash
# Multi-reversal ACE plasticity-switch experiment with 2 recurrent units + context signal.
# Task: 10 training reversals (20 phases) + 1 plasticity-off reversal (2 phases).
# 500 trials per phase, stimuli A (H→L) / C (50%) / E (L→H).
# readout_fraction=1.0 so both units project to actor/critic.

set -euo pipefail

python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs            10 \
    --n_train_reversals 10 \
    --n_test_reversals   1 \
    --trials_per_phase  500 \
    --hidden_size         2 \
    --readout_fraction  1.0 \
    --task_pkl          task_data/reversal_ace_multirev_train20_test2_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --results_dir       results/17_04_26_multirev_ace_2units

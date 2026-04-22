#!/usr/bin/env bash
# Multi-reversal ACE plasticity-switch experiment with 1 recurrent unit + context signal.
# Identical to the 2-unit run except hidden_size=1.

set -euo pipefail

python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs            10 \
    --n_train_reversals 10 \
    --n_test_reversals   1 \
    --trials_per_phase  500 \
    --hidden_size         1 \
    --readout_fraction  1.0 \
    --task_pkl          task_data/reversal_ace_multirev_train20_test2_tpp500_seed42.pkl \
    --plot_stims        0,2,4 \
    --results_dir       results/17_04_26_multirev_ace_1unit

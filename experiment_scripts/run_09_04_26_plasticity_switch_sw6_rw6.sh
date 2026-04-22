#!/usr/bin/env bash
# Run plasticity-switch ABEF with stim_window=6, reward_window=6
# Results → results/09_04_26_plasticity_switch_sw6rw6_tpp400

python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs 10 \
    --n_reversals 10 \
    --trials_per_phase 400 \
    --stim_window 6 \
    --reward_window 6 \
    --results_dir results/09_04_26_plasticity_switch_sw6rw6_tpp400

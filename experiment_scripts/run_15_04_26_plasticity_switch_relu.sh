#!/usr/bin/env bash
# Repeat 09_04_26 plasticity-switch ABEF run with ReLU RNN nonlinearity.
# Identical hyperparameters; only change is nonlinearity='relu' in models.py.

python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs 10 \
    --n_reversals 10 \
    --trials_per_phase 400 \
    --stim_window 6 \
    --reward_window 6 \
    --results_dir results/15_04_26_plasticity_switch_relu

#!/usr/bin/env bash
# Plasticity-switch experiment with ReLU networks and the 50% mid-value stim C added.
# Stimulus set: A (H→L), B (H→H), C (50% always), E (L→H), F (L→L)
# Otherwise identical to run_15_04_26_plasticity_switch_relu.sh

python3 experiment_scripts/run_context_signal_abef_plasticity_switch_multirun.py \
    --n_runs 10 \
    --n_reversals 10 \
    --trials_per_phase 400 \
    --stim_window 6 \
    --reward_window 6 \
    --include_mid_stim \
    --results_dir results/15_04_26_plasticity_switch_relu_abcef

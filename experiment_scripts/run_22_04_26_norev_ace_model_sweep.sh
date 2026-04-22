#!/usr/bin/env bash
# No-reversal ACE plasticity-switch — no-context, 1000 trials/phase,
# punishment=0, model size sweep.
#
# Mirrors run_22_04_26_singlerev_ace_tpp1000_model_sweep.sh but for the
# no-reversal control task (stable contingencies throughout):
#   A: 100% always   C: 50% always   E: 0% always
#   lick_no_reward = 0.0  (no punishment for unrewarded licks)
#
# 2000 training trials (phases 0+1), plasticity OFF from trial 2000.
# Runs hidden unit sizes: 2, 4, 8, 16, 32, 64, 128 — 10 seeds each.
#
# Results layout:
#   results/22_04_26_norev_ace_model_sweep/
#     abef_plswitch_nocontext_train1rev_test1rev_tpp1000_punish0_h2_partialro1_v2/
#     abef_plswitch_nocontext_train1rev_test1rev_tpp1000_punish0_h4_partialro2_v2/
#     ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

RESULTS_DIR="results/22_04_26_norev_ace_model_sweep"
TASK_PKL="task_data/ace_norev_plast_tpp1000_seed42.pkl"

# Generate task data if not already present
python3 -m cog_nn.tasks.generate_norev_ace_plasticity \
    --trials_per_phase 1000 \
    --seed 42

for HIDDEN in 2 4 8 16 32 64 128; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  hidden_size = ${HIDDEN}"
    echo "════════════════════════════════════════"
    python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
        --n_runs            10 \
        --n_train_reversals 1 \
        --n_test_reversals  1 \
        --trials_per_phase  1000 \
        --hidden_size       "$HIDDEN" \
        --task_pkl          "$TASK_PKL" \
        --plot_stims        0,2,4 \
        --lick_no_reward    0.0 \
        --no_context \
        --results_dir       "$RESULTS_DIR"
done

echo ""
echo "All sizes done. Results in: $RESULTS_DIR"

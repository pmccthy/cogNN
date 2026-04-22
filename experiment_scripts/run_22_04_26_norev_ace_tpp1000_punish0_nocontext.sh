#!/usr/bin/env bash
# No-reversal ACE plasticity-switch — no-context, 1000 trials/phase, punishment=0.
#
# Identical to run_22_04_26_norev_ace_tpp1000_nocontext.sh except:
#   lick_no_reward = 0.0  (omission only — licking when unrewarded gives 0, not -1)
#
# The standard run uses lick_no_reward=-1.0 (punishment).  This version removes
# the punishment signal so the model never receives negative reward, only
# reward (A lick = +1) or nothing (C miss / E lick / no-lick = 0).
#
# Comparison between punishment=-1 and punishment=0 reveals whether the
# negative signal on unrewarded trials shapes the hidden-unit tuning.
#
# Results:
#   results/22_04_26_norev_ace_tpp1000_nocontext/
#     abef_plswitch_nocontext_train1rev_test1rev_tpp1000_punish0_h128_partialro64_v2/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

TASK_PKL="task_data/ace_norev_plast_tpp1000_seed42.pkl"

# Generate task data if not already present
python3 -m cog_nn.tasks.generate_norev_ace_plasticity \
    --trials_per_phase 1000 \
    --seed 42

python3 experiment_scripts/run_ace_plasticity_multirun_v2.py \
    --n_runs            10 \
    --n_train_reversals 1 \
    --n_test_reversals  1 \
    --trials_per_phase  1000 \
    --task_pkl          "$TASK_PKL" \
    --plot_stims        0,2,4 \
    --lick_no_reward    0.0 \
    --no_context \
    --results_dir       results/22_04_26_norev_ace_tpp1000_nocontext

echo ""
echo "Done. Results in: results/22_04_26_norev_ace_tpp1000_nocontext"

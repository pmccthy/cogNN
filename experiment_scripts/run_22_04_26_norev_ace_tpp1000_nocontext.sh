#!/usr/bin/env bash
# No-reversal ACE plasticity-switch — no-context variant, 1000 trials/phase.
#
# Control condition for the single-reversal ACE experiment.
# Contingencies are CONSTANT throughout — no reversal ever occurs:
#   A: 100% always   C: 50% always   E: 0% always
#
# Phase structure (all same contingencies):
#   Phase 0 & 1: plasticity ON  (2000 training trials total)
#   Phase 2 & 3: plasticity OFF (2000 test trials total)
#
# The model must learn stable contingencies, then run frozen.
# Comparing selectivity / SI distributions against the singlerev run
# reveals what tuning properties are driven by the reversal itself.
#
# Results:
#   results/22_04_26_norev_ace_tpp1000_nocontext/

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
    --no_context \
    --results_dir       results/22_04_26_norev_ace_tpp1000_nocontext

echo ""
echo "Done. Results in: results/22_04_26_norev_ace_tpp1000_nocontext"

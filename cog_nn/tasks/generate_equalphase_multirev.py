"""
Generate equal-phase multi-reversal task data for the ABCDEF partial-reversal task.

All phases have the same number of trials (unlike the original multirev data where
phase 1 is doubled).  The output pkl contains a combined training + test sequence so
the same file is used for both the training and the frozen-inference test phase.

Default: 10 training phases + 10 test phases, 4 000 trials each.

Usage
-----
    python cog_nn/tasks/generate_equalphase_multirev.py
    python cog_nn/tasks/generate_equalphase_multirev.py \
        --n_train_phases 10 --n_test_phases 10 --trials_per_phase 4000 --seed 42

Output
------
task_data/reversal_abcdef_equalphase_partial_train10_test10_tpp4000.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# ── Fixed trial structure ──────────────────────────────────────────────────────

STATE_MAP: dict[str, int] = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "reward_unknown": 6, "unrewarded": 7, "rewarded": 8, "ITI": 9,
}
STATE_DIM = 10

STIM_WINDOW   = 5   # timesteps showing stimulus
REWARD_WINDOW = 3   # timesteps in reward-availability window
MIN_ITI = 10
MAX_ITI = 20

# ── Reward contingencies ───────────────────────────────────────────────────────

# Even-indexed phases (pre-contingency)
REWARD_PROB_PRE: dict[int, float] = {
    0: 1.0,  # A: 100 %
    1: 1.0,  # B: 100 %
    2: 0.5,  # C: 50 %
    3: 0.5,  # D: 50 %
    4: 0.0,  # E:   0 %
    5: 0.0,  # F:   0 %
}

# Odd-indexed phases (post-contingency) — partial reversal: A↔E only
REWARD_PROB_POST_PARTIAL: dict[int, float] = {
    0: 0.0,  # A: 100 % → 0 %
    1: 1.0,  # B: unchanged 100 %
    2: 0.5,  # C: unchanged
    3: 0.5,  # D: unchanged
    4: 1.0,  # E:   0 % → 100 %
    5: 0.0,  # F: unchanged 0 %
}

ALL_STIMULI = list(range(6))


# ── Generation helpers ─────────────────────────────────────────────────────────

def _sample_reward(prob: float, rng: np.random.Generator) -> int:
    return 1 if rng.random() < prob else 0


def generate_sequence(phase_n_trials: list[int], seed: int = 42) -> dict:
    """Generate trial-level stimulus/reward/phase data."""
    rng = np.random.default_rng(seed)
    trial_data: dict = {"stimuli": [], "rewards": [], "masks": {"reversal": []}}
    for phase_idx, n_trials in enumerate(phase_n_trials):
        probs = REWARD_PROB_PRE if phase_idx % 2 == 0 else REWARD_PROB_POST_PARTIAL
        for _ in range(n_trials):
            stim = int(rng.choice(ALL_STIMULI))
            rew  = _sample_reward(probs[stim], rng)
            trial_data["stimuli"].append(stim)
            trial_data["rewards"].append(rew)
            trial_data["masks"]["reversal"].append(phase_idx)
    return trial_data


def expand_to_timesteps(
    trial_data: dict, seed: int = 42
) -> tuple[list, list, list, int]:
    """Expand trial-level data to per-timestep sequences.

    Returns (state_sequence, reward_sequence, trial_structure, total_timesteps).
    """
    # Separate RNG seed for ITI lengths so stimulus draws are reproducible.
    iti_rng = np.random.default_rng(seed + 1000)

    state_sequence: list[int]   = []
    reward_sequence: list[float] = []
    trial_structure: list[dict]  = []
    timestep = 0

    for trial_idx, (stim, reward_avail, phase_idx) in enumerate(zip(
        trial_data["stimuli"],
        trial_data["rewards"],
        trial_data["masks"]["reversal"],
    )):
        reversal_phase = int(phase_idx % 2)

        # ── Stimulus window ────────────────────────────────────────────────────
        stim_ts: list[int] = []
        for _ in range(STIM_WINDOW):
            state_sequence.append(stim)
            reward_sequence.append(0.0)
            stim_ts.append(timestep)
            timestep += 1

        # ── Reward window ──────────────────────────────────────────────────────
        rew_ts: list[int] = []
        for _ in range(REWARD_WINDOW):
            state_sequence.append(STATE_MAP["reward_unknown"])
            reward_sequence.append(float(reward_avail == 1))
            rew_ts.append(timestep)
            timestep += 1

        # ── ITI ───────────────────────────────────────────────────────────────
        iti_len = int(iti_rng.integers(MIN_ITI, MAX_ITI + 1))
        iti_ts: list[int] = []
        for _ in range(iti_len):
            state_sequence.append(STATE_MAP["ITI"])
            reward_sequence.append(0.0)
            iti_ts.append(timestep)
            timestep += 1

        trial_structure.append({
            "trial_idx":        trial_idx,
            "stimulus":         stim,
            "reward_available": bool(reward_avail == 1),
            "reversal_phase":   reversal_phase,
            "phase_idx":        phase_idx,
            "trial_start":      stim_ts[0],
            "stim_window":      stim_ts,
            "reward_window":    rew_ts,
            "iti_window":       iti_ts,
            "trial_end":        timestep - 1,
        })

    return state_sequence, reward_sequence, trial_structure, timestep


def make_phase_boundaries(
    trial_structure: list[dict],
    phase_n_trials: list[int],
    n_train_phases: int,
) -> dict:
    """Build phase_boundaries dict compatible with the existing training scripts."""
    phases: list[dict] = []
    cum = 0
    for pi, n in enumerate(phase_n_trials):
        start_trial = cum
        end_trial   = cum + n - 1
        start_ts    = trial_structure[start_trial]["trial_start"]
        end_ts      = trial_structure[end_trial]["trial_end"]
        phases.append({
            "phase_idx":   pi,
            "start":       start_ts,
            "end":         end_ts,
            "n_trials":    n,
            "contingency": "pre" if pi % 2 == 0 else "post",
            "split":       "train" if pi < n_train_phases else "test",
        })
        cum += n

    reversal_points = [phases[i]["start"] for i in range(1, len(phases))]
    total_ts        = trial_structure[-1]["trial_end"] + 1
    train_end_ts    = phases[n_train_phases - 1]["end"] + 1

    return {
        "phases":          phases,
        "reversal_points": reversal_points,
        "n_train_phases":  n_train_phases,
        "train_end":       train_end_ts,
        "total_end":       total_ts,
        # Backwards-compat keys used by existing training loop
        "pre_reversal":  {"start": 0,                     "end": phases[0]["end"]},
        "post_reversal": {"start": phases[1]["start"],     "end": total_ts},
    }


def _to_ohe(state_sequence: list[int]) -> np.ndarray:
    ohe = np.zeros((len(state_sequence), STATE_DIM), dtype=np.float32)
    for i, s in enumerate(state_sequence):
        if 0 <= s < STATE_DIM:
            ohe[i, s] = 1.0
    return ohe


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_train_phases",  type=int, default=10,
                        help="Number of training phases (each = 1 pre or post block).")
    parser.add_argument("--n_test_phases",   type=int, default=10,
                        help="Number of frozen-inference test phases.")
    parser.add_argument("--trials_per_phase", type=int, default=4000,
                        help="Trials per phase (same for all phases).")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output_dir",      type=str, default=None,
                        help="Directory to save pkl (default: <repo_root>/task_data).")
    args = parser.parse_args()

    # Locate repo root
    here = Path(__file__).resolve()
    root = here.parent
    while not (root / "experiment_scripts").exists() and root != root.parent:
        root = root.parent
    output_dir = Path(args.output_dir) if args.output_dir else root / "task_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train = args.n_train_phases
    n_test  = args.n_test_phases
    tpp     = args.trials_per_phase
    phase_n_trials = [tpp] * (n_train + n_test)
    n_total_phases = n_train + n_test

    print(f"Generating equal-phase partial-reversal ABCDEF data:")
    print(f"  {n_train} training phases + {n_test} test phases = {n_total_phases} total")
    print(f"  {tpp} trials per phase  ({tpp * n_total_phases:,} total trials)")
    print(f"  Contingency: partial reversal (A↔E only)")
    print(f"  Seed: {args.seed}")

    # Generate
    trial_data = generate_sequence(phase_n_trials, seed=args.seed)
    state_sequence, reward_sequence, trial_structure, total_ts = expand_to_timesteps(
        trial_data, seed=args.seed)
    phase_boundaries = make_phase_boundaries(trial_structure, phase_n_trials, n_train)
    state_sequence_ohe = _to_ohe(state_sequence)

    data = {
        "state_sequence_ohe": state_sequence_ohe,
        "reward_sequence":    np.array(reward_sequence, dtype=np.float32),
        "sequence": {
            "stimuli": trial_data["stimuli"],
            "rewards": trial_data["rewards"],
            "masks":   trial_data["masks"],
        },
        "phase_boundaries":   phase_boundaries,
        "trial_structure":    trial_structure,
        "state_map":          STATE_MAP,
        "trial_params": {
            "stim_window":   STIM_WINDOW,
            "reward_window": REWARD_WINDOW,
            "min_iti":       MIN_ITI,
            "max_iti":       MAX_ITI,
        },
        "phase_n_trials":      phase_n_trials,
        "n_train_phases":      n_train,
        "n_test_phases":       n_test,
        "reversal_type":       "partial",
        "equal_phase_lengths": True,
    }

    fname    = f"reversal_abcdef_equalphase_partial_train{n_train}_test{n_test}_tpp{tpp}.pkl"
    out_path = output_dir / fname
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    pb = phase_boundaries
    print(f"\nSaved: {out_path}")
    print(f"Total timesteps:     {total_ts:,}")
    print(f"Training ends at ts: {pb['train_end']:,}")
    print(f"Test starts at ts:   {pb['phases'][n_train]['start']:,}")
    print(f"First 6 phases:")
    for ph in pb["phases"][:6]:
        print(f"  Phase {ph['phase_idx']:2d} ({ph['contingency']:4s}, {ph['split']:5s}, "
              f"{ph['n_trials']} trials): ts {ph['start']:>10,} – {ph['end']:>10,}")
    if n_total_phases > 6:
        print(f"  ...")


if __name__ == "__main__":
    main()

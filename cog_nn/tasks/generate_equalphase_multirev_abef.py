"""
Generate equal-phase multi-reversal task data for an ABCDEF-reduced task (A/B/E/F).

This is a simplified version of the equal-phase partial-reversal ABCDEF task where
the mid-value stimuli C/D are removed entirely. Only stimuli A, B, E, F are sampled.

Important: the state encoding remains compatible with the existing ABCDEF multitimestep
environment (`ReversalABCDEFMultiTimestepEnv`):
- Stimuli keep their original indices: A=0, B=1, E=4, F=5
- Outcome/ITI states keep their original indices: reward_unknown=6, unrewarded=7,
  rewarded=8, ITI=9

So you can run the existing training scripts unchanged, just by pointing them at the
new task pkl.

Default: 10 training phases + 10 test phases, 4 000 trials each.

Usage
-----
    python -m cog_nn.tasks.generate_equalphase_multirev_abef
    python -m cog_nn.tasks.generate_equalphase_multirev_abef \\
        --n_train_phases 10 --n_test_phases 10 --trials_per_phase 4000 --seed 42

Output
------
task_data/reversal_abef_equalphase_partial_train10_test10_tpp4000.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# ── Fixed trial structure ──────────────────────────────────────────────────────

STATE_MAP: dict[str, int] = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "reward_unknown": 6,
    "unrewarded": 7,
    "rewarded": 8,
    "ITI": 9,
}
STATE_DIM = 10

STIM_WINDOW = 5
REWARD_WINDOW = 3
MIN_ITI = 10
MAX_ITI = 20

# ── Reward contingencies ───────────────────────────────────────────────────────

REWARD_PROB_PRE: dict[int, float] = {
    0: 1.0,  # A
    1: 1.0,  # B
    2: 0.5,  # C (unused)
    3: 0.5,  # D (unused)
    4: 0.0,  # E
    5: 0.0,  # F
}

REWARD_PROB_POST_PARTIAL: dict[int, float] = {
    0: 0.0,  # A: 100 % → 0 %
    1: 1.0,  # B: unchanged
    2: 0.5,  # C (unused)
    3: 0.5,  # D (unused)
    4: 1.0,  # E:   0 % → 100 %
    5: 0.0,  # F: unchanged
}

# Only these stimuli are sampled.
ALL_STIMULI = [0, 1, 4, 5]


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
            rew = _sample_reward(probs[stim], rng)
            trial_data["stimuli"].append(stim)
            trial_data["rewards"].append(rew)
            trial_data["masks"]["reversal"].append(phase_idx)
    return trial_data


def expand_to_timesteps(trial_data: dict, seed: int = 42) -> tuple[list, list, list, int]:
    """Expand trial-level data to per-timestep sequences."""
    iti_rng = np.random.default_rng(seed + 1000)

    state_sequence: list[int] = []
    reward_sequence: list[float] = []
    trial_structure: list[dict] = []
    timestep = 0

    for trial_idx, (stim, reward_avail, phase_idx) in enumerate(
        zip(
            trial_data["stimuli"],
            trial_data["rewards"],
            trial_data["masks"]["reversal"],
        )
    ):
        reversal_phase = int(phase_idx % 2)

        stim_ts: list[int] = []
        for _ in range(STIM_WINDOW):
            state_sequence.append(int(stim))
            reward_sequence.append(0.0)
            stim_ts.append(timestep)
            timestep += 1

        rew_ts: list[int] = []
        for _ in range(REWARD_WINDOW):
            state_sequence.append(STATE_MAP["reward_unknown"])
            reward_sequence.append(float(reward_avail == 1))
            rew_ts.append(timestep)
            timestep += 1

        iti_len = int(iti_rng.integers(MIN_ITI, MAX_ITI + 1))
        iti_ts: list[int] = []
        for _ in range(iti_len):
            state_sequence.append(STATE_MAP["ITI"])
            reward_sequence.append(0.0)
            iti_ts.append(timestep)
            timestep += 1

        trial_structure.append(
            {
                "trial_idx": trial_idx,
                "stimulus": int(stim),
                "reward_available": bool(reward_avail == 1),
                "reversal_phase": reversal_phase,
                "phase_idx": int(phase_idx),
                "trial_start": stim_ts[0],
                "stim_window": stim_ts,
                "reward_window": rew_ts,
                "iti_window": iti_ts,
                "trial_end": timestep - 1,
            }
        )

    return state_sequence, reward_sequence, trial_structure, timestep


def make_phase_boundaries(
    trial_structure: list[dict],
    phase_n_trials: list[int],
    n_train_phases: int,
) -> dict:
    """Build phase_boundaries dict compatible with existing training scripts."""
    phases: list[dict] = []
    cum = 0
    for pi, n in enumerate(phase_n_trials):
        start_trial = cum
        end_trial = cum + n - 1
        start_ts = int(trial_structure[start_trial]["trial_start"])
        end_ts = int(trial_structure[end_trial]["trial_end"])
        phases.append(
            {
                "phase_idx": int(pi),
                "start": start_ts,
                "end": end_ts,
                "n_trials": int(n),
                "contingency": "pre" if pi % 2 == 0 else "post",
                "split": "train" if pi < n_train_phases else "test",
            }
        )
        cum += n

    reversal_points = [phases[i]["start"] for i in range(1, len(phases))]
    total_ts = int(trial_structure[-1]["trial_end"]) + 1
    train_end_ts = int(phases[n_train_phases - 1]["end"]) + 1

    return {
        "phases": phases,
        "reversal_points": reversal_points,
        "n_train_phases": int(n_train_phases),
        "n_test_phases": int(len(phases) - n_train_phases),
        "train_end": train_end_ts,
        "total_end": total_ts,
        "pre_reversal": {"start": 0, "end": int(phases[0]["end"])},
        "post_reversal": {"start": int(phases[1]["start"]), "end": total_ts},
    }


def _to_ohe(state_sequence: list[int]) -> np.ndarray:
    ohe = np.zeros((len(state_sequence), STATE_DIM), dtype=np.float32)
    for i, s in enumerate(state_sequence):
        if 0 <= int(s) < STATE_DIM:
            ohe[i, int(s)] = 1.0
    return ohe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_train_phases", type=int, default=10)
    parser.add_argument("--n_test_phases", type=int, default=10)
    parser.add_argument("--trials_per_phase", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve()
    root = here.parent
    while not (root / "experiment_scripts").exists() and root != root.parent:
        root = root.parent
    output_dir = Path(args.output_dir) if args.output_dir else root / "task_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_n_trials = [int(args.trials_per_phase)] * int(args.n_train_phases + args.n_test_phases)

    trial_data = generate_sequence(phase_n_trials, seed=int(args.seed))
    state_sequence, reward_sequence, trial_structure, total_ts = expand_to_timesteps(
        trial_data, seed=int(args.seed)
    )
    phase_boundaries = make_phase_boundaries(trial_structure, phase_n_trials, int(args.n_train_phases))
    state_sequence_ohe = _to_ohe(state_sequence)

    data = {
        "state_sequence_ohe": state_sequence_ohe,
        "reward_sequence": np.asarray(reward_sequence, dtype=np.float32),
        "sequence": {
            "stimuli": trial_data["stimuli"],
            "rewards": trial_data["rewards"],
            "masks": trial_data["masks"],
        },
        "phase_boundaries": phase_boundaries,
        "trial_structure": trial_structure,
        "state_map": STATE_MAP,
        "trial_params": {
            "stim_window": STIM_WINDOW,
            "reward_window": REWARD_WINDOW,
            "min_iti": MIN_ITI,
            "max_iti": MAX_ITI,
        },
        "phase_n_trials": phase_n_trials,
        "n_train_phases": int(args.n_train_phases),
        "n_test_phases": int(args.n_test_phases),
        "reversal_type": "partial",
        "equal_phase_lengths": True,
        "stimulus_set": ["A", "B", "E", "F"],
        "stimulus_indices": ALL_STIMULI,
    }

    fname = (
        f"reversal_abef_equalphase_partial_train{int(args.n_train_phases)}_"
        f"test{int(args.n_test_phases)}_tpp{int(args.trials_per_phase)}.pkl"
    )
    out_path = output_dir / fname
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {out_path}")
    print(f"Stimuli sampled: {data['stimulus_set']} (indices={ALL_STIMULI})")
    print(f"Total timesteps: {total_ts:,}")


if __name__ == "__main__":
    main()


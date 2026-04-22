"""
Regenerate plots from saved run outputs.

This script rebuilds figures from:
- per-run `metrics_numpy.pkl` (hidden states, lick/value, grad norms)
- per-run `run_results.pkl`   (decoder + TDR outputs already computed)
- task `.pkl` file            (trial_structure + phase boundaries for markers)

It does NOT require model weights.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Reuse plotting utilities from the main runner script.
from experiment_scripts.run_meta_ac_abcdef_partial_multirun import (  # noqa: E402
    _log,
    plot_decoder,
    plot_grad_norms,
    plot_lick_value,
    plot_phase_generalisation_matrices,
    plot_tdr_value_psth,
)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _stim_window_mean_activations(metrics_numpy, trial_structure):
    """Return (acts, stim_labels, reversal_phase_labels, phase_idx_labels)."""
    acts, stims, revs, phs = [], [], [], []
    for tb in metrics_numpy["trial_boundaries"]:
        ti = tb["trial_idx"]
        hs_list = metrics_numpy["hidden_states"].get(ti)
        if not hs_list:
            continue
        info = trial_structure[ti]
        t0 = info["trial_start"]
        stim_rel = np.asarray(info["stim_window"]) - t0
        wt_ts = np.asarray(metrics_numpy["within_trial_timesteps"][ti])
        wt_rel = wt_ts - t0
        xs = []
        for tr in stim_rel:
            m = np.where(wt_rel == tr)[0]
            if len(m) and m[0] < len(hs_list):
                xs.append(np.asarray(hs_list[m[0]]).squeeze())
        if not xs:
            continue
        acts.append(np.mean(np.stack(xs, axis=0), axis=0))
        stims.append(int(info["stimulus"]))
        revs.append(int(info.get("reversal_phase", 0)))
        phs.append(int(info.get("phase_idx", info.get("reversal_phase", 0))))
    return np.asarray(acts), np.asarray(stims), np.asarray(revs), np.asarray(phs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant_dir",
        type=str,
        required=True,
        help="Path to a single variant directory containing run_*/ subfolders.",
    )
    p.add_argument(
        "--task_pkl",
        type=str,
        required=True,
        help="Path to the task .pkl used for these runs (for trial_structure).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing plots.")
    args = p.parse_args()

    variant_dir = Path(args.variant_dir).resolve()
    task_pkl = Path(args.task_pkl).resolve()
    task = _load_pickle(task_pkl)
    trial_structure = task["trial_structure"]
    phase_boundaries = task.get("phase_boundaries", {}) or {}

    run_dirs = sorted([p for p in variant_dir.glob("run_*") if p.is_dir()])
    _log(f"Regenerating plots for {len(run_dirs)} runs in {variant_dir}")

    for rd in run_dirs:
        metrics_pkl = rd / "metrics_numpy.pkl"
        results_pkl = rd / "run_results.pkl"
        if not metrics_pkl.exists() or not results_pkl.exists():
            continue

        metrics = _load_pickle(metrics_pkl)
        run_results = _load_pickle(results_pkl)
        cfg = run_results.get("params", {})
        freeze_label = "regen"
        run_idx = 0
        seed = cfg.get("seed", None)
        plot_stims = [0, 1, 2, 3, 4, 5]

        # Per-run plots (skip if present unless --overwrite)
        def _maybe(path: Path):
            return args.overwrite or not path.exists()

        if _maybe(rd / "lick_value.png"):
            plot_lick_value(metrics, rd, cfg, plot_stims, run_idx, seed, freeze_label)
        if _maybe(rd / "grad_norms.png"):
            plot_grad_norms(metrics, rd, cfg, freeze_label, run_idx, seed, phase_boundaries)
        if _maybe(rd / "value_decoder.png"):
            plot_decoder(run_results["decoder_results"], rd, freeze_label)
        # TDR PSTH plots can be regenerated from stored run_results fields
        if _maybe(rd / "tdr_value_psth_pre_axis.png") or _maybe(rd / "tdr_value_psth_post_axis.png"):
            plot_tdr_value_psth(run_results, rd, cfg, {i: t for i, t in enumerate(trial_structure)}, plot_stims, freeze_label)

        # Phase generalisation matrices: reconstruct minimal `results` needed
        acts, stim, rev, ph = _stim_window_mean_activations(metrics, trial_structure)
        if acts.size:
            readout_size = int(cfg.get("readout_size", max(1, acts.shape[1] // 2)))
            pops = {
                "Full": acts,
                "Projecting": acts[:, :readout_size],
            }
            if readout_size < acts.shape[1]:
                pops["Non-proj"] = acts[:, readout_size:]
            # Reconstruct "Readout act" if weights were saved.
            rw = run_results.get("readout_weights", None)
            if rw is not None:
                ri = rw.get("readout_indices", list(range(readout_size)))
                proj = acts[:, ri]
                W_a = rw["actor_fc_weight"]
                b_a = rw["actor_fc_bias"]
                W_c = rw["critic_fc_weight"]
                b_c = rw["critic_fc_bias"]
                actor_logits = proj @ W_a.T + b_a
                critic_value = proj @ W_c.T + b_c
                pops["Readout act"] = np.hstack([actor_logits, critic_value])
                pops["Actor logits"] = actor_logits
                pops["Critic value"] = critic_value
            results_min = {
                "all_stim_labels": stim,
                "all_phase_labels": rev,
                "all_phase_idx": ph,
                "populations": pops,
                "PRE_HIGH": [0, 1],
                "PRE_LOW": [4, 5],
                "POST_HIGH": [1, 4],
                "POST_LOW": [0, 5],
            }
            plot_phase_generalisation_matrices(results_min, rd, freeze_label)

    _log("Done.")


if __name__ == "__main__":
    main()


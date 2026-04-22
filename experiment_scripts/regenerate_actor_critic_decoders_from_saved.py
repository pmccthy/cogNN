"""
Regenerate decoder bar plots and phase heatmaps for Actor/Critic readout activations.

This script does NOT rerun simulations. It recomputes decoding from:
- `metrics_numpy.pkl`  (hidden states per trial)
- `run_results.pkl`    (params + saved readout_weights, if present)
- task `.pkl`          (trial_structure, to align stimulus windows and phase_idx)

It produces:
- `value_decoder.png` (updated, includes Actor logits and Critic value populations)
- `decoder_phase_generalisation_matrix_*.png` (updated, includes those populations)

Limitations
-----------
- Requires `readout_weights` to be present in `run_results.pkl` to reconstruct
  Actor logits / Critic value / Readout act. Runs created before this field was
  saved cannot be regenerated without rerunning.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _stim_window_mean_hidden(metrics_numpy, trial_structure):
    """Return per-trial stim-window mean hidden state and labels.

    Returns
    -------
    h : (N, hidden) array
    stim : (N,) int in 0..5
    rev_phase : (N,) int in {0,1}
    phase_idx : (N,) int in 0..K-1
    """
    h_list, stim_list, rev_list, ph_list = [], [], [], []
    for tb in metrics_numpy["trial_boundaries"]:
        ti = tb["trial_idx"]
        hs = metrics_numpy["hidden_states"].get(ti)
        if not hs:
            continue
        info = trial_structure[ti]
        t0 = info["trial_start"]
        stim_rel = np.asarray(info["stim_window"]) - t0
        wt_ts = np.asarray(metrics_numpy["within_trial_timesteps"][ti])
        wt_rel = wt_ts - t0
        xs = []
        for tr in stim_rel:
            m = np.where(wt_rel == tr)[0]
            if len(m) and m[0] < len(hs):
                xs.append(np.asarray(hs[m[0]]).squeeze())
        if not xs:
            continue
        h_list.append(np.mean(np.stack(xs, axis=0), axis=0))
        stim_list.append(int(info["stimulus"]))
        rev_list.append(int(info.get("reversal_phase", 0)))
        ph_list.append(int(info.get("phase_idx", info.get("reversal_phase", 0))))
    return (
        np.asarray(h_list),
        np.asarray(stim_list),
        np.asarray(rev_list),
        np.asarray(ph_list),
    )


def _block_mask_phase_idx(stim_labels, phase_idx_labels, phase_idx, block, use_second_half=False):
    start, end = block
    mask = np.zeros(len(stim_labels), dtype=bool)
    for s in range(6):
        idx = np.where((stim_labels == s) & (phase_idx_labels == phase_idx))[0]
        if idx.size == 0:
            continue
        if use_second_half:
            idx = idx[idx.size // 2 :]
        mask[idx[start:end]] = True
    return mask


def _value_dataset(acts, stim_labels, rev_phase_labels, phase, high, low, conv_mask):
    m = (rev_phase_labels == phase) & np.isin(stim_labels, high + low) & conv_mask
    X = acts[m]
    y = np.isin(stim_labels[m], high).astype(int)
    return X, y


def _compute_decoder_results(populations, stim_labels, rev_phase_labels, phase_idx_labels, cfg):
    """Recompute decoder bar-plot numbers for all populations."""
    tdr_pre_blk = cfg.get("tdr_pre_block", (50, None))
    tdr_post_blk = cfg.get("tdr_post_block", (50, None))

    pre_mask = _block_mask_phase_idx(stim_labels, phase_idx_labels, 0, tdr_pre_blk, use_second_half=False)
    post_mask = _block_mask_phase_idx(stim_labels, phase_idx_labels, 1, tdr_post_blk, use_second_half=True)

    PRE_HIGH, PRE_LOW = [0, 1], [4, 5]
    POST_HIGH, POST_LOW = [1, 4], [0, 5]

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, acts in populations.items():
        if acts is None:
            continue
        X_pre, y_pre = _value_dataset(acts, stim_labels, rev_phase_labels, 0, PRE_HIGH, PRE_LOW, pre_mask)
        X_post, y_post = _value_dataset(acts, stim_labels, rev_phase_labels, 1, POST_HIGH, POST_LOW, post_mask)
        if X_pre.shape[0] < 20 or X_post.shape[0] < 20:
            continue
        clf = Pipeline(
            [("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42))]
        )
        cv_pre = cross_val_score(clf, X_pre, y_pre, cv=cv5, scoring="accuracy")
        cv_post = cross_val_score(clf, X_post, y_post, cv=cv5, scoring="accuracy")

        sc1 = StandardScaler()
        lr1 = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        lr1.fit(sc1.fit_transform(X_pre), y_pre)
        p2p = lr1.score(sc1.transform(X_post), y_post)

        sc2 = StandardScaler()
        lr2 = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        lr2.fit(sc2.fit_transform(X_post), y_post)
        p2pr = lr2.score(sc2.transform(X_pre), y_pre)

        out[name] = {"pre_cv": cv_pre, "post_cv": cv_post, "pre_to_post": p2p, "post_to_pre": p2pr}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant_dir",
        type=str,
        default="",
        help="Single variant directory (contains run_*/).",
    )
    p.add_argument(
        "--results_root",
        type=str,
        default="",
        help="Results root containing multiple variant dirs (each with config.csv).",
    )
    p.add_argument("--task_pkl", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.variant_dir and not args.results_root:
        raise ValueError("Pass --variant_dir or --results_root")
    if args.variant_dir and args.results_root:
        raise ValueError("Pass only one of --variant_dir or --results_root")

    task_pkl = Path(args.task_pkl).resolve()
    task = _load_pickle(task_pkl)
    trial_structure = task["trial_structure"]

    # Load plotting funcs from runner module.
    repo_root = Path(args.results_root or args.variant_dir).resolve()
    while not (repo_root / "experiment_scripts").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent
    runner_py = repo_root / "experiment_scripts" / "run_meta_ac_abcdef_partial_multirun.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_meta_ac_abcdef_partial_multirun", runner_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    if args.results_root:
        results_root = Path(args.results_root).resolve()
        variant_dirs = sorted(
            [p for p in results_root.iterdir() if p.is_dir() and (p / "config.csv").exists()]
        )
    else:
        variant_dirs = [Path(args.variant_dir).resolve()]

    for variant_dir in variant_dirs:
        run_dirs = sorted([p for p in variant_dir.glob("run_*") if p.is_dir()])
        mod._log(f"Recomputing actor/critic decoders for {len(run_dirs)} runs in {variant_dir.name}")

        for rd in run_dirs:
            metrics_pkl = rd / "metrics_numpy.pkl"
            results_pkl = rd / "run_results.pkl"
            if not metrics_pkl.exists() or not results_pkl.exists():
                continue

            if (rd / "value_decoder.png").exists() and not args.overwrite:
                # Still regenerate heatmaps if missing.
                pass

            metrics = _load_pickle(metrics_pkl)
            run_results = _load_pickle(results_pkl)
            cfg = run_results.get("params", {})

            rw = run_results.get("readout_weights")
            if rw is None:
                mod._log(f"  Skipping {rd.name}: missing readout_weights (need rerun)", indent=1)
                continue

            h, stim, rev, ph = _stim_window_mean_hidden(metrics, trial_structure)
            if h.size == 0:
                continue

            ri = rw.get("readout_indices", list(range(int(cfg.get("readout_size", 64)))))
            proj = h[:, ri]
            W_a, b_a = rw["actor_fc_weight"], rw["actor_fc_bias"]
            W_c, b_c = rw["critic_fc_weight"], rw["critic_fc_bias"]
            actor_logits = proj @ W_a.T + b_a
            critic_value = proj @ W_c.T + b_c

            # Reconstruct the full set of populations, then add actor/critic splits.
            readout_size = int(cfg.get("readout_size", len(ri)))
            pops = {
                "Full": h,
                "Projecting": h[:, :readout_size] if h.shape[1] >= readout_size else h,
            }
            if h.shape[1] > readout_size:
                pops["Non-proj"] = h[:, readout_size:]
            pops["Actor logits"] = actor_logits
            pops["Critic value"] = critic_value
            pops["Readout act"] = np.hstack([actor_logits, critic_value])

            # Decoder bar plot (recomputed)
            decoder_results = _compute_decoder_results(pops, stim, rev, ph, cfg)
            if decoder_results:
                mod.plot_decoder(decoder_results, rd, freeze_label="regen")

            # Heatmaps (reuse runner's plotter; it will include these populations)
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
            obj = mod.plot_phase_generalisation_matrices(results_min, rd, freeze_label="regen")
            if obj is not None:
                with open(rd / "decoder_phase_generalisation_matrices.pkl", "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    mod._log("Done.")


if __name__ == "__main__":
    main()


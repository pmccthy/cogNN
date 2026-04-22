"""
Resume incomplete variant runs and refresh aggregate plots.

This script is intended for long-running sweeps where some variants have partially
completed runs. It:
1) Regenerates aggregate plots from whatever run folders already exist.
2) Resumes training for missing seeds (without overwriting completed runs).
3) Regenerates aggregate plots again.

It operates on a *results root directory* that contains multiple variant folders,
each created by `run_meta_ac_abcdef_partial_multirun.py`.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import importlib.util
import pickle
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    root = here.parent
    while not (root / "experiment_scripts").exists() and root != root.parent:
        root = root.parent
    return root


def _load_runner_module():
    """Load run_meta_ac_abcdef_partial_multirun as a module by file path.

    This avoids issues when running from within experiment_scripts/ where
    `experiment_scripts` isn't importable as a package.
    """
    root = _repo_root()
    runner_py = root / "experiment_scripts" / "run_meta_ac_abcdef_partial_multirun.py"
    spec = importlib.util.spec_from_file_location("run_meta_ac_abcdef_partial_multirun", runner_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {runner_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_config_csv(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return cfg
        for row in r:
            if len(row) >= 2:
                cfg[row[0]] = row[1]
    return cfg


def _parse_bool_int(s: str) -> int:
    return 1 if str(s).strip().lower() in ("1", "true", "yes") else 0


def _parse_int(s: str, default: int) -> int:
    try:
        return int(float(s))
    except Exception:
        return default


def _parse_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _infer_task_from_variant_dirname(name: str) -> str | None:
    # variant_name = f'{task}_partialro{readout_size}_frozen{freeze_mode}{_pra_str}'
    token = "_partialro"
    if token not in name:
        return None
    return name.split(token, 1)[0]


def _freeze_label(freeze_mode: str) -> str:
    return {
        "none": "full plasticity",
        "readout_only": "readout only",
        "rnn_only": "RNN only",
        "all": "frozen",
    }.get(freeze_mode, freeze_mode)


def _run_dirs(variant_dir: Path) -> list[Path]:
    return sorted([p for p in variant_dir.glob("run_*") if p.is_dir()])


def _completed_runs(run_dirs: list[Path]) -> int:
    return sum(1 for rd in run_dirs if (rd / "run_results.pkl").exists())


def _refresh_aggregates(variant_dir: Path, freeze_label: str, plot_stims: list[int]) -> None:
    runner = _load_runner_module()
    _log = runner._log
    aggregate_plots = runner.aggregate_plots

    run_dirs = _run_dirs(variant_dir)
    agg_dir = variant_dir / "aggregate"
    _log(f"[refresh] {variant_dir.name}: {len(run_dirs)} run dirs, {_completed_runs(run_dirs)} completed")
    aggregate_plots(run_dirs, agg_dir, cfg={}, freeze_label=freeze_label, plot_stims=plot_stims)


def _stim_window_mean_activations(metrics_numpy, trial_structure, stim_window_len: int = 5):
    """Return (acts, stim_labels, reversal_phase_labels, phase_idx_labels)."""
    import numpy as np

    acts, stims, revs, phs = [], [], [], []
    for tb in metrics_numpy["trial_boundaries"]:
        ti = tb["trial_idx"]
        hs_list = metrics_numpy["hidden_states"].get(ti)
        if not hs_list:
            continue
        info = trial_structure[ti]
        t0 = info["trial_start"]
        stim_window = info.get("stim_window") or []
        stim_rel = [t - t0 for t in stim_window[:stim_window_len]]
        wt_ts = metrics_numpy["within_trial_timesteps"].get(ti)
        if wt_ts is None:
            continue
        wt_rel = [t - t0 for t in wt_ts]
        xs = []
        for tr in stim_rel:
            try:
                k = wt_rel.index(tr)
            except ValueError:
                continue
            if 0 <= k < len(hs_list):
                xs.append(np.asarray(hs_list[k]).squeeze())
        if not xs:
            continue
        acts.append(np.mean(np.stack(xs, axis=0), axis=0))
        stims.append(int(info["stimulus"]))
        revs.append(int(info.get("reversal_phase", 0)))
        phs.append(int(info.get("phase_idx", info.get("reversal_phase", 0))))
    if not acts:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
    return np.asarray(acts), np.asarray(stims), np.asarray(revs), np.asarray(phs)


def _ensure_new_style_heatmaps_for_variant(variant_dir: Path, task_data_dir: Path) -> None:
    """Regenerate new-style heatmap pkls/images from existing metrics if needed."""
    runner = _load_runner_module()

    task = _infer_task_from_variant_dirname(variant_dir.name)
    if task is None:
        return
    task_pkl = task_data_dir / f"{task}.pkl"
    if not task_pkl.exists():
        return
    task_data = _load_pickle(task_pkl)
    trial_structure = task_data.get("trial_structure", [])
    if not trial_structure:
        return

    for rd in _run_dirs(variant_dir):
        metrics_pkl = rd / "metrics_numpy.pkl"
        if not metrics_pkl.exists():
            continue
        out_pkl = rd / "decoder_phase_generalisation_matrices.pkl"

        needs_regen = True
        if out_pkl.exists():
            try:
                obj = _load_pickle(out_pkl)
                bp = (obj or {}).get("by_population", {})
                needs_regen = not any(
                    isinstance(v, dict) and ("phase_matrix" in v) for v in bp.values()
                )
            except Exception:
                needs_regen = True
        if not needs_regen:
            continue

        metrics = _load_pickle(metrics_pkl)
        acts, stim, rev, ph = _stim_window_mean_activations(metrics, trial_structure, stim_window_len=5)
        if getattr(acts, "size", 0) == 0 or acts.ndim != 2:
            continue

        # Infer readout_size from variant dirname if possible, else default 64.
        readout_size = 64
        if "_partialro" in variant_dir.name:
            try:
                readout_size = int(variant_dir.name.split("_partialro", 1)[1].split("_", 1)[0])
            except Exception:
                pass

        pops = {"Full": acts}
        pops["Projecting"] = acts[:, : min(readout_size, acts.shape[1])]
        if readout_size < acts.shape[1]:
            pops["Non-proj"] = acts[:, readout_size:]

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

        obj = runner.plot_phase_generalisation_matrices(results_min, rd, freeze_label="regen")
        if obj is not None:
            with open(out_pkl, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _resume_variant(
    runner_path: Path,
    variant_dir: Path,
    results_root: Path,
    n_runs_target: int,
    config: dict[str, str],
    dry_run: bool,
    run_indices: list[int] | None = None,
) -> None:
    freeze_mode = config.get("freeze_mode", "none")
    use_prev_ar = _parse_bool_int(config.get("use_prev_action_reward", "0"))
    hidden_size = _parse_int(config.get("hidden_size", "128"), 128)
    readout_size = _parse_int(config.get("readout_size", str(max(1, hidden_size // 2))), max(1, hidden_size // 2))
    lr = _parse_float(config.get("learning_rate", "5e-4"), 5e-4)
    gamma = _parse_float(config.get("gamma", "0.0"), 0.0)
    policy_clip = _parse_float(config.get("policy_clip", "0.25"), 0.25)

    task = _infer_task_from_variant_dirname(variant_dir.name) or config.get("task", None)
    if task is None:
        raise ValueError(f"Could not infer task from variant dir: {variant_dir.name}")

    readout_fraction = readout_size / max(hidden_size, 1)

    cmd = [
        sys.executable,
        str(runner_path),
        "--freeze_mode",
        freeze_mode,
        "--use_prev_action_reward",
        str(use_prev_ar),
        "--n_runs",
        str(n_runs_target),
        "--hidden_size",
        str(hidden_size),
        "--readout_fraction",
        str(readout_fraction),
        "--learning_rate",
        str(lr),
        "--gamma",
        str(gamma),
        "--policy_clip",
        str(policy_clip),
        "--task",
        task,
        "--results_dir",
        str(results_root),
    ]
    if run_indices:
        cmd += ["--run_indices", ",".join(str(i) for i in sorted(set(run_indices)))]
        cmd += ["--overwrite"]

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return

    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Results directory containing variant subfolders.",
    )
    p.add_argument(
        "--n_runs_target",
        type=int,
        default=10,
        help="Target number of runs per variant (runner will skip completed runs).",
    )
    p.add_argument(
        "--include_mid_stim",
        action="store_true",
        help="Include C/D in aggregate lick/value plots.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands but do not execute.",
    )
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    runner_path = (_repo_root() / "experiment_scripts" / "run_meta_ac_abcdef_partial_multirun.py").resolve()
    task_data_dir = (_repo_root() / "task_data").resolve()

    if not results_root.exists():
        raise FileNotFoundError(results_root)

    plot_stims = [0, 1, 2, 3, 4, 5] if args.include_mid_stim else [0, 1, 4, 5]

    variant_dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and (p / "config.csv").exists()])
    if not variant_dirs:
        print(f"No variant dirs with config.csv found under {results_root}")
        return

    # 1) Refresh aggregates from current data
    for vd in variant_dirs:
        cfg = _read_config_csv(vd / "config.csv")
        freeze_mode = cfg.get("freeze_mode", "none")
        _ensure_new_style_heatmaps_for_variant(vd, task_data_dir)
        _refresh_aggregates(vd, _freeze_label(freeze_mode), plot_stims)

    # 2) Resume missing runs (runner skips completed runs unless --overwrite is given)
    for vd in variant_dirs:
        cfg = _read_config_csv(vd / "config.csv")
        run_dirs = _run_dirs(vd)
        done = _completed_runs(run_dirs)
        # Identify which run indices need to be rerun specifically to add readout_weights.
        missing_readout_weights = []
        for rd in run_dirs:
            rr = rd / "run_results.pkl"
            if not rr.exists():
                continue
            try:
                obj = _load_pickle(rr)
            except Exception:
                continue
            if obj.get("readout_weights") is None:
                # Parse run index from folder name "run_XX_seedYY"
                try:
                    idx = int(rd.name.split("_")[1])
                    missing_readout_weights.append(idx)
                except Exception:
                    pass

        # First, rerun only the indices missing readout_weights (if any).
        if missing_readout_weights:
            _resume_variant(
                runner_path,
                vd,
                results_root,
                args.n_runs_target,
                cfg,
                args.dry_run,
                run_indices=missing_readout_weights,
            )

        # Then, resume any missing runs up to n_runs_target (no overwrite).
        if done >= args.n_runs_target:
            continue
        _resume_variant(runner_path, vd, results_root, args.n_runs_target, cfg, args.dry_run)

    # 3) Refresh aggregates again
    for vd in variant_dirs:
        cfg = _read_config_csv(vd / "config.csv")
        freeze_mode = cfg.get("freeze_mode", "none")
        _ensure_new_style_heatmaps_for_variant(vd, task_data_dir)
        _refresh_aggregates(vd, _freeze_label(freeze_mode), plot_stims)


if __name__ == "__main__":
    main()


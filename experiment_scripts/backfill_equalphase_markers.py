"""
Backfill marker indices into existing lick/value pkls.

This script adds keys used by `plot_train_then_infer_lick_value.py` for older runs:

- `readout_freeze_trial_index_global` in `run_*/train/lick_value_data.pkl`
  (for variants with readout freeze after N cycles)
- `prev_ar_switch_trial_index_global` in `run_*/infer_frozen/lick_value_data.pkl`
  (for staggered prev-A/R inference variants)

It infers indices by:
1) Loading each run's `metrics_numpy.pkl` (for `trial_boundaries` trial_start times)
2) Loading the task pkl from `config.csv` (for `phase_boundaries["phases"]` start times)
3) Converting the relevant switch timestep into a *global trial index*:
   the first entry in `trial_boundaries` whose `trial_start >= switch_ts`.

Usage
-----
~/.local/share/mamba/envs/cog_nn/bin/python3 experiment_scripts/backfill_equalphase_markers.py \
  --results_dir results/27_03_26_equalphase_multirev

Notes
-----
For `prev_ar_mode=train_on_infer_off`, the original run may have used a custom
`--infer_prev_ar_on_test_phases`. If you used a non-default value, pass it here
so backfilled switch markers match your runs.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_config_csv(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if len(row) >= 2:
                cfg[row[0]] = row[1]
    return cfg


def _infer_trial_index_at_timestep(metrics_numpy: dict, ts: int) -> int | None:
    tbs = metrics_numpy.get("trial_boundaries") or []
    for i, tb in enumerate(tbs):
        if int(tb.get("trial_start", 0)) >= int(ts):
            return int(i)
    return None


def _phase_start_timestep_after_phases(phase_boundaries: dict, n_phases: int) -> int | None:
    phases = (phase_boundaries or {}).get("phases") or []
    if not phases:
        return None
    if n_phases <= 0:
        return int(phases[0]["start"])
    if n_phases >= len(phases):
        return None
    return int(phases[int(n_phases)]["start"])


def _infer_prev_ar_switch_ts(
    phase_boundaries: dict,
    keep_prev_ar_on_test_phases: int,
) -> int | None:
    pb = phase_boundaries or {}
    n_train = int(pb.get("n_train_phases", 0))
    n_test = int(pb.get("n_test_phases", 0))
    phases = (pb.get("phases") or [])
    if n_train <= 0 or n_test <= 0 or not phases:
        return None
    keep = int(keep_prev_ar_on_test_phases)
    if keep < 0:
        keep = max(1, n_test // 2)
    switch_phase_idx = n_train + keep
    if 0 <= switch_phase_idx < len(phases):
        return int(phases[switch_phase_idx]["start"])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument(
        "--infer_prev_ar_on_test_phases",
        type=int,
        default=-1,
        help="Must match what you used when running staggered inference (-1 = half).",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    results_root = Path(args.results_dir).resolve()
    variant_dirs = sorted(
        [
            d
            for d in results_root.iterdir()
            if d.is_dir() and d.name.startswith("equalphase_multirev_") and (d / "config.csv").exists()
        ]
    )
    if not variant_dirs:
        raise FileNotFoundError(f"No variant dirs with config.csv under {results_root}")

    n_updated = 0
    n_scanned = 0

    for vd in variant_dirs:
        cfg = _read_config_csv(vd / "config.csv")
        task_pkl_s = cfg.get("task_pkl", "")
        if not task_pkl_s:
            continue
        task_pkl = Path(task_pkl_s)
        if not task_pkl.exists():
            continue
        task = _load_pickle(task_pkl)
        phase_boundaries = task.get("phase_boundaries") or {}
        phases = phase_boundaries.get("phases") or []

        # Infer behaviour from variant dirname / config
        prev_ar_mode = cfg.get("prev_ar_mode", "")
        variant_mode = cfg.get("variant_mode", "")
        # Support both old and new config keys
        freeze_phases_s = cfg.get("readout_freeze_after_phases", "")
        if not freeze_phases_s:
            # Older variants used *_after2cycles; best-effort: interpret as phases = 2*cycles
            freeze_cycles_s = cfg.get("readout_freeze_after_cycles", "")
            if freeze_cycles_s:
                try:
                    freeze_phases_s = str(2 * int(float(freeze_cycles_s)))
                except Exception:
                    freeze_phases_s = ""

        readout_freeze_after_phases = None
        if freeze_phases_s not in ("", "None", "none"):
            try:
                readout_freeze_after_phases = int(float(freeze_phases_s))
            except Exception:
                readout_freeze_after_phases = None

        readout_freeze_ts = None
        if readout_freeze_after_phases is not None:
            readout_freeze_ts = _phase_start_timestep_after_phases(
                phase_boundaries, readout_freeze_after_phases
            )

        prev_ar_switch_ts = None
        if prev_ar_mode == "train_on_infer_off":
            prev_ar_switch_ts = _infer_prev_ar_switch_ts(
                phase_boundaries, args.infer_prev_ar_on_test_phases
            )

        # Actor/critic staggered freeze markers (only for the special variant mode)
        critic_freeze_ts = None
        if variant_mode == "actor_frozen_critic_plastic_first_phase_only" and phases:
            try:
                critic_freeze_ts = int(phases[0]["end"]) + 1
            except Exception:
                critic_freeze_ts = None

        for rd in sorted([p for p in vd.glob("run_*") if p.is_dir()]):
            n_scanned += 1

            # Backfill readout freeze marker into train lv pkl
            if readout_freeze_ts is not None:
                train_metrics = rd / "train" / "metrics_numpy.pkl"
                train_lv = rd / "train" / "lick_value_data.pkl"
                if train_metrics.exists() and train_lv.exists():
                    mn = _load_pickle(train_metrics)
                    idx = _infer_trial_index_at_timestep(mn, readout_freeze_ts)
                    if idx is not None:
                        lv = _load_pickle(train_lv)
                        if lv.get("readout_freeze_trial_index_global") != idx:
                            lv["readout_freeze_trial_index_global"] = int(idx)
                            if not args.dry_run:
                                _save_pickle(train_lv, lv)
                            n_updated += 1

            # Backfill actor/critic freeze markers into train lv pkl
            if variant_mode == "actor_frozen_critic_plastic_first_phase_only":
                train_metrics = rd / "train" / "metrics_numpy.pkl"
                train_lv = rd / "train" / "lick_value_data.pkl"
                if train_metrics.exists() and train_lv.exists():
                    try:
                        mn = _load_pickle(train_metrics)
                        lv = _load_pickle(train_lv)
                        changed = False
                        if lv.get("actor_readout_freeze_trial_index_global") != 0:
                            lv["actor_readout_freeze_trial_index_global"] = 0
                            changed = True
                        if critic_freeze_ts is not None:
                            idx = _infer_trial_index_at_timestep(mn, critic_freeze_ts)
                            if idx is not None and lv.get("critic_readout_freeze_trial_index_global") != idx:
                                lv["critic_readout_freeze_trial_index_global"] = int(idx)
                                changed = True
                        if changed:
                            if not args.dry_run:
                                _save_pickle(train_lv, lv)
                            n_updated += 1
                    except Exception:
                        pass

            # Backfill prev A/R switch marker into inference lv pkl
            if prev_ar_switch_ts is not None:
                infer_metrics = rd / "infer_frozen" / "metrics_numpy.pkl"
                infer_lv = rd / "infer_frozen" / "lick_value_data.pkl"
                if infer_metrics.exists() and infer_lv.exists():
                    mn = _load_pickle(infer_metrics)
                    idx = _infer_trial_index_at_timestep(mn, prev_ar_switch_ts)
                    if idx is not None:
                        lv = _load_pickle(infer_lv)
                        if lv.get("prev_ar_switch_trial_index_global") != idx:
                            lv["prev_ar_switch_trial_index_global"] = int(idx)
                            if not args.dry_run:
                                _save_pickle(infer_lv, lv)
                            n_updated += 1

    print(f"Scanned run dirs: {n_scanned}")
    print(f"Updated lv files: {n_updated}")


if __name__ == "__main__":
    main()


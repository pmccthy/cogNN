"""
Plot training then frozen-inference lick/value on one timeline.

This script is designed for results produced by:
`experiment_scripts/run_meta_ac_abcdef_equalphase_multirev_interleaved.py`

For each variant directory, it loads:
- `run_XX_seedYY/train/lick_value_data.pkl`
- `run_XX_seedYY/infer_frozen/lick_value_data.pkl`

and produces a single 2-panel plot where the inference traces follow immediately
after the training traces on the same x-axis (global trial index within each
segment, with inference offset by the final train x).

Outputs
-------
Per run:
  run_XX_seedYY/combined_train_then_infer_lick_value.png

Per variant aggregate:
  aggregate/combined_train_then_infer_lick_value.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_scripts.run_meta_ac_abcdef_partial_multirun import (  # noqa: E402
    STIM_COLORS_DARK,
    STIM_NAMES_MAP,
)

import cog_nn.plot_style  # noqa: F401  (apply mpl style)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _series_xy(lv: dict, k: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(lv[metric][k], dtype=float)
    x_map = lv.get("global_trial_indices_by_stim") or {}
    if k in x_map:
        x = np.asarray(x_map[k], dtype=float)[: len(y)]
    else:
        x = np.arange(len(y), dtype=float)
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _smooth(y: np.ndarray, win: int) -> np.ndarray:
    if y.size == 0 or win <= 1:
        return y
    # Simple moving average with edge handling
    w = np.ones(win, dtype=float) / float(win)
    ypad = np.pad(y, (win - 1, 0), mode="edge")
    return np.convolve(ypad, w, mode="valid")


def _plot_combined(
    lv_train: dict,
    lv_infer: dict,
    out_path: Path,
    title: str,
    plot_stims: list[int],
    smooth: int = 20,
    individual_train: list[dict] | None = None,
    individual_infer: list[dict] | None = None,
) -> None:
    stim_keys = ["ABCDEF"[s] for s in plot_stims]
    stim_idx = {k: i for i, k in enumerate("ABCDEF")}

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.0), sharex=True)
    metrics = [("trial_lick_probs", "Lick probability"), ("trial_values", "Value estimate")]

    # Offset inference x so it starts immediately after training ends.
    train_xmax = 0.0
    for k in stim_keys:
        x, _ = _series_xy(lv_train, k, "trial_lick_probs")
        if x.size:
            train_xmax = max(train_xmax, float(np.max(x)))
    infer_offset = train_xmax + 1.0

    # Phase-change markers (use rev_indices_global if present)
    train_phase_marks = [int(x) for x in (lv_train.get("rev_indices_global") or [])]
    infer_phase_marks = [int(x) for x in (lv_infer.get("rev_indices_global") or [])]
    prev_ar_switch_idx = lv_infer.get("prev_ar_switch_trial_index_global", None)
    readout_freeze_idx = lv_train.get("readout_freeze_trial_index_global", None)
    actor_freeze_idx = lv_train.get("actor_readout_freeze_trial_index_global", None)
    critic_freeze_idx = lv_train.get("critic_readout_freeze_trial_index_global", None)
    # Use a single linestyle for all reversal markers for readability.
    phase_marker_ls = ":"

    for panel_i, (ax, (metric, ylab)) in enumerate(zip(axes, metrics)):
        # Thin per-run traces (optional)
        if individual_train is not None and individual_infer is not None:
            for lvtr, lvin in zip(individual_train, individual_infer):
                # Compute per-run offset in the same way as the mean plot
                _txmax = 0.0
                for k in stim_keys:
                    _x, _ = _series_xy(lvtr, k, "trial_lick_probs")
                    if _x.size:
                        _txmax = max(_txmax, float(np.max(_x)))
                _off = _txmax + 1.0

                for k in stim_keys:
                    si = stim_idx[k]
                    col = STIM_COLORS_DARK[si]
                    ls = "-" if si in (0, 4) else "--"
                    xtr, ytr = _series_xy(lvtr, k, metric)
                    xin, yin = _series_xy(lvin, k, metric)
                    ytr_s = _smooth(ytr, smooth)
                    yin_s = _smooth(yin, smooth)
                    if xtr.size and ytr_s.size:
                        ax.plot(xtr, ytr_s, color=col, lw=0.8, ls=ls, alpha=0.18, zorder=1)
                    if xin.size and yin_s.size:
                        ax.plot(
                            xin + _off,
                            yin_s,
                            color=col,
                            lw=0.8,
                            ls=ls,
                            alpha=0.18,
                            zorder=1,
                        )

        for k in stim_keys:
            si = stim_idx[k]
            col = STIM_COLORS_DARK[si]
            ls = "-" if si in (0, 4) else "--"

            xtr, ytr = _series_xy(lv_train, k, metric)
            xin, yin = _series_xy(lv_infer, k, metric)
            ytr_s = _smooth(ytr, smooth)
            yin_s = _smooth(yin, smooth)

            if xtr.size and ytr_s.size:
                ax.plot(
                    xtr,
                    ytr_s,
                    color=col,
                    lw=2.2,
                    ls=ls,
                    alpha=0.95,
                    label=STIM_NAMES_MAP[si],
                    zorder=3,
                )
            if xin.size and yin_s.size:
                ax.plot(
                    xin + infer_offset,
                    yin_s,
                    color=col,
                    lw=2.2,
                    ls=ls,
                    alpha=0.95,
                    zorder=3,
                )

        # Mark phase boundaries in train
        for i, x0 in enumerate(train_phase_marks):
            ax.axvline(
                float(x0),
                color="k",
                lw=1.0,
                alpha=0.45,
                linestyle=phase_marker_ls,
            )
            ax.text(
                float(x0),
                1.01,
                f"T{i+1}",
                fontsize=6,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                color="k",
            )
        # Mark phase boundaries in inference (offset on x)
        for i, x0 in enumerate(infer_phase_marks):
            ax.axvline(
                float(x0) + infer_offset,
                color="k",
                lw=1.0,
                alpha=0.45,
                linestyle=phase_marker_ls,
            )
            ax.text(
                float(x0) + infer_offset,
                1.01,
                f"I{i+1}",
                fontsize=6,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                color="k",
            )
        # Readout freeze markers.
        # If actor/critic-specific markers exist, place them on the relevant panel.
        if actor_freeze_idx is not None and np.isfinite(float(actor_freeze_idx)) and panel_i == 0:
            xr = float(actor_freeze_idx)
            ax.axvline(xr, color="k", lw=2.2, alpha=0.95, linestyle="--")
            ax.text(
                xr,
                1.06,
                "actor frozen",
                fontsize=7,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                color="k",
            )
        if critic_freeze_idx is not None and np.isfinite(float(critic_freeze_idx)) and panel_i == 1:
            xr = float(critic_freeze_idx)
            ax.axvline(xr, color="k", lw=2.2, alpha=0.95, linestyle="--")
            ax.text(
                xr,
                1.06,
                "critic frozen",
                fontsize=7,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                color="k",
            )
        # Fallback single-marker behaviour.
        if (
            (actor_freeze_idx is None and critic_freeze_idx is None)
            and readout_freeze_idx is not None
            and np.isfinite(float(readout_freeze_idx))
        ):
            xr = float(readout_freeze_idx)
            ax.axvline(xr, color="k", lw=2.2, alpha=0.95, linestyle="--")
            ax.text(
                xr,
                1.06,
                "readout frozen",
                fontsize=7,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                color="k",
            )

        # Boundary between train and infer (plasticity off) — bold + labelled
        xb = infer_offset - 0.5
        ax.axvline(xb, color="k", ls="-", lw=2.4, alpha=0.95)
        ax.text(
            xb,
            1.06,
            "plasticity off",
            fontsize=7,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            color="k",
        )
        if prev_ar_switch_idx is not None and np.isfinite(float(prev_ar_switch_idx)):
            xs = float(prev_ar_switch_idx) + infer_offset
            ax.axvline(xs, color="k", lw=2.2, alpha=0.95, linestyle=(0, (1, 3)))
            ax.text(
                xs,
                1.06,
                "prev A/R off",
                fontsize=7,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                color="k",
            )
        ax.set_ylabel(ylab)
        ax.set_xlim(left=0)

    axes[-1].set_xlabel("Global trial index (train then infer offset)")

    # Shared legend outside panels
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=8,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root results directory containing variant folders.",
    )
    p.add_argument(
        "--exclude_mid_stim",
        type=int,
        default=1,
        help="1=exclude C/D, 0=include all stimuli.",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Moving-average window in trials.",
    )
    args = p.parse_args()

    results_root = Path(args.results_dir).resolve()
    plot_stims = [0, 1, 4, 5] if args.exclude_mid_stim else list(range(6))

    variant_dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("equalphase_multirev_")])
    if not variant_dirs:
        raise FileNotFoundError(f"No variant dirs found under {results_root}")

    for vd in variant_dirs:
        run_dirs = sorted([p for p in vd.glob("run_*") if p.is_dir()])
        if not run_dirs:
            continue

        # Per-run plots + collect for aggregate
        agg_train, agg_infer = [], []

        for rd in run_dirs:
            p_train = rd / "train" / "lick_value_data.pkl"
            p_infer = rd / "infer_frozen" / "lick_value_data.pkl"
            if not (p_train.exists() and p_infer.exists()):
                continue
            lv_train = _load_pickle(p_train)
            lv_infer = _load_pickle(p_infer)
            agg_train.append(lv_train)
            agg_infer.append(lv_infer)

            out_path = rd / "combined_train_then_infer_lick_value.png"
            title = f"{vd.name}\n{rd.name}  |  train then infer (plasticity off)"
            _plot_combined(
                lv_train,
                lv_infer,
                out_path,
                title=title,
                plot_stims=plot_stims,
                smooth=int(args.smooth),
            )

        # Aggregate plot (mean across runs)
        if not agg_train:
            continue

        def _mean_lv(lvs: list[dict], metric: str, k: str) -> tuple[np.ndarray, np.ndarray]:
            # Align by x index and take nanmean
            series = []
            xs_all = []
            for lv in lvs:
                x, y = _series_xy(lv, k, metric)
                if x.size == 0:
                    continue
                xs_all.append(x)
                series.append((x, _smooth(y, int(args.smooth))))
            if not series:
                return np.array([]), np.array([])
            x_union = np.unique(np.concatenate(xs_all))
            x_union.sort()
            mat = np.full((len(series), len(x_union)), np.nan, dtype=float)
            for i, (x, y) in enumerate(series):
                # map x positions to union
                idx = np.searchsorted(x_union, x)
                mat[i, idx] = y
            return x_union, np.nanmean(mat, axis=0)

        # Build mean lv dicts on the fly (only what _plot_combined needs)
        lv_train_mean = {
            "trial_lick_probs": {},
            "trial_values": {},
            "global_trial_indices_by_stim": {},
            "rev_indices_global": [],
        }
        lv_infer_mean = {
            "trial_lick_probs": {},
            "trial_values": {},
            "global_trial_indices_by_stim": {},
            "rev_indices_global": [],
        }
        for k in "ABCDEF":
            for metric in ("trial_lick_probs", "trial_values"):
                x, y = _mean_lv(agg_train, metric, k)
                lv_train_mean[metric][k] = y
                lv_train_mean["global_trial_indices_by_stim"][k] = x
                x, y = _mean_lv(agg_infer, metric, k)
                lv_infer_mean[metric][k] = y
                lv_infer_mean["global_trial_indices_by_stim"][k] = x

        # Aggregate phase markers: use median across runs (robust to small mismatches)
        def _median_markers(lvs: list[dict]) -> list[int]:
            lists = [lv.get("rev_indices_global") for lv in lvs if lv.get("rev_indices_global") is not None]
            lists = [list(map(int, xs)) for xs in lists if isinstance(xs, (list, tuple)) and len(xs) > 0]
            if not lists:
                return []
            n = min(len(xs) for xs in lists)
            if n <= 0:
                return []
            out = []
            for i in range(n):
                vals = [xs[i] for xs in lists if len(xs) > i]
                out.append(int(np.median(vals)))
            return out

        lv_train_mean["rev_indices_global"] = _median_markers(agg_train)
        lv_infer_mean["rev_indices_global"] = _median_markers(agg_infer)

        # Aggregate prev A/R switch marker (only present for staggered variants)
        sw = [lv.get("prev_ar_switch_trial_index_global") for lv in agg_infer if lv.get("prev_ar_switch_trial_index_global") is not None]
        if sw:
            try:
                lv_infer_mean["prev_ar_switch_trial_index_global"] = int(np.median([int(x) for x in sw]))
            except Exception:
                pass

        # Aggregate readout freeze marker(s)
        af = [
            lv.get("actor_readout_freeze_trial_index_global")
            for lv in agg_train
            if lv.get("actor_readout_freeze_trial_index_global") is not None
        ]
        if af:
            try:
                lv_train_mean["actor_readout_freeze_trial_index_global"] = int(
                    np.median([int(x) for x in af])
                )
            except Exception:
                pass
        cf = [
            lv.get("critic_readout_freeze_trial_index_global")
            for lv in agg_train
            if lv.get("critic_readout_freeze_trial_index_global") is not None
        ]
        if cf:
            try:
                lv_train_mean["critic_readout_freeze_trial_index_global"] = int(
                    np.median([int(x) for x in cf])
                )
            except Exception:
                pass

        # Aggregate readout freeze marker (fallback single-marker)
        rf = [
            lv.get("readout_freeze_trial_index_global")
            for lv in agg_train
            if lv.get("readout_freeze_trial_index_global") is not None
        ]
        if rf:
            try:
                lv_train_mean["readout_freeze_trial_index_global"] = int(
                    np.median([int(x) for x in rf])
                )
            except Exception:
                pass

        out_path = vd / "aggregate" / "combined_train_then_infer_lick_value.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_combined(
            lv_train_mean,
            lv_infer_mean,
            out_path,
            title=f"{vd.name}\nAggregate mean across runs  |  train then infer (plasticity off)",
            plot_stims=plot_stims,
            smooth=1,  # already smoothed above
            individual_train=agg_train,
            individual_infer=agg_infer,
        )


if __name__ == "__main__":
    main()


"""
Interleaved multi-variant runner for equal-phase ABCDEF multi-reversal experiments.

This script is a companion to `run_meta_ac_abcdef_partial_multirun.py`, but it:
- Uses an *equal-phase* multi-reversal task pkl with an explicit train/test split.
- Trains on the training phases, then runs an inference/test phase where plasticity
  is frozen (no parameter updates) while stepping through the remaining phases.
- Runs multiple variants in an interleaved fashion: run_00 for every variant, then
  run_01 for every variant, etc.

Variants (default = 12)
----------------------
We parameterise variants along:
- **readout_freeze_after_phases**: when to freeze readout (phase count).
- **prev_action_reward_mode**:
    - "on":  prev action/reward inputs used in both train and inference.
    - "off": prev action/reward inputs disabled in both train and inference.
    - "train_on_infer_off": used during training, disabled during inference.

This yields 4 × 3 = 12 variants by default (for the default `full12` sweep).

Outputs
-------
Creates one variant folder per configuration under `--results_dir`, each with:
- `run_XX_seedYY/` containing:
  - `train/` plots + pkls
  - `infer_frozen/` plots + pkls (frozen plasticity)
- `aggregate/` containing aggregate plots for train and inference.

Notes
-----
This runner reuses plotting utilities from `run_meta_ac_abcdef_partial_multirun.py`
so plot styling changes propagate automatically.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cog_nn.tasks.reversal_envs import (  # noqa: E402
    ReversalABCDEFMultiTimestepEnv,
    load_reversal_abcdef_multitimestep_multirev_data,
)
from cog_nn.models import RNNActorCriticPartialReadout  # noqa: E402
from cog_nn.agents import MetaA2CAgent  # noqa: E402

# Reuse plotting utilities from the existing runner.
from experiment_scripts.run_meta_ac_abcdef_partial_multirun import (  # noqa: E402
    _compute_reversal_global_indices,
    _global_trial_indices_per_stim,
    _log,
    analyse,
    aggregate_plots,
    plot_decoder,
    plot_grad_norms,
    plot_lick_value,
    plot_reward_consumed,
    plot_aux_iti_accuracy,
    plot_phase_generalisation_matrices,
    plot_tdr_value_psth,
)

# Optional: combined train→infer lick/value plot for every run.
try:
    from experiment_scripts.plot_train_then_infer_lick_value import _plot_combined as _plot_combined_train_infer  # noqa: E402
except Exception:  # pragma: no cover
    _plot_combined_train_infer = None


def _make_aggregate_combined_plot(variant_dir: Path, plot_stims: list[int]) -> None:
    """Create aggregate combined train→infer plot for a variant directory."""
    if _plot_combined_train_infer is None:
        return

    run_dirs = sorted([p for p in variant_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return

    lv_tr_list, lv_in_list = [], []
    for rd in run_dirs:
        p_tr = rd / "train" / "lick_value_data.pkl"
        p_in = rd / "infer_frozen" / "lick_value_data.pkl"
        if not (p_tr.exists() and p_in.exists()):
            continue
        try:
            with open(p_tr, "rb") as f:
                lv_tr_list.append(pickle.load(f))
            with open(p_in, "rb") as f:
                lv_in_list.append(pickle.load(f))
        except Exception:
            continue

    if not lv_tr_list:
        return

    stim_keys = ["ABCDEF"[s] for s in plot_stims]

    def _series_xy(lv: dict, k: str, metric: str):
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
        w = np.ones(win, dtype=float) / float(win)
        ypad = np.pad(y, (win - 1, 0), mode="edge")
        return np.convolve(ypad, w, mode="valid")

    def _mean_lv(lvs: list[dict], metric: str, k: str, smooth_win: int = 20):
        series = []
        xs_all = []
        for lv in lvs:
            x, y = _series_xy(lv, k, metric)
            if x.size == 0:
                continue
            xs_all.append(x)
            series.append((x, _smooth(y, smooth_win)))
        if not series:
            return np.array([]), np.array([])
        x_union = np.unique(np.concatenate(xs_all))
        x_union.sort()
        mat = np.full((len(series), len(x_union)), np.nan, dtype=float)
        for i, (x, y) in enumerate(series):
            idx = np.searchsorted(x_union, x)
            mat[i, idx] = y
        return x_union, np.nanmean(mat, axis=0)

    def _median_markers(lvs: list[dict]) -> list[int]:
        lists = [
            lv.get("rev_indices_global")
            for lv in lvs
            if lv.get("rev_indices_global") is not None
        ]
        lists = [
            list(map(int, xs))
            for xs in lists
            if isinstance(xs, (list, tuple)) and len(xs) > 0
        ]
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

    lv_train_mean = {
        "trial_lick_probs": {},
        "trial_values": {},
        "global_trial_indices_by_stim": {},
        "rev_indices_global": _median_markers(lv_tr_list),
    }
    lv_infer_mean = {
        "trial_lick_probs": {},
        "trial_values": {},
        "global_trial_indices_by_stim": {},
        "rev_indices_global": _median_markers(lv_in_list),
    }
    for k in "ABCDEF":
        for metric in ("trial_lick_probs", "trial_values"):
            x, y = _mean_lv(lv_tr_list, metric, k, smooth_win=20)
            lv_train_mean[metric][k] = y
            lv_train_mean["global_trial_indices_by_stim"][k] = x
            x, y = _mean_lv(lv_in_list, metric, k, smooth_win=20)
            lv_infer_mean[metric][k] = y
            lv_infer_mean["global_trial_indices_by_stim"][k] = x

    # Include aggregate special markers if present in any run
    sw = [
        lv.get("prev_ar_switch_trial_index_global")
        for lv in lv_in_list
        if lv.get("prev_ar_switch_trial_index_global") is not None
    ]
    if sw:
        try:
            lv_infer_mean["prev_ar_switch_trial_index_global"] = int(
                np.median([int(x) for x in sw])
            )
        except Exception:
            pass
    rf = [
        lv.get("readout_freeze_trial_index_global")
        for lv in lv_tr_list
        if lv.get("readout_freeze_trial_index_global") is not None
    ]
    if rf:
        try:
            lv_train_mean["readout_freeze_trial_index_global"] = int(
                np.median([int(x) for x in rf])
            )
        except Exception:
            pass

    out_dir = variant_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_combined_train_infer(
        lv_train=lv_train_mean,
        lv_infer=lv_infer_mean,
        out_path=out_dir / "combined_train_then_infer_lick_value.png",
        title=f"{variant_dir.name}\nAggregate mean across runs  |  train then infer (plasticity off)",
        plot_stims=plot_stims,
        smooth=1,  # already smoothed in mean computation
        individual_train=lv_tr_list,
        individual_infer=lv_in_list,
    )


PrevARMode = Literal["on", "off", "train_on_infer_off"]
VariantMode = Literal[
    "baseline",
    "fixed_policy_always_lick",
    "actor_frozen_critic_plastic_first_phase_only",
]

RewardConsumptionMode = Literal["sampled", "expected"]


@dataclass(frozen=True)
class Variant:
    """Container for an experiment variant."""

    name: str
    prev_ar_mode: PrevARMode
    # None = never freeze; 0 = frozen from start; 1 = after first contingency block;
    # 2 = after one full cycle (pre+post); etc.
    readout_freeze_after_phases: int | None
    mode: VariantMode = "baseline"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    root = here.parent
    while not (root / "experiment_scripts").exists() and root != root.parent:
        root = root.parent
    return root


def _load_task(task_pkl: Path):
    return load_reversal_abcdef_multitimestep_multirev_data(task_pkl)


def _build_model(state_size: int, action_size: int, cfg: dict) -> RNNActorCriticPartialReadout:
    hidden_size = int(cfg["hidden_size"])
    readout_size = int(cfg["readout_size"])
    lr = float(cfg["learning_rate"])
    gamma = float(cfg["gamma"])

    model = RNNActorCriticPartialReadout(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        readout_indices=list(range(readout_size)),
    )
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.gamma = gamma
    # Optional auxiliary ITI prediction heads (full hidden state readout)
    if cfg.get("aux_iti_predict", False):
        n_stim = int(cfg.get("aux_n_stim_classes", 6))
        model.aux_stim = torch.nn.Linear(hidden_size, n_stim)
        model.aux_reward = torch.nn.Linear(hidden_size, 1)
        # Ensure optimiser sees new params if heads added after optimiser creation
        model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model


def _freeze_readout_params(model: RNNActorCriticPartialReadout) -> None:
    for name, p in model.named_parameters():
        if name.startswith("actor_fc") or name.startswith("critic_fc"):
            p.requires_grad = False


def _freeze_actor_readout_params(model: RNNActorCriticPartialReadout) -> None:
    for name, p in model.named_parameters():
        if name.startswith("actor_fc"):
            p.requires_grad = False


def _freeze_critic_readout_params(model: RNNActorCriticPartialReadout) -> None:
    for name, p in model.named_parameters():
        if name.startswith("critic_fc"):
            p.requires_grad = False


def _freeze_all_params(model: RNNActorCriticPartialReadout) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _phase_start_timestep_after_phases(phase_boundaries: dict, n_phases: int) -> int | None:
    """Return timestep at which to freeze readout after N phases."""
    phases = (phase_boundaries or {}).get("phases") or []
    if not phases:
        return None
    if n_phases <= 0:
        return int(phases[0]["start"])
    if n_phases >= len(phases):
        return None
    return int(phases[int(n_phases)]["start"])


def _trial_structure_local_phase_idx(
    trial_structure: list[dict],
    active_trial_indices: set[int],
) -> list[dict]:
    """Shift ``phase_idx`` so the minimum phase in the active segment is 0.

    ``analyse()`` selects TDR windows using global ``phase_idx`` 0 and 1 (first pre/post
    block in the *passed* trial_structure). For test-only metrics, global indices are
    e.g. 10–19; remapping makes phase 0/1 refer to the first pre/post of the test split.
    """
    if not active_trial_indices:
        return trial_structure
    pmin = min(int(trial_structure[i]["phase_idx"]) for i in active_trial_indices)
    out: list[dict] = []
    for i, td in enumerate(trial_structure):
        d = dict(td)
        if i in active_trial_indices:
            d["phase_idx"] = int(td["phase_idx"]) - pmin
        out.append(d)
    return out


def _split_trial_structure(trial_structure: list[dict], n_train_phases: int) -> tuple[list[int], list[int]]:
    """Return (train_trial_indices, test_trial_indices) based on trial phase_idx."""
    train_idx, test_idx = [], []
    for i, td in enumerate(trial_structure):
        ph = int(td.get("phase_idx", td.get("reversal_phase", 0)))
        (train_idx if ph < n_train_phases else test_idx).append(i)
    return train_idx, test_idx


def _filter_metrics_to_trials(
    metrics_numpy: dict,
    trial_indices: set[int],
    trial_structure: list[dict],
) -> dict:
    """Filter metrics dict to only include selected trials (by trial_idx).

    Keeps per-trial arrays aligned: ``trial_lick_probs`` / ``trial_values`` are rebuilt by
    walking *all* trials in order (using full ``trial_boundaries``) so per-stimulus
    occurrence indices match the original recording order. Timestep-level series
    (``lick_probs``, ``values``, ``timesteps`` per stimulus; ``rewards``) are filtered
    to timesteps that fall inside a selected trial.

    Args:
        metrics_numpy: Full rollout metrics (before filtering).
        trial_indices: Trial indices to retain.
        trial_structure: Full task trial list (for timestep membership).
    """
    tb_all = list(metrics_numpy["trial_boundaries"])
    tb = [x for x in tb_all if int(x["trial_idx"]) in trial_indices]

    def _filter_map(m):
        return {k: v for k, v in m.items() if int(k) in trial_indices}

    allowed_ts: set[int] = set()
    for td in trial_structure:
        if int(td["trial_idx"]) not in trial_indices:
            continue
        a, b = int(td["trial_start"]), int(td["trial_end"])
        allowed_ts.update(range(a, b + 1))

    out = dict(metrics_numpy)
    out["trial_boundaries"] = tb
    for key in (
        "within_trial_lick_probs",
        "within_trial_values",
        "within_trial_timesteps",
        "within_trial_states",
        "hidden_states",
    ):
        if key in out:
            out[key] = _filter_map(out[key])

    # Per-stimulus trial-end summaries: indices align with occurrence order of each stimulus
    # across *all* trials; rebuild by walking trials in trial_idx order.
    occ = {k: 0 for k in "ABCDEF"}
    new_tlp = {k: [] for k in "ABCDEF"}
    new_tv = {k: [] for k in "ABCDEF"}
    for x in sorted(tb_all, key=lambda z: int(z["trial_idx"])):
        ti = int(x["trial_idx"])
        stim = int(x["stimulus"])
        if not (0 <= stim < 6):
            continue
        k = "ABCDEF"[stim]
        o = occ[k]
        if ti in trial_indices:
            tlp_a = np.asarray(metrics_numpy["trial_lick_probs"][k])
            tv_a = np.asarray(metrics_numpy["trial_values"][k])
            if o < len(tlp_a):
                new_tlp[k].append(float(tlp_a[o]))
            if o < len(tv_a):
                new_tv[k].append(float(tv_a[o]))
        occ[k] += 1
    for k in "ABCDEF":
        out["trial_lick_probs"][k] = np.asarray(new_tlp[k], dtype=float)
        out["trial_values"][k] = np.asarray(new_tv[k], dtype=float)

    # Parallel arrays: one row per trial (stim-window end), same order as full rollout
    tix = np.asarray(metrics_numpy["trial_indices"], dtype=int)
    if tix.size:
        mask = np.array([int(i) in trial_indices for i in tix], dtype=bool)
        out["trial_timesteps"] = np.asarray(metrics_numpy["trial_timesteps"])[mask]
        out["trial_indices"] = tix[mask]
        out["trial_reversal_phases"] = np.asarray(metrics_numpy["trial_reversal_phases"])[mask]
    else:
        out["trial_timesteps"] = np.array([], dtype=float)
        out["trial_indices"] = np.array([], dtype=int)
        out["trial_reversal_phases"] = np.array([], dtype=int)

    # Timestep-level stimulus streams: keep only timesteps inside selected trials
    for k in "ABCDEF":
        ts = np.asarray(metrics_numpy["timesteps"][k], dtype=int)
        if ts.size == 0:
            out["lick_probs"][k] = np.array([], dtype=float)
            out["values"][k] = np.array([], dtype=float)
            out["timesteps"][k] = np.array([], dtype=int)
            continue
        mask = np.array([int(t) in allowed_ts for t in ts], dtype=bool)
        out["timesteps"][k] = ts[mask]
        out["lick_probs"][k] = np.asarray(metrics_numpy["lick_probs"][k])[mask]
        out["values"][k] = np.asarray(metrics_numpy["values"][k])[mask]

    rt = np.asarray(metrics_numpy.get("reward_timesteps", []), dtype=int)
    rw = np.asarray(metrics_numpy.get("rewards", []), dtype=float)
    if rt.size and rw.size == rt.size:
        rmask = np.array([int(t) in allowed_ts for t in rt], dtype=bool)
        out["reward_timesteps"] = rt[rmask]
        out["rewards"] = rw[rmask]
    elif rt.size == 0:
        out["reward_timesteps"] = np.array([], dtype=int)
        out["rewards"] = np.array([], dtype=float)

    return out


def _run_segment(
    *,
    env: ReversalABCDEFMultiTimestepEnv,
    model: RNNActorCriticPartialReadout,
    trial_structure: list[dict],
    phase_boundaries: dict,
    seed: int,
    train_end_timestep: int,
    update_enabled_until: int,
    use_prev_ar: bool,
    prev_ar_switch_at_ts: int | None = None,
    readout_freeze_at_ts: int | None,
    fixed_policy_always_lick: bool = False,
    actor_frozen: bool = False,
    critic_plastic_until_ts: int | None = None,
    reward_scale: float = 1.0,
    reward_consumption: RewardConsumptionMode = "sampled",
    aux_iti_predict: bool = False,
    aux_stim_weight: float = 1.0,
    aux_reward_weight: float = 1.0,
    log_prefix: str = "",
    log_every: int = 50_000,
) -> tuple[dict, RNNActorCriticPartialReadout, list[dict]]:
    """Run the full environment sequence, but update only while enabled.

    We always step through the whole sequence so that trial indices/timesteps remain
    compatible with existing plotting/analysis utilities. Plasticity is disabled by
    skipping updates beyond `update_enabled_until`.
    """
    hidden_size = int(model.rnn.hidden_size)
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    agent = MetaA2CAgent(state_size=state_size, action_size=action_size, hidden_size=hidden_size)
    agent.model = model

    metrics = {
        "lick_probs": {k: [] for k in "ABCDEF"},
        "values": {k: [] for k in "ABCDEF"},
        "timesteps": {k: [] for k in "ABCDEF"},
        "trial_lick_probs": {k: [] for k in "ABCDEF"},
        "trial_values": {k: [] for k in "ABCDEF"},
        "trial_timesteps": [],
        "trial_indices": [],
        "trial_reversal_phases": [],
        "within_trial_lick_probs": {},
        "within_trial_values": {},
        "within_trial_timesteps": {},
        "within_trial_states": {},
        "hidden_states": {},
        "rewards": [],
        "reward_timesteps": [],
        "trial_reward_consumed": {},  # trial_idx -> sum of rewards during reward-available steps
        "aux_iti": {"trial_global_idx": [], "stim_acc": [], "reward_mse": []},
        "trial_boundaries": [],
        "grad_norms": {
            "rnn": [],
            "rnn_in": [],
            "rnn_rec": [],
            "actor_fc": [],
            "critic_fc": [],
            "total": [],
            "timestep": [],
        },
    }

    idx_to_stim = {i: k for i, k in enumerate("ABCDEF")}

    model.train()
    agent.reset_hidden_state()
    obs, info = env.reset(seed=seed)
    prev_action = torch.zeros(action_size)
    prev_reward = torch.tensor(0.0)

    states_b, pa_b, pr_b, act_b, rew_b, ns_b, npa_b, npr_b, done_b = [[] for _ in range(9)]
    aux_stim_b, aux_rew_b, aux_mask_b = [], [], []

    def record_step(t_idx, obs, action, action_prob, value, rnn_out, reward, info):
        state_idx = int(np.argmax(obs))
        trial_info = info.get("trial_idx")
        lick_p = action_prob if action == 0 else (1.0 - action_prob)
        if state_idx in idx_to_stim:
            k = idx_to_stim[state_idx]
            metrics["lick_probs"][k].append(float(lick_p))
            metrics["values"][k].append(float(value))
            metrics["timesteps"][k].append(int(t_idx))
        if trial_info is not None:
            td = trial_structure[trial_info]
            if t_idx == td["trial_start"]:
                metrics["trial_boundaries"].append(
                    {
                        "trial_idx": trial_info,
                        "trial_start": td["trial_start"],
                        "trial_end": td["trial_end"],
                        "stimulus": td["stimulus"],
                        "reversal_phase": td["reversal_phase"],
                        "phase_idx": td.get("phase_idx", td["reversal_phase"]),
                        "reward_available": td["reward_available"],
                    }
                )
                for key in (
                    "within_trial_lick_probs",
                    "within_trial_values",
                    "within_trial_timesteps",
                    "within_trial_states",
                    "hidden_states",
                ):
                    metrics[key][trial_info] = []
            if trial_info in metrics["within_trial_lick_probs"]:
                metrics["within_trial_lick_probs"][trial_info].append(float(lick_p))
                metrics["within_trial_values"][trial_info].append(float(value))
                metrics["within_trial_timesteps"][trial_info].append(int(t_idx))
                metrics["within_trial_states"][trial_info].append(info.get("state_name", "?"))
                metrics["hidden_states"][trial_info].append(rnn_out.detach().cpu().numpy())
            if t_idx == td["stim_window"][-1]:
                k = idx_to_stim.get(int(td["stimulus"]))
                if k:
                    metrics["trial_lick_probs"][k].append(float(lick_p))
                    metrics["trial_values"][k].append(float(value))
                metrics["trial_timesteps"].append(int(t_idx))
                metrics["trial_indices"].append(int(trial_info))
                metrics["trial_reversal_phases"].append(int(td["reversal_phase"]))
        if trial_info is not None and info.get("reward_available", False):
            metrics["trial_reward_consumed"][int(trial_info)] = metrics["trial_reward_consumed"].get(
                int(trial_info), 0.0
            ) + float(reward)
        if info.get("reward_available", False):
            metrics["rewards"].append(float(reward))
            metrics["reward_timesteps"].append(int(t_idx))

    post_end = int((phase_boundaries or {}).get("total_end") or phase_boundaries["post_reversal"]["end"])

    # Phase lookup for progress logs
    phases = (phase_boundaries or {}).get("phases") or []
    phase_ends = {int(ph["end"]): int(ph["phase_idx"]) for ph in phases if "end" in ph and "phase_idx" in ph}

    weight_snaps: list[dict] = []

    def _snap_weights(t_idx: int, phase_idx: int | None) -> None:
        # Save float32 copies to keep files small and consistent.
        wi = getattr(model.rnn, "weight_ih_l0", None)
        wh = getattr(model.rnn, "weight_hh_l0", None)
        wa = getattr(model, "actor_fc").weight
        wc = getattr(model, "critic_fc").weight
        weight_snaps.append(
            {
                "timestep": int(t_idx),
                "phase_idx": None if phase_idx is None else int(phase_idx),
                "W_ih": None if wi is None else wi.detach().cpu().numpy().astype(np.float32),
                "W_hh": None if wh is None else wh.detach().cpu().numpy().astype(np.float32),
                "W_actor": wa.detach().cpu().numpy().astype(np.float32),
                "W_critic": wc.detach().cpu().numpy().astype(np.float32),
            }
        )

    def _phase_idx_for_timestep(t: int) -> int | None:
        if not phases:
            return None
        for ph in reversed(phases):
            if t >= int(ph["start"]):
                return int(ph["phase_idx"])
        return int(phases[0]["phase_idx"])

    _win_rewards: list[float] = []
    _win_lick_p: list[float] = []
    _last_log_t = 0

    for t_idx in range(post_end):
        if readout_freeze_at_ts is not None and t_idx == readout_freeze_at_ts:
            _freeze_readout_params(model)
            _log(f"{log_prefix}freezing readout at t={t_idx:,}", indent=1)
        if actor_frozen and t_idx == 0:
            _freeze_actor_readout_params(model)
        if actor_frozen and critic_plastic_until_ts is not None and t_idx == int(critic_plastic_until_ts):
            _freeze_critic_readout_params(model)
            _log(f"{log_prefix}freezing critic readout at t={t_idx:,}", indent=1)

        state = torch.from_numpy(obs).float()
        _use_prev = use_prev_ar if (prev_ar_switch_at_ts is None or t_idx < prev_ar_switch_at_ts) else False
        _pa = prev_action if _use_prev else torch.zeros(action_size)
        _pr = (prev_reward * float(reward_scale)) if _use_prev else torch.tensor(0.0)
        if fixed_policy_always_lick:
            # Forward pass to advance hidden state / compute value, but force lick action.
            with torch.no_grad():
                probs, val, agent.hidden_state, rnn_out = model(
                    state.unsqueeze(0),
                    _pa.unsqueeze(0),
                    _pr.unsqueeze(0),
                    agent.hidden_state,
                    return_rnn_out=True,
                )
            action = 0
            action_prob = 1.0  # for plotting: max lick probability by construction
            value = float(val.squeeze().item())
            rnn_out = rnn_out.squeeze(0)
        else:
            action, action_prob, value, rnn_out = agent.select_action(
                state, _pa, _pr, deterministic=False, policy_clip=0.25, return_rnn_out=True
            )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Optionally replace sampled reward with expected/consumed reward:
        # r_t = 1[reward_available] * p(lick)
        if reward_consumption == "expected":
            reward_available = 1.0 if info.get("reward_available", False) else 0.0
            lick_prob = float(action_prob if action == 0 else (1.0 - action_prob))
            reward = float(reward_available * lick_prob)

        record_step(t_idx, obs, action, action_prob, value, rnn_out, reward, info)

        # Rolling window for progress logs
        _win_rewards.append(float(reward))
        _win_lick_p.append(float(action_prob if action == 0 else 1.0 - action_prob))

        next_prev_action = torch.zeros(action_size)
        next_prev_action[action] = 1.0

        if t_idx < update_enabled_until:
            states_b.append(obs)
            pa_b.append(_pa.numpy())
            pr_b.append(float(_pr.item()))
            act_b.append(int(action))
            rew_b.append(float(reward))
            ns_b.append(next_obs)
            npa_b.append(next_prev_action.numpy() if _use_prev else np.zeros(action_size))
            npr_b.append((float(reward) * float(reward_scale)) if _use_prev else 0.0)
            done_b.append(bool(done))

            # Aux targets: first ITI timestep of the just-finished trial.
            # We read out from the current hidden state (ITI observation).
            trial_i = info.get("trial_idx")
            aux_mask = 0.0
            aux_stim_t = 0
            aux_rew_t = 0.0
            if trial_i is not None:
                td = trial_structure[int(trial_i)]
                iti = td.get("iti_window") or []
                if iti and int(t_idx) == int(iti[0]) and aux_iti_predict:
                    aux_mask = 1.0
                    aux_stim_t = int(td.get("stimulus", 0))
                    aux_rew_t = float(metrics["trial_reward_consumed"].get(int(trial_i), 0.0))
            aux_mask_b.append(aux_mask)
            aux_stim_b.append(aux_stim_t)
            aux_rew_b.append(aux_rew_t)

            if len(states_b) >= 1:
                upd = agent.update(
                    torch.from_numpy(np.array(states_b)).float(),
                    torch.from_numpy(np.array(pa_b)).float(),
                    torch.from_numpy(np.array(pr_b)).float(),
                    torch.from_numpy(np.array(act_b)).long(),
                    torch.from_numpy(np.array(rew_b)).float(),
                    torch.from_numpy(np.array(ns_b)).float(),
                    torch.from_numpy(np.array(npa_b)).float(),
                    torch.from_numpy(np.array(npr_b)).float(),
                    torch.from_numpy(np.array(done_b)).float(),
                    aux_stim_target=torch.from_numpy(np.array(aux_stim_b)).long(),
                    aux_reward_target=torch.from_numpy(np.array(aux_rew_b)).float(),
                    aux_mask=torch.from_numpy(np.array(aux_mask_b)).float(),
                    aux_stim_weight=float(aux_stim_weight),
                    aux_reward_weight=float(aux_reward_weight),
                )
                # Record aux metrics at first-ITI steps
                if aux_iti_predict and upd:
                    if "aux_stim_acc" in upd or "aux_reward_mse" in upd:
                        # trial_boundaries appends at trial_start; by ITI start, this trial is already present.
                        gidx = len(metrics["trial_boundaries"]) - 1
                        metrics["aux_iti"]["trial_global_idx"].append(int(gidx))
                        if "aux_stim_acc" in upd:
                            metrics["aux_iti"]["stim_acc"].append(float(upd["aux_stim_acc"]))
                        if "aux_reward_mse" in upd:
                            metrics["aux_iti"]["reward_mse"].append(float(upd["aux_reward_mse"]))
                if upd and "grad_norms" in upd:
                    for gk in ("rnn", "rnn_in", "rnn_rec", "actor_fc", "critic_fc", "total"):
                        if gk in upd["grad_norms"]:
                            metrics["grad_norms"].setdefault(gk, [])
                            metrics["grad_norms"][gk].append(float(upd["grad_norms"][gk]))
                    metrics["grad_norms"]["timestep"].append(int(t_idx))
                states_b, pa_b, pr_b, act_b, rew_b, ns_b, npa_b, npr_b, done_b = [[] for _ in range(9)]
                aux_stim_b, aux_rew_b, aux_mask_b = [], [], []

        obs = next_obs
        prev_action = next_prev_action
        prev_reward = torch.tensor(float(reward), dtype=torch.float32)

        # Snapshot weights at end of each phase (end timestep is inclusive).
        if t_idx in phase_ends:
            _snap_weights(t_idx, phase_ends[t_idx])

        if log_every and ((t_idx - _last_log_t) >= log_every or t_idx == 0 or t_idx == post_end - 1):
            pct = 100.0 * (t_idx + 1) / max(post_end, 1)
            ph_i = _phase_idx_for_timestep(t_idx)
            ph_s = f"phase={ph_i}" if ph_i is not None else "phase=?"
            n_trials = len(metrics["trial_boundaries"])
            rr = float(np.mean(_win_rewards)) if _win_rewards else 0.0
            lp = float(np.mean(_win_lick_p)) if _win_lick_p else 0.0
            upd = "updates=on" if (t_idx < update_enabled_until) else "updates=off"
            _log(
                f"{log_prefix}t={t_idx:>10,}/{post_end:,}  ({pct:5.1f}%)  {ph_s}  "
                f"trials={n_trials:,}  rew_rate={rr:+.3f}  lick_p={lp:.3f}  {upd}",
                indent=1,
            )
            _win_rewards.clear()
            _win_lick_p.clear()
            _last_log_t = t_idx

    # Convert to numpy-like format matching existing runner
    metrics_numpy = {
        "lick_probs": {k: np.asarray(v) for k, v in metrics["lick_probs"].items()},
        "values": {k: np.asarray(v) for k, v in metrics["values"].items()},
        "timesteps": {k: np.asarray(v) for k, v in metrics["timesteps"].items()},
        "trial_lick_probs": {k: np.asarray(v) for k, v in metrics["trial_lick_probs"].items()},
        "trial_values": {k: np.asarray(v) for k, v in metrics["trial_values"].items()},
        "trial_timesteps": np.asarray(metrics["trial_timesteps"]),
        "trial_indices": np.asarray(metrics["trial_indices"]),
        "trial_reversal_phases": np.asarray(metrics["trial_reversal_phases"]),
        "rewards": np.asarray(metrics["rewards"]),
        "reward_timesteps": np.asarray(metrics["reward_timesteps"]),
        "trial_boundaries": metrics["trial_boundaries"],
        "trial_reward_consumed": metrics["trial_reward_consumed"],
        "aux_iti": metrics["aux_iti"],
        "within_trial_lick_probs": metrics["within_trial_lick_probs"],
        "within_trial_values": metrics["within_trial_values"],
        "within_trial_timesteps": metrics["within_trial_timesteps"],
        "within_trial_states": metrics["within_trial_states"],
        "hidden_states": metrics["hidden_states"],
        "grad_norms": metrics["grad_norms"],
    }

    return metrics_numpy, model, weight_snaps


def _plot_weight_snapshots_grid(weight_snaps: list[dict], out_path: Path, title: str) -> None:
    """Plot a phase×matrix grid of weight snapshots."""
    if not weight_snaps:
        return
    # Keep only snaps with phase_idx
    snaps = [s for s in weight_snaps if s.get("phase_idx") is not None]
    if not snaps:
        return
    snaps = sorted(snaps, key=lambda d: int(d["phase_idx"]))

    mats = [
        ("W_ih", "RNN W_ih"),
        ("W_hh", "RNN W_hh"),
        ("W_actor", "Actor FC"),
        ("W_critic", "Critic FC"),
    ]

    n_rows = len(snaps)
    n_cols = len(mats)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 1.8 * n_rows),
        squeeze=False,
    )

    # Per-matrix colour scaling (symmetric) across phases
    scales: dict[str, float] = {}
    for key, _ in mats:
        arrs = [s.get(key) for s in snaps if s.get(key) is not None]
        if not arrs:
            scales[key] = 1.0
            continue
        mx = float(np.max([np.max(np.abs(a)) for a in arrs]))
        scales[key] = mx if mx > 1e-12 else 1.0

    for r, s in enumerate(snaps):
        ph = int(s["phase_idx"])
        for c, (key, col_title) in enumerate(mats):
            ax = axes[r][c]
            a = s.get(key)
            if a is None:
                ax.axis("off")
                continue
            vmax = scales[key]
            im = ax.imshow(a, cmap="jet", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"ph{ph}", fontsize=8)

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_lightweight_lv(metrics_numpy: dict, out_path: Path) -> None:
    # Per-trial reward consumed (global trial order = trial_boundaries order)
    tbs = metrics_numpy.get("trial_boundaries") or []
    trc = metrics_numpy.get("trial_reward_consumed") or {}
    trial_reward_consumed = [float(trc.get(int(tb["trial_idx"]), 0.0)) for tb in tbs]
    lv = {
        "trial_lick_probs": {k: metrics_numpy["trial_lick_probs"][k] for k in "ABCDEF"},
        "trial_values": {k: metrics_numpy["trial_values"][k] for k in "ABCDEF"},
        "trial_reward_consumed": np.asarray(trial_reward_consumed, dtype=float),
        "rev_indices_global": _compute_reversal_global_indices(metrics_numpy["trial_boundaries"]),
        "global_trial_indices_by_stim": _global_trial_indices_per_stim(metrics_numpy["trial_boundaries"]),
        "grad_norms": metrics_numpy.get("grad_norms", {}),
    }
    with open(out_path, "wb") as f:
        pickle.dump(lv, f, protocol=pickle.HIGHEST_PROTOCOL)


def _infer_prev_ar_switch_trial_index(metrics_numpy: dict, switch_ts: int) -> int | None:
    """Return global trial index (within metrics trial_boundaries) at which to switch prev A/R."""
    tbs = metrics_numpy.get("trial_boundaries") or []
    for i, tb in enumerate(tbs):
        if int(tb.get("trial_start", 0)) >= int(switch_ts):
            return int(i)
    return None


def _infer_trial_index_at_timestep(metrics_numpy: dict, ts: int) -> int | None:
    """Return global trial index (within trial_boundaries) whose trial_start >= ts."""
    tbs = metrics_numpy.get("trial_boundaries") or []
    for i, tb in enumerate(tbs):
        if int(tb.get("trial_start", 0)) >= int(ts):
            return int(i)
    return None


def _variants_default(variant_set: str) -> list[Variant]:
    """Return variants for the requested sweep.

    - full12: baseline only; 4 readout-freeze schedules × 3 prevAR.
    - sweep18: baseline + 2 new modes; 2 readout-freeze schedules × 3 prevAR.
    - sweep12: like sweep18 but drops prevAR=on (2 prevAR modes).
    """
    variants: list[Variant] = []

    if variant_set == "full12":
        freeze_choices = (None, 0, 1, 2)
        modes: list[VariantMode] = ["baseline"]
        prev_modes = ("on", "off", "train_on_infer_off")
    elif variant_set == "sweep18":
        # As discussed: drop freeze-from-start. Keep nofreeze and freeze-after-1-phase.
        freeze_choices = (None, 1)
        modes = [
            "baseline",
            "fixed_policy_always_lick",
            "actor_frozen_critic_plastic_first_phase_only",
        ]
        prev_modes = ("on", "off", "train_on_infer_off")
    elif variant_set == "sweep12":
        # Like sweep18 but without prevAR=on (redundant for your current questions).
        freeze_choices = (None, 1)
        modes = [
            "baseline",
            "fixed_policy_always_lick",
            "actor_frozen_critic_plastic_first_phase_only",
        ]
        prev_modes = ("off", "train_on_infer_off")
    elif variant_set == "iter2_rewardscale":
        # Minimal iteration: only actor frozen from start, critic plastic for phase 0 only,
        # prevAR train_on_infer_off, and two reward input scalings (x1 vs x10),
        # PLUS prevAR=off with reward scale fixed at x1 (control).
        freeze_choices = (None,)
        modes = ["actor_frozen_critic_plastic_first_phase_only"]
        prev_modes = ("off", "train_on_infer_off")
    elif variant_set == "iter2_noscale":
        # Same structure as iter2_rewardscale, but without reward scaling (all rscale=1x).
        freeze_choices = (None,)
        modes = ["actor_frozen_critic_plastic_first_phase_only"]
        prev_modes = ("off", "train_on_infer_off")
    else:
        raise ValueError(f"Unknown variant_set: {variant_set}")

    for freeze_phases in freeze_choices:
        if freeze_phases is None:
            freeze_name = "nofreeze"
        elif freeze_phases == 0:
            freeze_name = "freezeRO_fromstart"
        elif freeze_phases == 1:
            freeze_name = "freezeRO_after1phase"
        else:
            freeze_name = f"freezeRO_after{freeze_phases}phases"

        for prev_mode in prev_modes:
            for vm in modes:
                if variant_set == "iter2_rewardscale":
                    rs_list = (1.0,) if prev_mode == "off" else (1.0, 10.0)
                    for rs in rs_list:
                        variants.append(
                            Variant(
                                name=f"{freeze_name}_{vm}_prevAR_{prev_mode}_rscale{int(rs)}x",
                                prev_ar_mode=prev_mode,  # type: ignore[assignment]
                                readout_freeze_after_phases=freeze_phases,
                                mode=vm,
                            )
                        )
                elif variant_set == "iter2_noscale":
                    variants.append(
                        Variant(
                            name=f"{freeze_name}_{vm}_prevAR_{prev_mode}_rscale1x",
                            prev_ar_mode=prev_mode,  # type: ignore[assignment]
                            readout_freeze_after_phases=freeze_phases,
                            mode=vm,
                        )
                    )
                else:
                    variants.append(
                        Variant(
                            name=f"{freeze_name}_{vm}_prevAR_{prev_mode}",
                            prev_ar_mode=prev_mode,  # type: ignore[assignment]
                            readout_freeze_after_phases=freeze_phases,
                            mode=vm,
                        )
                    )

    return variants


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task_pkl", type=str, required=True, help="Path to equal-phase multirev task pkl.")
    p.add_argument("--results_dir", type=str, default=str(_repo_root() / "results"))
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--readout_fraction", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--policy_clip", type=float, default=0.25)
    p.add_argument("--exclude_mid_stim", type=int, default=1)
    p.add_argument(
        "--aux_iti_predict",
        action="store_true",
        help="Enable auxiliary ITI prediction: stimulus CE + reward MSE at first ITI timestep.",
    )
    p.add_argument("--aux_stim_weight", type=float, default=1.0)
    p.add_argument("--aux_reward_weight", type=float, default=1.0)
    p.add_argument(
        "--reward_consumption",
        type=str,
        default="sampled",
        choices=["sampled", "expected"],
        help="Reward used for learning/logging: environment sampled vs expected (reward_available * p(lick)).",
    )
    p.add_argument(
        "--variant_set",
        type=str,
        default="full12",
        choices=["full12", "sweep18", "sweep12", "iter2_rewardscale", "iter2_noscale"],
        help="Which variant family set to run.",
    )
    p.add_argument(
        "--tdr_pre_start",
        type=int,
        default=50,
        help="TDR pre-phase block start index (per stimulus).",
    )
    p.add_argument(
        "--tdr_pre_end",
        type=int,
        default=-1,
        help="TDR pre-phase block end (exclusive); -1 means use all remaining.",
    )
    p.add_argument(
        "--tdr_post_start",
        type=int,
        default=50,
        help="TDR post-phase block start index (per stimulus).",
    )
    p.add_argument(
        "--tdr_post_end",
        type=int,
        default=-1,
        help="TDR post-phase block end (exclusive); -1 means use all remaining.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--run_indices",
        type=str,
        default="",
        help="Optional comma-separated run indices to execute (e.g. '0,3,7').",
    )
    p.add_argument(
        "--only_prev_ar_mode",
        type=str,
        default="",
        choices=["", "on", "off", "train_on_infer_off"],
        help="If set, only run variants with this prev_ar_mode.",
    )
    p.add_argument(
        "--only_variant_substr",
        type=str,
        default="",
        help="If set, only run variants whose name contains this substring.",
    )
    p.add_argument(
        "--infer_prev_ar_on_test_phases",
        type=int,
        default=-1,
        help=(
            "Only used for prev_ar_mode=train_on_infer_off. "
            "Number of test phases to keep prev A/R ON while plasticity is OFF. "
            "After that, prev A/R is switched OFF for the remainder of inference. "
            "-1 means half of test phases."
        ),
    )
    args = p.parse_args()

    task_pkl = Path(args.task_pkl).resolve()
    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    (
        state_sequence,
        reward_sequence,
        reversal_mask,
        phase_boundaries,
        trial_structure,
        _state_map,
    ) = _load_task(task_pkl)

    pb = phase_boundaries or {}
    n_train_phases = int(pb.get("n_train_phases", 0))
    n_test_phases = int(pb.get("n_test_phases", 0))
    train_end_ts = int(pb.get("train_end", 0))
    total_end_ts = int(pb.get("total_end", state_sequence.shape[0]))

    if n_train_phases <= 0 or train_end_ts <= 0:
        raise ValueError("Task pkl must include phase_boundaries['n_train_phases'] and ['train_end'].")

    train_trials, test_trials = _split_trial_structure(trial_structure, n_train_phases=n_train_phases)
    train_trials_set = set(train_trials)
    test_trials_set = set(test_trials)

    _log(f"Task: {task_pkl.name}")
    _log(f"  Timesteps: train_end={train_end_ts:,}  total_end={total_end_ts:,}")
    _log(f"  Trials: train={len(train_trials):,}  test={len(test_trials):,}")

    plot_stims = [s for s in range(6) if not (args.exclude_mid_stim and s in (2, 3))]

    hidden_size = int(args.hidden_size)
    readout_size = max(1, int(hidden_size * float(args.readout_fraction)))

    tdr_pre_end = None if args.tdr_pre_end < 0 else args.tdr_pre_end
    tdr_post_end = None if args.tdr_post_end < 0 else args.tdr_post_end
    base_cfg = {
        "hidden_size": hidden_size,
        "readout_size": readout_size,
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "policy_clip": float(args.policy_clip),
        "aux_iti_predict": bool(args.aux_iti_predict),
        "aux_n_stim_classes": 6,
        "aux_stim_weight": float(args.aux_stim_weight),
        "aux_reward_weight": float(args.aux_reward_weight),
        # Required by `analyse()` (same semantics as partial multirun).
        "tdr_pre_block": (args.tdr_pre_start, tdr_pre_end),
        "tdr_post_block": (args.tdr_post_start, tdr_post_end),
        # used by plotting utils (for titles)
        "phase_boundaries_for_plots": phase_boundaries,
    }

    variants = _variants_default(args.variant_set)
    if args.only_prev_ar_mode:
        variants = [v for v in variants if v.prev_ar_mode == args.only_prev_ar_mode]
    if args.only_variant_substr:
        variants = [v for v in variants if args.only_variant_substr in v.name]
    if not variants:
        raise ValueError("No variants selected after filtering.")

    # Interleaved execution
    _t_global = time.time()
    if args.run_indices.strip():
        run_indices = sorted({int(x) for x in args.run_indices.split(",") if x.strip() != ""})
    else:
        run_indices = list(range(int(args.n_runs)))
    total_jobs = len(run_indices) * len(variants)
    jobs_done = 0

    def _fmt_eta(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        s = int(round(seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    for run_idx in run_indices:
        for v in variants:
            seed = run_idx * 7 + 42
            variant_dir = results_root / f"equalphase_multirev_{task_pkl.stem}_partialro{readout_size}_{v.name}"
            run_dir = variant_dir / f"run_{run_idx:02d}_seed{seed}"
            train_dir = run_dir / "train"
            infer_dir = run_dir / "infer_frozen"
            agg_dir = variant_dir / "aggregate"

            train_dir.mkdir(parents=True, exist_ok=True)
            infer_dir.mkdir(parents=True, exist_ok=True)
            agg_dir.mkdir(parents=True, exist_ok=True)

            # Config CSV once per variant
            cfg_csv = variant_dir / "config.csv"
            if not cfg_csv.exists():
                with open(cfg_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["param", "value"])
                    for k, val in base_cfg.items():
                        if k == "phase_boundaries_for_plots":
                            continue
                        w.writerow([k, val])
                    w.writerow(["task_pkl", str(task_pkl)])
                    w.writerow(["readout_freeze_after_phases", v.readout_freeze_after_phases])
                    w.writerow(["prev_ar_mode", v.prev_ar_mode])
                    w.writerow(["variant_mode", v.mode])

            done_flag = infer_dir / "run_results.pkl"
            if done_flag.exists() and not args.overwrite:
                jobs_done += 1
                continue

            _log("")
            _log(f"{'─'*56}")
            # ETA based on average time per completed job so far
            if jobs_done > 0:
                avg_per_job = (time.time() - _t_global) / jobs_done
                eta_s = avg_per_job * (total_jobs - jobs_done)
                eta_str = _fmt_eta(eta_s)
            else:
                eta_str = "?:?:?"
            _log(
                f"Run {run_idx+1}/{args.n_runs}  seed={seed}  variant={v.name}  "
                f"[job {jobs_done+1}/{total_jobs} | ETA {eta_str}]"
            )
            _log(f"{'─'*56}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            # Build env + model
            env = ReversalABCDEFMultiTimestepEnv(
                state_sequence,
                reward_sequence,
                reversal_mask,
                trial_structure,
                reward_lick=1.0,
                lick_no_reward=-1.0,
                no_lick=0.0,
            )
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            model = _build_model(state_size, action_size, base_cfg)

            readout_freeze_at_ts = None
            if v.readout_freeze_after_phases is not None:
                readout_freeze_at_ts = _phase_start_timestep_after_phases(
                    phase_boundaries, v.readout_freeze_after_phases
                )

            # Determine prev A/R usage
            use_prev_train = v.prev_ar_mode in ("on", "train_on_infer_off")
            use_prev_infer = v.prev_ar_mode in ("on", "train_on_infer_off")
            # Reward scaling (only used when prev reward input is enabled)
            reward_scale = 10.0 if "_rscale10x" in v.name else 1.0


            # For train_on_infer_off we stagger within inference:
            # plasticity is always off for inference, but prev A/R starts ON then switches OFF.
            prev_ar_switch_ts = None
            if v.prev_ar_mode == "train_on_infer_off":
                if n_test_phases <= 0:
                    raise ValueError("phase_boundaries must include n_test_phases for staggered prev A/R inference.")
                keep_on = args.infer_prev_ar_on_test_phases
                if keep_on < 0:
                    keep_on = max(1, n_test_phases // 2)
                # Switch at the start of test phase (n_train_phases + keep_on)
                phases = (phase_boundaries or {}).get("phases") or []
                switch_phase_idx = n_train_phases + int(keep_on)
                if 0 <= switch_phase_idx < len(phases):
                    prev_ar_switch_ts = int(phases[switch_phase_idx]["start"])

            t0 = time.time()

            # Run full sequence with updates enabled through train_end_ts
            metrics_full, model, weight_snaps_full = _run_segment(
                env=env,
                model=model,
                trial_structure=trial_structure,
                phase_boundaries=phase_boundaries,
                seed=seed,
                train_end_timestep=train_end_ts,
                update_enabled_until=train_end_ts,
                use_prev_ar=use_prev_train,
                readout_freeze_at_ts=readout_freeze_at_ts,
                fixed_policy_always_lick=(v.mode == "fixed_policy_always_lick"),
                actor_frozen=(v.mode == "actor_frozen_critic_plastic_first_phase_only"),
                critic_plastic_until_ts=int((phase_boundaries or {}).get("phases", [{}])[0].get("end", 0)) + 1
                if (v.mode == "actor_frozen_critic_plastic_first_phase_only") and (phase_boundaries or {}).get("phases")
                else None,
                reward_scale=reward_scale,
                reward_consumption=args.reward_consumption,
                aux_iti_predict=bool(args.aux_iti_predict),
                aux_stim_weight=float(args.aux_stim_weight),
                aux_reward_weight=float(args.aux_reward_weight),
                log_prefix="train: ",
                log_every=50_000,
            )

            # Split metrics into train / test portions by trial membership
            metrics_train = _filter_metrics_to_trials(
                metrics_full, train_trials_set, trial_structure
            )

            # For frozen-inference, rerun the sequence with *zero updates*.
            # This preserves hidden-state evolution but prevents any further learning.
            # To implement "train_on_infer_off", we switch prev A/R off in this pass.
            env2 = ReversalABCDEFMultiTimestepEnv(
                state_sequence,
                reward_sequence,
                reversal_mask,
                trial_structure,
                reward_lick=1.0,
                lick_no_reward=-1.0,
                no_lick=0.0,
            )
            metrics_full_frozen, _, weight_snaps_frozen = _run_segment(
                env=env2,
                model=model,
                trial_structure=trial_structure,
                phase_boundaries=phase_boundaries,
                seed=seed,
                train_end_timestep=train_end_ts,
                update_enabled_until=0,  # no updates anywhere (pure inference roll-out)
                use_prev_ar=use_prev_infer,
                prev_ar_switch_at_ts=prev_ar_switch_ts,
                readout_freeze_at_ts=readout_freeze_at_ts,
                fixed_policy_always_lick=(v.mode == "fixed_policy_always_lick"),
                actor_frozen=(v.mode == "actor_frozen_critic_plastic_first_phase_only"),
                critic_plastic_until_ts=None,
                reward_scale=reward_scale,
                reward_consumption=args.reward_consumption,
                aux_iti_predict=False,  # don't train aux during frozen inference
                log_prefix="infer: ",
                log_every=50_000,
            )
            metrics_test_frozen = _filter_metrics_to_trials(
                metrics_full_frozen, test_trials_set, trial_structure
            )

            # Analyse/plots: reuse existing analysis functions by calling the same pipeline as the base runner.
            # Here we only produce the standard plots by feeding in the metrics for that segment.
            cfg_train = dict(base_cfg)
            cfg_train["use_prev_action_reward"] = bool(use_prev_train)
            cfg_infer = dict(base_cfg)
            cfg_infer["use_prev_action_reward"] = bool(use_prev_infer)

            # Train-segment plots
            with open(train_dir / "metrics_numpy.pkl", "wb") as f:
                pickle.dump(metrics_train, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Save lightweight lv + freeze marker for plotting
            train_lv_path = train_dir / "lick_value_data.pkl"
            _save_lightweight_lv(metrics_train, train_lv_path)
            if readout_freeze_at_ts is not None:
                try:
                    with open(train_lv_path, "rb") as f:
                        lv = pickle.load(f)
                    lv["readout_freeze_trial_index_global"] = _infer_trial_index_at_timestep(
                        metrics_train, readout_freeze_at_ts
                    )
                    with open(train_lv_path, "wb") as f:
                        pickle.dump(lv, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            # For actor-frozen/critic-plastic-first-phase, store separate markers.
            if v.mode == "actor_frozen_critic_plastic_first_phase_only":
                try:
                    phases = (phase_boundaries or {}).get("phases") or []
                    critic_freeze_ts = int(phases[0]["end"]) + 1 if phases else None
                    with open(train_lv_path, "rb") as f:
                        lv = pickle.load(f)
                    lv["actor_readout_freeze_trial_index_global"] = 0
                    if critic_freeze_ts is not None:
                        lv["critic_readout_freeze_trial_index_global"] = _infer_trial_index_at_timestep(
                            metrics_train, critic_freeze_ts
                        )
                    with open(train_lv_path, "wb") as f:
                        pickle.dump(lv, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            plot_lick_value(metrics_train, train_dir, cfg_train, plot_stims, run_idx, seed, "train")
            plot_reward_consumed(metrics_train, train_dir, cfg_train, run_idx, seed, "train")
            plot_aux_iti_accuracy(metrics_train, train_dir, cfg_train, run_idx, seed, "train")
            pb_for_gn = dict(phase_boundaries or {})
            pb_for_gn["readout_freeze_ts"] = readout_freeze_at_ts
            pb_for_gn["plasticity_off_ts"] = int(train_end_ts)
            plot_grad_norms(metrics_train, train_dir, cfg_train, "train", run_idx, seed, pb_for_gn)

            # Full analysis + plots (same as existing pipeline)
            trial_structure_train = _trial_structure_local_phase_idx(
                trial_structure, train_trials_set
            )
            results_train = analyse(
                metrics_train, model, cfg_train, trial_structure_train
            )
            plot_tdr_value_psth(
                results_train, train_dir, cfg_train, trial_structure_train, plot_stims, "train"
            )
            plot_decoder(results_train["decoder_results"], train_dir, "train")
            mat_obj = plot_phase_generalisation_matrices(results_train, train_dir, "train")
            if mat_obj is not None:
                with open(train_dir / "decoder_phase_generalisation_matrices.pkl", "wb") as f:
                    pickle.dump(mat_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Frozen-inference plots
            with open(infer_dir / "metrics_numpy.pkl", "wb") as f:
                pickle.dump(metrics_test_frozen, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Save lightweight lv + the prev A/R switch point (trial index) for plotting.
            lv_path = infer_dir / "lick_value_data.pkl"
            _save_lightweight_lv(metrics_test_frozen, lv_path)
            if prev_ar_switch_ts is not None:
                try:
                    with open(lv_path, "rb") as f:
                        lv = pickle.load(f)
                    lv["prev_ar_switch_trial_index_global"] = _infer_prev_ar_switch_trial_index(
                        metrics_test_frozen, prev_ar_switch_ts
                    )
                    with open(lv_path, "wb") as f:
                        pickle.dump(lv, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            plot_lick_value(metrics_test_frozen, infer_dir, cfg_infer, plot_stims, run_idx, seed, "infer (frozen)")
            plot_reward_consumed(metrics_test_frozen, infer_dir, cfg_infer, run_idx, seed, "infer (frozen)")
            plot_aux_iti_accuracy(metrics_test_frozen, infer_dir, cfg_infer, run_idx, seed, "infer (frozen)")
            plot_grad_norms(metrics_test_frozen, infer_dir, cfg_infer, "infer (frozen)", run_idx, seed, phase_boundaries or {})

            trial_structure_infer = _trial_structure_local_phase_idx(
                trial_structure, test_trials_set
            )
            results_infer = analyse(
                metrics_test_frozen, model, cfg_infer, trial_structure_infer
            )
            plot_tdr_value_psth(
                results_infer, infer_dir, cfg_infer, trial_structure_infer, plot_stims, "infer (frozen)"
            )
            plot_decoder(results_infer["decoder_results"], infer_dir, "infer (frozen)")
            mat_obj = plot_phase_generalisation_matrices(results_infer, infer_dir, "infer (frozen)")
            if mat_obj is not None:
                with open(infer_dir / "decoder_phase_generalisation_matrices.pkl", "wb") as f:
                    pickle.dump(mat_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            def _save_run_results(out_dir: Path, res_obj: dict, label: str) -> None:
                run_results = {
                    "decoder_results": res_obj["decoder_results"],
                    "value_axes": res_obj["value_axes"],
                    "tdr_projs": res_obj["tdr_projs"],
                    "tdr_va_psth": res_obj["tdr_va_psth"],
                    "all_stim_labels": res_obj["all_stim_labels"],
                    "all_phase_labels": res_obj["all_phase_labels"],
                    "pre_conv_mask": res_obj["pre_conv_mask"],
                    "post_conv_mask": res_obj["post_conv_mask"],
                    "readout_weights": {
                        "actor_fc_weight": model.actor_fc.weight.detach().cpu().numpy(),
                        "actor_fc_bias": model.actor_fc.bias.detach().cpu().numpy(),
                        "critic_fc_weight": model.critic_fc.weight.detach().cpu().numpy(),
                        "critic_fc_bias": model.critic_fc.bias.detach().cpu().numpy(),
                        "readout_indices": list(range(readout_size)),
                    },
                    "params": {**base_cfg, "seed": seed, "label": label, "use_prev_action_reward": bool(label != "infer (frozen)" and use_prev_train)},
                }
                with open(out_dir / "run_results.pkl", "wb") as f:
                    pickle.dump(run_results, f, protocol=pickle.HIGHEST_PROTOCOL)

            _save_run_results(train_dir, results_train, "train")
            _save_run_results(infer_dir, results_infer, "infer (frozen)")

            # Save weight snapshots + grid plots (train and infer)
            try:
                with open(train_dir / "weight_snapshots.pkl", "wb") as f:
                    pickle.dump(weight_snaps_full, f, protocol=pickle.HIGHEST_PROTOCOL)
                _plot_weight_snapshots_grid(
                    weight_snaps_full,
                    out_path=train_dir / "weight_snapshots_grid.png",
                    title=f"{variant_dir.name} | {run_dir.name} | train weight snapshots",
                )
            except Exception as e:
                _log(f"  WARNING: train weight snapshot save/plot failed: {e}", indent=1)
            try:
                with open(infer_dir / "weight_snapshots.pkl", "wb") as f:
                    pickle.dump(weight_snaps_frozen, f, protocol=pickle.HIGHEST_PROTOCOL)
                _plot_weight_snapshots_grid(
                    weight_snaps_frozen,
                    out_path=infer_dir / "weight_snapshots_grid.png",
                    title=f"{variant_dir.name} | {run_dir.name} | infer (frozen) weight snapshots",
                )
            except Exception as e:
                _log(f"  WARNING: infer weight snapshot save/plot failed: {e}", indent=1)

            # Always save a combined train→infer plot at run root.
            if _plot_combined_train_infer is not None:
                try:
                    with open(train_dir / "lick_value_data.pkl", "rb") as f:
                        lv_train = pickle.load(f)
                    with open(infer_dir / "lick_value_data.pkl", "rb") as f:
                        lv_infer = pickle.load(f)
                    _plot_combined_train_infer(
                        lv_train=lv_train,
                        lv_infer=lv_infer,
                        out_path=run_dir / "combined_train_then_infer_lick_value.png",
                        title=f"{variant_dir.name}\n{run_dir.name}  |  train then infer (plasticity off)",
                        plot_stims=plot_stims,
                        smooth=20,
                    )
                except Exception as e:
                    _log(f"  WARNING: combined train→infer plot failed: {e}", indent=1)

            _log(f"  Completed in {time.time()-t0:.0f}s → {run_dir.name}", indent=1)
            jobs_done += 1

            # Refresh aggregate lick/value + grad norms (cheap, gives you progress as runs accumulate).
            try:
                train_run_dirs = sorted([p / "train" for p in variant_dir.glob("run_*") if (p / "train").is_dir()])
                infer_run_dirs = sorted([p / "infer_frozen" for p in variant_dir.glob("run_*") if (p / "infer_frozen").is_dir()])
                (agg_dir / "train").mkdir(parents=True, exist_ok=True)
                (agg_dir / "infer_frozen").mkdir(parents=True, exist_ok=True)
                # Note: aggregate_plots expects run_results.pkl content for decoder/TDR; our placeholder
                # makes it safe but those plots will be empty until analysis is added.
                aggregate_plots(train_run_dirs, agg_dir / "train", cfg_train, "train", plot_stims)
                aggregate_plots(infer_run_dirs, agg_dir / "infer_frozen", cfg_infer, "infer (frozen)", plot_stims)
                _make_aggregate_combined_plot(variant_dir, plot_stims=plot_stims)
            except Exception as e:
                _log(f"  WARNING: aggregate refresh failed: {e}", indent=1)


if __name__ == "__main__":
    main()


"""
Multi-run training + analysis script for Meta-AC with context-signal input on the
ABEF partial-reversal multi-reversal task.

The model receives the current reward contingency as a one-hot context vector
(size 2: [pre-reversal, post-reversal]) instead of previous action and reward.
This acts as an oracle signal indicating which phase the network is in.

Usage:
    python run_context_signal_abef_multirun.py \
        --n_runs 10 \
        --results_dir ../results/context_signal_abef \
        --n_reversals 10 \
        --trials_per_phase 100

Supported freeze modes: none | readout_only | rnn_only | all
"""

import argparse
import pickle
import sys
import csv
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import torch
from torch.optim import Adam
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

_script_start = time.time()

def _log(msg, indent=0):
    elapsed = time.time() - _script_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    prefix  = f"[{h:02d}:{m:02d}:{s:02d}]" + "  " * indent
    print(f"{prefix} {msg}", flush=True)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cog_nn.tasks.reversal_envs import (
    ReversalABCDEFMultiTimestepEnv,
    load_reversal_abcdef_multitimestep_multirev_data,
)
from cog_nn.models import RNNActorCriticPartialReadoutContextInput
from cog_nn.agents import MetaA2CContextAgent

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _lighten(hex_color, amount=0.55):
    c = np.array(mc.to_rgb(hex_color))
    return tuple(c + amount * (np.ones(3) - c))

STIM_COLORS_DARK  = {0:'#e41a1c',1:'#ff7f00',2:'#4daf4a',3:'#377eb8',4:'#984ea3',5:'#a65628'}
PHASE_COLOR_MAP   = {0: STIM_COLORS_DARK, 1: STIM_COLORS_DARK}
PHASE_LS_MAP      = {0: '--', 1: '-'}
STIM_NAMES_MAP    = {0:'A (H→L)',1:'B (H→H)',2:'C (mid)',3:'D (mid)',4:'E (L→H)',5:'F (L→L)'}
POP_COLORS        = {'Full':'#444444','Projecting':'#1f77b4',
                     'Non-proj':'#d62728',
                     'Readout':'#2ca02c',
                     'Actor logits':'#ff7f00',
                     'Critic value':'#9467bd'}

N_CONTEXTS = 2  # binary contingency: pre / post

# ── Task data generation ───────────────────────────────────────────────────────

def ensure_task_data(task_data_dir, n_phases, trials_per_phase, seed):
    """Generate task pkl if it doesn't exist; return path."""
    # n_phases must be even (equal train/test split in generator)
    half = n_phases // 2
    fname = (f"reversal_abef_multirev_partial_train{half}_tpp{trials_per_phase}"
             f"_test{half}_tpp{trials_per_phase}_seed{seed}.pkl")
    out_path = Path(task_data_dir) / fname
    if out_path.exists():
        _log(f"Task data found: {out_path.name}")
        return out_path

    _log(f"Generating task data → {out_path.name}")
    from cog_nn.tasks.generate_multirev_abef_train_test_phaselen import (
        generate_sequence, expand_to_timesteps, make_phase_boundaries, _to_ohe,
        STIM_WINDOW, REWARD_WINDOW, MIN_ITI, MAX_ITI,
    )
    phase_n_trials = [int(trials_per_phase)] * int(n_phases)
    trial_data = generate_sequence(phase_n_trials, seed=int(seed))
    state_seq, rew_seq, trial_structure, total_ts = expand_to_timesteps(
        trial_data, seed=int(seed),
        stim_window=STIM_WINDOW, reward_window=REWARD_WINDOW,
        min_iti=MIN_ITI, max_iti=MAX_ITI,
    )
    phase_boundaries = make_phase_boundaries(trial_structure, phase_n_trials, half)
    from cog_nn.tasks.generate_multirev_abef_train_test_phaselen import STATE_MAP
    data = {
        "state_sequence_ohe":  _to_ohe(state_seq),
        "reward_sequence":     np.asarray(rew_seq, dtype=np.float32),
        "sequence": {
            "stimuli": trial_data["stimuli"],
            "rewards": trial_data["rewards"],
            "masks":   trial_data["masks"],
        },
        "phase_boundaries":    phase_boundaries,
        "trial_structure":     trial_structure,
        "state_map":           STATE_MAP,
        "trial_params": {
            "stim_window": STIM_WINDOW, "reward_window": REWARD_WINDOW,
            "min_iti": MIN_ITI, "max_iti": MAX_ITI,
        },
        "phase_n_trials":   phase_n_trials,
        "n_train_phases":   half,
        "n_test_phases":    half,
        "reversal_type":    "partial",
        "equal_phase_lengths": True,
        "stimulus_set":     ["A", "B", "E", "F"],
        "stimulus_indices": [0, 1, 4, 5],
        "train_trials_per_phase": int(trials_per_phase),
        "test_trials_per_phase":  int(trials_per_phase),
        "seed":             int(seed),
        "total_timesteps":  int(total_ts),
    }
    Path(task_data_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"  Saved: {out_path}  ({int(total_ts):,} timesteps, {len(trial_structure)} trials)")
    return out_path

# ── Build model ────────────────────────────────────────────────────────────────

def build_model(state_size, action_size, cfg):
    hidden_size   = cfg['hidden_size']
    readout_size  = cfg['readout_size']
    freeze_mode   = cfg['freeze_mode']
    learning_rate = cfg['learning_rate']
    gamma         = cfg['gamma']

    readout_indices = list(range(readout_size))
    model = RNNActorCriticPartialReadoutContextInput(
        state_size=state_size,
        action_size=action_size,
        n_contexts=N_CONTEXTS,
        hidden_size=hidden_size,
        readout_indices=readout_indices,
    )
    if freeze_mode == "readout_only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("actor_fc") or name.startswith("critic_fc")
        model.optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    elif freeze_mode == "rnn_only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("rnn")
        model.optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    elif freeze_mode == "all":
        for p in model.parameters():
            p.requires_grad = False
        model.optimizer = Adam([], lr=learning_rate)
    else:
        model.optimizer = Adam(model.parameters(), lr=learning_rate)
    model.gamma = gamma
    return model

# ── Training ───────────────────────────────────────────────────────────────────

def _build_ts_to_context(trial_structure):
    """Precompute a {timestep: context_idx} mapping from trial_structure."""
    ts_to_ctx = {}
    for t in trial_structure:
        ctx = int(t['reversal_phase'])  # 0 = pre, 1 = post
        for ts in t.get('stim_window', []) + t.get('reward_window', []) + t.get('iti_window', []):
            ts_to_ctx[int(ts)] = ctx
    return ts_to_ctx


def train(env, model, cfg, trial_structure, phase_boundaries, seed):
    hidden_size   = cfg['hidden_size']
    action_size   = env.action_space.n
    state_size    = env.observation_space.shape[0]
    freeze_mode   = cfg['freeze_mode']
    clip          = cfg['policy_clip']

    stim_keys   = ['A', 'B', 'C', 'D', 'E', 'F']
    idx_to_stim = {i: k for i, k in enumerate(stim_keys)}

    metrics = {
        'lick_probs':  {k: [] for k in stim_keys},
        'values':      {k: [] for k in stim_keys},
        'timesteps':   {k: [] for k in stim_keys},
        'trial_lick_probs':        {k: [] for k in stim_keys},
        'trial_values':            {k: [] for k in stim_keys},
        'trial_reward_lick_counts': {k: [] for k in stim_keys},
        'trial_reward_consumed':    {k: [] for k in stim_keys},
        'trial_timesteps':  [],
        'trial_indices':    [],
        'trial_reversal_phases': [],
        'within_trial_lick_probs':  {},
        'within_trial_values':      {},
        'within_trial_timesteps':   {},
        'within_trial_states':      {},
        'hidden_states':            {},
        'rewards':           [],
        'reward_timesteps':  [],
        'trial_boundaries':  [],
        'grad_norms': {'rnn': [], 'rnn_in': [], 'rnn_rec': [], 'actor_fc': [],
                       'critic_fc': [], 'total': [], 'timestep': []},
    }

    agent = MetaA2CContextAgent(
        state_size=state_size, action_size=action_size,
        n_contexts=N_CONTEXTS, hidden_size=hidden_size,
        readout_indices=list(range(cfg['readout_size'])),
    )
    agent.model = model

    # Precompute timestep → context mapping
    ts_to_ctx = _build_ts_to_context(trial_structure)
    # Per-trial accumulators for reward window: trial_idx -> count/sum
    _rew_lick_buf     = {}   # lick count
    _rew_consumed_buf = {}   # sum of max(0, reward) received

    pre_start = 0
    post_end  = phase_boundaries['post_reversal']['end']

    _phases_list = phase_boundaries.get('phases', None)
    if _phases_list:
        def _phase_label(t):
            for ph in reversed(_phases_list):
                if t >= ph['start']:
                    return f"ph{ph['phase_idx']}/{len(_phases_list)-1}"
            return 'ph0'
    else:
        def _phase_label(t):
            return 'post' if t >= phase_boundaries['post_reversal']['start'] else 'pre '

    model.train()
    agent.reset_hidden_state()
    obs, info = env.reset()
    current_ctx_idx = 0

    states_b, ctx_b, act_b, rew_b, ns_b, nctx_b, done_b = [[] for _ in range(7)]

    def record_step(t_idx, obs, action, action_prob, value, rnn_out, reward, info):
        state_idx  = int(np.argmax(obs))
        trial_info = info.get('trial_idx')
        lick_p = action_prob if action == 0 else (1.0 - action_prob)
        if state_idx in idx_to_stim:
            k = idx_to_stim[state_idx]
            metrics['lick_probs'][k].append(lick_p)
            metrics['values'][k].append(float(value))
            metrics['timesteps'][k].append(t_idx)
        if trial_info is not None:
            td = trial_structure[trial_info]
            if t_idx == td['trial_start']:
                metrics['trial_boundaries'].append({
                    'trial_idx':       trial_info,
                    'trial_start':     td['trial_start'],
                    'trial_end':       td['trial_end'],
                    'stimulus':        td['stimulus'],
                    'reversal_phase':  td['reversal_phase'],
                    'phase_idx':       td.get('phase_idx', td['reversal_phase']),
                    'reward_available':td['reward_available'],
                })
                for key in ('within_trial_lick_probs', 'within_trial_values',
                            'within_trial_timesteps', 'within_trial_states', 'hidden_states'):
                    metrics[key][trial_info] = []
            if trial_info in metrics['within_trial_lick_probs']:
                metrics['within_trial_lick_probs'][trial_info].append(lick_p)
                metrics['within_trial_values'][trial_info].append(float(value))
                metrics['within_trial_timesteps'][trial_info].append(t_idx)
                metrics['within_trial_states'][trial_info].append(info.get('state_name', '?'))
                metrics['hidden_states'][trial_info].append(rnn_out.cpu().numpy())
            if t_idx == td['stim_window'][-1]:
                k = idx_to_stim.get(td['stimulus'])
                if k:
                    metrics['trial_lick_probs'][k].append(lick_p)
                    metrics['trial_values'][k].append(float(value))
                metrics['trial_timesteps'].append(t_idx)
                metrics['trial_indices'].append(trial_info)
                metrics['trial_reversal_phases'].append(td['reversal_phase'])
                # Initialise reward-window accumulators for this trial
                _rew_lick_buf[trial_info]     = 0
                _rew_consumed_buf[trial_info] = 0.0
            # Accumulate during reward window
            if trial_info in _rew_lick_buf and t_idx in td['reward_window']:
                if action == 0:
                    _rew_lick_buf[trial_info] += 1
                _rew_consumed_buf[trial_info] += max(0.0, reward)
            # At last reward-window timestep, store both counts
            if len(td['reward_window']) > 0 and t_idx == td['reward_window'][-1]:
                k = idx_to_stim.get(td['stimulus'])
                if k:
                    if trial_info in _rew_lick_buf:
                        metrics['trial_reward_lick_counts'][k].append(_rew_lick_buf.pop(trial_info))
                    if trial_info in _rew_consumed_buf:
                        metrics['trial_reward_consumed'][k].append(_rew_consumed_buf.pop(trial_info))
        if info.get('reward_available', False):
            metrics['rewards'].append(reward)
            metrics['reward_timesteps'].append(t_idx)

    LOG_EVERY = 10_000
    _win_rewards, _win_lick_p = [], []
    _last_log_t = pre_start

    for t_idx in range(pre_start, post_end):
        # Derive context from precomputed map; carry forward if timestep not found
        current_ctx_idx = ts_to_ctx.get(t_idx, current_ctx_idx)
        context = torch.zeros(N_CONTEXTS)
        context[current_ctx_idx] = 1.0

        state = torch.from_numpy(obs).float()
        action, action_prob, value, rnn_out = agent.select_action(
            state, context, deterministic=False, policy_clip=clip, return_rnn_out=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        _win_rewards.append(reward)
        _win_lick_p.append(float(action_prob if action == 0 else 1.0 - action_prob))

        record_step(t_idx, obs, action, action_prob, value, rnn_out, reward, info)

        # Determine next-step context
        next_ctx_idx = ts_to_ctx.get(t_idx + 1, current_ctx_idx)
        next_context = torch.zeros(N_CONTEXTS)
        next_context[next_ctx_idx] = 1.0

        if freeze_mode != "all":
            states_b.append(obs)
            ctx_b.append(context.numpy())
            act_b.append(action)
            rew_b.append(reward)
            ns_b.append(next_obs)
            nctx_b.append(next_context.numpy())
            done_b.append(done)
            if len(states_b) >= 1:
                _upd = agent.update(
                    torch.from_numpy(np.array(states_b)).float(),
                    torch.from_numpy(np.array(ctx_b)).float(),
                    torch.from_numpy(np.array(act_b)).long(),
                    torch.from_numpy(np.array(rew_b)).float(),
                    torch.from_numpy(np.array(ns_b)).float(),
                    torch.from_numpy(np.array(nctx_b)).float(),
                    torch.from_numpy(np.array(done_b)).float(),
                )
                if _upd and 'grad_norms' in _upd:
                    for gk in ('rnn', 'rnn_in', 'rnn_rec', 'actor_fc', 'critic_fc', 'total'):
                        metrics['grad_norms'][gk].append(_upd['grad_norms'].get(gk, 0.0))
                    metrics['grad_norms']['timestep'].append(t_idx)
                states_b, ctx_b, act_b, rew_b, ns_b, nctx_b, done_b = [[] for _ in range(7)]

        obs = next_obs

        if (t_idx - _last_log_t) >= LOG_EVERY or t_idx == pre_start:
            pct      = 100 * (t_idx - pre_start) / max(post_end - pre_start, 1)
            phase    = _phase_label(t_idx)
            rew_rate = np.mean(_win_rewards) if _win_rewards else 0.0
            lick_p   = np.mean(_win_lick_p)  if _win_lick_p  else 0.0
            n_trials = len(metrics['trial_boundaries'])
            _log(f"  step {t_idx:>7d}/{post_end}  ({pct:4.1f}%)  phase={phase}"
                 f"  trials={n_trials:5d}"
                 f"  rew_rate={rew_rate:+.3f}  lick_p={lick_p:.3f}", indent=1)
            _win_rewards.clear(); _win_lick_p.clear()
            _last_log_t = t_idx

    metrics_numpy = {
        'lick_probs':  {k: np.array(v) for k, v in metrics['lick_probs'].items()},
        'values':      {k: np.array(v) for k, v in metrics['values'].items()},
        'timesteps':   {k: np.array(v) for k, v in metrics['timesteps'].items()},
        'trial_lick_probs':        {k: np.array(v) for k, v in metrics['trial_lick_probs'].items()},
        'trial_values':            {k: np.array(v) for k, v in metrics['trial_values'].items()},
        'trial_reward_lick_counts': {k: np.array(v) for k, v in metrics['trial_reward_lick_counts'].items()},
        'trial_reward_consumed':    {k: np.array(v) for k, v in metrics['trial_reward_consumed'].items()},
        'trial_timesteps':       np.array(metrics['trial_timesteps']),
        'trial_indices':         np.array(metrics['trial_indices']),
        'trial_reversal_phases': np.array(metrics['trial_reversal_phases']),
        'rewards':           np.array(metrics['rewards']),
        'reward_timesteps':  np.array(metrics['reward_timesteps']),
        'trial_boundaries':  metrics['trial_boundaries'],
        'within_trial_lick_probs': metrics['within_trial_lick_probs'],
        'within_trial_values':     metrics['within_trial_values'],
        'within_trial_timesteps':  metrics['within_trial_timesteps'],
        'within_trial_states':     metrics['within_trial_states'],
        'hidden_states':           metrics['hidden_states'],
        'grad_norms':              metrics['grad_norms'],
    }
    return metrics_numpy, agent

# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(metrics_numpy, model, cfg, trial_structure):
    hidden_size  = cfg['hidden_size']
    readout_size = cfg['readout_size']
    tdr_pre_blk  = cfg['tdr_pre_block']
    tdr_post_blk = cfg['tdr_post_block']

    readout_indices     = list(range(readout_size))
    non_readout_indices = list(range(readout_size, hidden_size))

    W_actor  = model.actor_fc.weight.detach().numpy()
    b_actor  = model.actor_fc.bias.detach().numpy()
    W_critic = model.critic_fc.weight.detach().numpy()
    b_critic = model.critic_fc.bias.detach().numpy()

    _t0 = time.time()

    # ── Stim-window averaged activations ──────────────────────────────────────
    _log("  Analysis: extracting stim-window activations...", indent=1)
    all_activations, all_stim_labels, all_phase_labels, all_phase_idx = [], [], [], []
    for tb in metrics_numpy['trial_boundaries']:
        ti = tb['trial_idx']
        if ti not in metrics_numpy['hidden_states']:
            continue
        t_info  = trial_structure[ti]
        hs_list = metrics_numpy['hidden_states'][ti]
        wt_ts   = np.array(metrics_numpy['within_trial_timesteps'][ti])
        t_start = t_info['trial_start']
        stim_rel = np.array(t_info['stim_window']) - t_start
        wt_rel   = wt_ts - t_start
        activs = []
        for tr in stim_rel:
            m = np.where(wt_rel == tr)[0]
            if len(m) > 0 and m[0] < len(hs_list):
                activs.append(hs_list[m[0]])
        if not activs:
            continue
        all_activations.append(np.mean(activs, axis=0).squeeze())
        all_stim_labels.append(t_info['stimulus'])
        all_phase_labels.append(t_info['reversal_phase'])
        all_phase_idx.append(t_info.get('phase_idx', t_info.get('reversal_phase', 0)))

    all_activations  = np.array(all_activations)
    all_stim_labels  = np.array(all_stim_labels)
    all_phase_labels = np.array(all_phase_labels)
    all_phase_idx    = np.array(all_phase_idx)
    _log(f"  Analysis: {len(all_activations)} trials extracted  "
         f"(pre={int((all_phase_labels==0).sum())}, post={int((all_phase_labels==1).sum())})",
         indent=1)

    act_proj    = all_activations[:, readout_indices]
    act_nonproj = all_activations[:, non_readout_indices] if non_readout_indices else None
    actor_logits = act_proj @ W_actor.T + b_actor
    critic_value = act_proj @ W_critic.T + b_critic
    readout_all  = np.hstack([actor_logits, critic_value])

    populations = OrderedDict()
    populations['Full']        = all_activations
    populations['Projecting']  = act_proj
    if non_readout_indices:
        populations['Non-proj'] = act_nonproj
    populations['Readout']       = readout_all
    populations['Actor logits']  = actor_logits
    populations['Critic value']  = critic_value

    # ── Regression ────────────────────────────────────────────────────────────
    _log("  Analysis: regression...", indent=1)
    stim_cols   = [2*(all_stim_labels==i).astype(float) - 1 for i in range(6)]
    ctx_reg_c   = 2*(all_phase_labels == 1).astype(float) - 1
    _val_pre    = {0: 1.0, 4: -1.0}
    _val_post   = {0: -1.0, 4: 1.0}
    value_reg_c = np.array([
        (_val_pre if ph==0 else _val_post).get(stim, 0.0)
        for stim, ph in zip(all_stim_labels, all_phase_labels)])
    X_reg = np.column_stack(stim_cols + [ctx_reg_c, value_reg_c])

    all_coefs = {}
    for name, acts in populations.items():
        c, _, _, _ = np.linalg.lstsq(X_reg, acts, rcond=None)
        all_coefs[name] = c.T

    # ── TDR ───────────────────────────────────────────────────────────────────
    _log("  Analysis: TDR...", indent=1)
    def compute_tdr(acts, coefs_mat):
        Q, _ = np.linalg.qr(coefs_mat, mode='reduced')
        return acts @ Q, Q

    tdr_projs = {}
    tdr_Qs    = {}
    for name, acts in populations.items():
        coefs = all_coefs[name]
        if acts.shape[1] >= coefs.shape[1]:
            proj, Q = compute_tdr(acts, coefs)
            tdr_projs[name] = proj
            tdr_Qs[name]    = Q
        else:
            tdr_projs[name] = None

    # ── Value axes & convergence masks ────────────────────────────────────────
    _log("  Analysis: value axes + convergence masks...", indent=1)
    PRE_HIGH  = [0, 1]; PRE_LOW  = [4, 5]
    POST_HIGH = [1, 4]; POST_LOW = [0, 5]

    def _block_mask_phase_idx(stim_labels, phase_idx_labels, phase_idx, block, use_second_half=False):
        start, end = block
        mask = np.zeros(len(stim_labels), dtype=bool)
        for s in range(6):
            idx = np.where((stim_labels == s) & (phase_idx_labels == phase_idx))[0]
            if idx.size == 0:
                continue
            if use_second_half:
                idx = idx[idx.size // 2:]
            mask[idx[start:end]] = True
        return mask

    pre_conv_mask  = _block_mask_phase_idx(all_stim_labels, all_phase_idx, 0, tdr_pre_blk,  use_second_half=False)
    post_conv_mask = _block_mask_phase_idx(all_stim_labels, all_phase_idx, 1, tdr_post_blk, use_second_half=True)
    _log(f"  Analysis: selected trials — pre={pre_conv_mask.sum()}, post={post_conv_mask.sum()}", indent=1)

    def compute_value_axis(acts, phase, high, low, conv_mask):
        sel = (all_phase_labels == phase) & conv_mask
        h_m = sel & np.isin(all_stim_labels, high)
        l_m = sel & np.isin(all_stim_labels, low)
        if h_m.sum() == 0 or l_m.sum() == 0:
            return None
        axis = acts[h_m].mean(0) - acts[l_m].mean(0)
        n = np.linalg.norm(axis)
        return axis / n if n > 1e-12 else axis

    value_axes = {}
    for name, acts in populations.items():
        pre_ax  = compute_value_axis(acts, 0, PRE_HIGH,  PRE_LOW,  pre_conv_mask)
        post_ax = compute_value_axis(acts, 1, POST_HIGH, POST_LOW, post_conv_mask)
        if pre_ax is None or post_ax is None:
            value_axes[name] = None
            _log(f"  Analysis: [{name}] insufficient data for value axis — skipping", indent=1)
            continue
        cos_a = np.clip(abs(np.dot(pre_ax, post_ax)), 0.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_a)))
        value_axes[name] = dict(pre=pre_ax, post=post_ax, angle_deg=angle)
        _log(f"  Analysis: [{name:15s}] pre/post value axis angle = {angle:.1f}°", indent=1)

    # ── TDR time-resolved value projections ───────────────────────────────────
    def _get_pop_hidden(h, name):
        h = h.squeeze()
        if name == 'Projecting':
            return h[readout_indices]
        elif name == 'Non-proj':
            return h[non_readout_indices]
        elif name == 'Readout':
            al = h[readout_indices] @ W_actor.T + b_actor
            cv = h[readout_indices] @ W_critic.T + b_critic
            return np.concatenate([al, cv])
        elif name == 'Actor logits':
            return h[readout_indices] @ W_actor.T + b_actor
        elif name == 'Critic value':
            return h[readout_indices] @ W_critic.T + b_critic
        return h

    _log("  Analysis: TDR value PSTH...", indent=1)
    tdr_va_psth = {}
    for name, ax_info in value_axes.items():
        if ax_info is None:
            tdr_va_psth[name] = None
            continue
        psth = {'pre': defaultdict(list), 'post': defaultdict(list)}
        for axis_key in ('pre', 'post'):
            ax_vec = ax_info[axis_key]
            for tb in metrics_numpy['trial_boundaries']:
                ti = tb['trial_idx']
                if ti not in metrics_numpy['hidden_states']:
                    continue
                t_info  = trial_structure[ti]
                hs_list = metrics_numpy['hidden_states'][ti]
                s, p    = t_info['stimulus'], t_info['reversal_phase']
                n_ts    = min(len(metrics_numpy['within_trial_timesteps'][ti]), len(hs_list))
                if n_ts == 0:
                    continue
                traj = np.array([_get_pop_hidden(hs_list[k], name) @ ax_vec
                                 for k in range(n_ts)])
                psth[axis_key][(s, p)].append(traj)
        tdr_va_psth[name] = psth

    # ── Decoder ───────────────────────────────────────────────────────────────
    _log("  Analysis: value decoder (5-fold CV + cross-generalisation)...", indent=1)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _value_dataset(acts, phase, high, low, conv_mask):
        mask = (all_phase_labels == phase) & np.isin(all_stim_labels, high + low) & conv_mask
        X = acts[mask]
        y = np.isin(all_stim_labels[mask], high).astype(int)
        return X, y

    def _decoder_safe_cv(X, y):
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            return np.array([np.nan])
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
        ])
        try:
            return cross_val_score(clf, X, y, cv=cv5, scoring="accuracy")
        except ValueError:
            return np.array([np.nan])

    decoder_results = {}
    for name, acts in populations.items():
        X_pre,  y_pre  = _value_dataset(acts, 0, PRE_HIGH,  PRE_LOW,  pre_conv_mask)
        X_post, y_post = _value_dataset(acts, 1, POST_HIGH, POST_LOW, post_conv_mask)
        cv_pre  = _decoder_safe_cv(X_pre, y_pre)
        cv_post = _decoder_safe_cv(X_post, y_post)
        if (X_pre.shape[0] > 0 and X_post.shape[0] > 0
                and len(np.unique(y_pre)) >= 2 and len(np.unique(y_post)) >= 2):
            sc1 = StandardScaler(); lr1 = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
            lr1.fit(sc1.fit_transform(X_pre), y_pre)
            p2p = lr1.score(sc1.transform(X_post), y_post)
            sc2 = StandardScaler(); lr2 = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
            lr2.fit(sc2.fit_transform(X_post), y_post)
            p2pr = lr2.score(sc2.transform(X_pre), y_pre)
        else:
            p2p = p2pr = float("nan")
        decoder_results[name] = {
            'pre_cv': cv_pre, 'post_cv': cv_post,
            'pre_to_post': p2p, 'post_to_pre': p2pr,
        }
        _log(f"  [{name:15s}] pre={np.nanmean(cv_pre):.3f}  "
             f"post={np.nanmean(cv_post):.3f}  "
             f"pre→post={p2p:.3f}  post→pre={p2pr:.3f}", indent=1)

    _log(f"  Analysis complete  ({time.time()-_t0:.1f}s)", indent=1)
    return {
        'populations':      populations,
        'all_stim_labels':  all_stim_labels,
        'all_phase_labels': all_phase_labels,
        'all_phase_idx':    all_phase_idx,
        'pre_conv_mask':    pre_conv_mask,
        'post_conv_mask':   post_conv_mask,
        'tdr_projs':        tdr_projs,
        'tdr_va_psth':      tdr_va_psth,
        'value_axes':       value_axes,
        'decoder_results':  decoder_results,
        'all_coefs':        all_coefs,
        'trial_structure':  trial_structure,
        'PRE_HIGH': PRE_HIGH, 'PRE_LOW': PRE_LOW,
        'POST_HIGH': POST_HIGH, 'POST_LOW': POST_LOW,
    }

# ── Helpers for per-run / aggregate ───────────────────────────────────────────

def _compute_reversal_global_indices(trial_boundaries):
    if not trial_boundaries:
        return []
    rev = []
    prev = trial_boundaries[0].get("phase_idx", trial_boundaries[0].get("reversal_phase", 0))
    for i, tb in enumerate(trial_boundaries):
        curr = tb.get("phase_idx", tb.get("reversal_phase", 0))
        if curr != prev:
            rev.append(i)
            prev = curr
    return rev


def _compute_reversal_trial_indices(trial_boundaries):
    a_trials = [tb for tb in trial_boundaries if tb['stimulus'] == 0]
    if not a_trials:
        return []
    rev_indices = []
    prev_phase = a_trials[0].get('phase_idx', a_trials[0]['reversal_phase'])
    for i, tb in enumerate(a_trials):
        curr_phase = tb.get('phase_idx', tb['reversal_phase'])
        if curr_phase != prev_phase:
            rev_indices.append(i)
            prev_phase = curr_phase
    return rev_indices


def _global_trial_indices_per_stim(trial_boundaries):
    out = {k: [] for k in "ABCDEF"}
    for i, tb in enumerate(trial_boundaries):
        s = tb.get("stimulus")
        if s is not None and 0 <= int(s) < 6:
            out["ABCDEF"[int(s)]].append(i)
    return out


def _subsample_for_plot(ts, ys, max_points=50_000):
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(ts)
    if n <= max_points:
        return ts, ys
    stride = int(np.ceil(n / max_points))
    return ts[::stride], ys[::stride]

# ── Per-run plots ──────────────────────────────────────────────────────────────

def plot_lick_value(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label):
    import pandas as pd
    smooth = 20
    stim_names = {0:'A (H→L)',1:'B (H→H)',2:'C (mid)',3:'D (mid)',4:'E (L→H)',5:'F (L→L)'}
    plot_keys  = [k for i, k in enumerate(['A','B','C','D','E','F']) if i in plot_stims]
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}

    rev_indices = _compute_reversal_global_indices(metrics_numpy["trial_boundaries"])
    x_by_stim   = _global_trial_indices_per_stim(metrics_numpy["trial_boundaries"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, metric in zip(axes, ['trial_lick_probs', 'trial_values']):
        for k in plot_keys:
            si = stim_idx[k]
            y  = metrics_numpy[metric][k]
            if len(y) == 0:
                continue
            sm  = pd.Series(y).rolling(smooth, min_periods=1).mean().values
            col = STIM_COLORS_DARK[si]
            ls  = '-' if si in (0, 4) else '--'
            x   = np.array(x_by_stim.get(k, list(range(len(sm)))))[:len(sm)]
            n   = min(len(x), len(sm))
            if n == 0:
                continue
            ax.plot(x[:n], sm[:n], color=col, lw=1.5, ls=ls, label=stim_names[si])
        for ri, rv in enumerate(rev_indices):
            ax.axvline(rv, color='k', ls=':', lw=1.0, label='Reversal' if ri == 0 else None)
            ax.text(rv, 1.01, f"R{ri+1}", fontsize=6, ha="center",
                    transform=ax.get_xaxis_transform(), color="k")
        ax.set_xlim(left=0)
        ax.set_ylabel('Lick probability' if metric == 'trial_lick_probs' else 'Value estimate')
        _xs = []
        for k in plot_keys:
            _xs.extend(list(x_by_stim.get(k, [])))
        if _xs:
            ax.set_xlim(0, max(_xs))
    axes[1].set_xlabel("Global trial index")
    plt.suptitle(
        f'Run {run_idx+1}  seed={seed}  |  {variant_label}  |  {len(rev_indices)} reversals\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)',
        fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=1,
               loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    fig.savefig(run_dir / "lick_value.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_grad_norms(metrics_numpy, run_dir, cfg, variant_label, run_idx, seed, phase_boundaries):
    import pandas as pd
    gn = metrics_numpy['grad_norms']
    if not gn.get('timestep'):
        return
    ts     = np.array(gn['timestep'])
    smooth = 2000
    max_plot_points = 50_000
    rev_ts = phase_boundaries.get('reversal_points', [])

    keys   = ['total', 'rnn', 'rnn_in', 'rnn_rec', 'actor_fc', 'critic_fc']
    labels = ['Total', 'RNN', 'RNN input (W_ih)', 'RNN recurrent (W_hh)', 'Actor FC', 'Critic FC']
    colors_gn = ['#333333', '#1f77b4', '#4c72b0', '#55a868', '#ff7f00', '#2ca02c']

    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.5*len(keys)), sharex=True)
    for ax, k, lbl, col in zip(axes, keys, labels, colors_gn):
        vals = np.array(gn.get(k, []), dtype=float)
        if len(vals) == 0:
            continue
        sm = pd.Series(vals).rolling(smooth, min_periods=1).mean().values
        vals_plot = np.maximum(vals, 1e-12)
        sm_plot   = np.maximum(sm,   1e-12)
        ts_r, vals_r = _subsample_for_plot(ts, vals_plot, max_plot_points)
        ts_s, sm_s   = _subsample_for_plot(ts, sm_plot,   max_plot_points)
        ax.semilogy(ts_r, vals_r, color=col, lw=0.4, alpha=0.2)
        ax.semilogy(ts_s, sm_s,   color=col, lw=1.5, label=lbl)
        for rt in rev_ts:
            ax.axvline(rt, color='k', ls=':', lw=0.8, alpha=0.5)
        ref = max(float(np.mean(sm_plot)) if np.all(np.isfinite(sm_plot)) else float(np.mean(vals_plot)), 1e-12)
        ax.set_ylim(ref / 50.0, ref * 50.0)
        ax.set_ylabel(f'{lbl}\ngrad norm')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Timestep')
    plt.suptitle(f'Gradient norms — Run {run_idx+1} seed={seed}  |  {variant_label}', fontsize=9)
    plt.tight_layout()
    fig.savefig(run_dir / 'grad_norms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_tdr_value_psth(results, run_dir, cfg, trial_structure, plot_stims, variant_label):
    value_axes  = results['value_axes']
    tdr_va_psth = results['tdr_va_psth']

    for axis_col, (axis_key, axis_label) in enumerate([
            ('pre',  'Pre-defined value axis'),
            ('post', 'Post-defined value axis'),
    ]):
        valid_pops = [(n, a) for n, a in value_axes.items() if a is not None]
        if not valid_pops:
            continue
        fig, axes = plt.subplots(len(valid_pops), 1,
                                 figsize=(10, 3.5*len(valid_pops)), squeeze=False)
        for row, (name, ax_info) in enumerate(valid_pops):
            ax = axes[row, 0]
            psth_data = tdr_va_psth.get(name)
            if psth_data is None:
                continue
            phase_data = psth_data[axis_key]
            for s in plot_stims:
                for p in [0, 1]:
                    trajs = phase_data.get((s, p), [])
                    if not trajs:
                        continue
                    min_len = min(t.shape[0] for t in trajs)
                    stacked = np.stack([t[:min_len] for t in trajs])
                    mean_tr = stacked.mean(0)
                    sem_tr  = stacked.std(0) / np.sqrt(len(trajs))
                    col = PHASE_COLOR_MAP[p][s]
                    ls  = PHASE_LS_MAP[p]
                    ax.plot(np.arange(min_len), mean_tr, color=col, lw=1.5, ls=ls,
                            label=f"{STIM_NAMES_MAP[s]} {'pre' if p==0 else 'post'}")
                    ax.fill_between(np.arange(min_len), mean_tr - sem_tr, mean_tr + sem_tr,
                                    color=col, alpha=0.15)
            ax.axhline(0, color='grey', lw=0.6, ls=':')
            ax.set_title(f"{name}  |  angle={ax_info['angle_deg']:.1f}°", fontsize=9)
            ax.legend(fontsize=6, ncol=4)
            ax.set_ylabel('Proj (a.u.)')
        axes[-1, 0].set_xlabel('Timestep within trial')
        plt.suptitle(f'{axis_label}  [{variant_label}]', fontsize=9)
        plt.tight_layout()
        fname = f"tdr_value_psth_{'pre' if axis_col==0 else 'post'}_axis.png"
        fig.savefig(run_dir / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_decoder(decoder_results, run_dir, variant_label):
    pop_list  = list(decoder_results.keys())
    cond_keys = ['pre_cv', 'post_cv', 'pre_to_post', 'post_to_pre']
    cond_cols = ['#aec6e8', '#1f77b4', '#ffb347', '#d62728']
    cond_lbls = ['Pre CV', 'Post CV', 'Pre→Post', 'Post→Pre']
    n_p = len(pop_list)
    fig, axes = plt.subplots(1, n_p, figsize=(4*n_p, 4.5), sharey=True)
    if n_p == 1:
        axes = [axes]
    for ax, name in zip(axes, pop_list):
        res = decoder_results[name]
        for xi, (ck, col, cl) in enumerate(zip(cond_keys, cond_cols, cond_lbls)):
            val = res[ck]
            if hasattr(val, '__len__'):
                ax.bar(xi, val.mean(), color=col, label=cl, alpha=0.85)
                ax.errorbar(xi, val.mean(), yerr=val.std(),
                            fmt='none', color='k', capsize=4, lw=1.5)
            else:
                ax.bar(xi, val, color=col, label=cl, alpha=0.85)
                ax.text(xi, val+0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax.axhline(0.5, color='k', ls='--', lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(cond_lbls, rotation=30, ha='right', fontsize=7)
        ax.set_ylim(0.35, 1.05)
        ax.set_title(name, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Accuracy')
    plt.suptitle(f'Value decoder  [{variant_label}]', fontsize=8.5)
    plt.tight_layout()
    fig.savefig(run_dir / "value_decoder.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


_STIM_STATES   = {'A', 'B', 'C', 'D', 'E', 'F'}
_REWARD_STATES = {'reward_unknown', 'unrewarded', 'rewarded'}
_ITI_STATES    = {'ITI'}


def plot_pca_hidden_states(metrics_numpy, trial_structure, run_dir, cfg,
                           plot_stims, run_idx, seed, variant_label,
                           n_pcs=5, smooth_window=10,
                           plast_off_trial_idx=None):
    """
    Fit PCA on stim-window-averaged hidden states, then plot projections onto
    the first n_pcs PCs as a function of trial index, split by stimulus and
    time window (stim / reward / ITI).

    Layout: n_pcs rows × 3 columns.
    """
    stim_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']
    stim_names  = list('ABCDEF')

    # ── collect per-trial, per-window averages ─────────────────────────────
    seq_idx_list, stim_label_list = [], []
    win_avgs = {'stim': [], 'reward': [], 'iti': []}

    seq = 0
    for tb in metrics_numpy['trial_boundaries']:
        ti = tb['trial_idx']
        hs_list = metrics_numpy['hidden_states'].get(ti)
        ws_list = metrics_numpy['within_trial_states'].get(ti)
        if not hs_list or not ws_list:
            continue
        hs = np.array(hs_list)           # (T, H)
        hidden_size = hs.shape[1]
        sm = np.array([s in _STIM_STATES   for s in ws_list])
        rm = np.array([s in _REWARD_STATES for s in ws_list])
        im = np.array([s in _ITI_STATES    for s in ws_list])
        if sm.sum() == 0:
            continue
        seq_idx_list.append(seq);  seq += 1
        stim_label_list.append(tb['stimulus'])
        win_avgs['stim'].append(hs[sm].mean(0))
        win_avgs['reward'].append(hs[rm].mean(0) if rm.sum() > 0 else np.zeros(hidden_size))
        win_avgs['iti'].append(   hs[im].mean(0) if im.sum() > 0 else np.zeros(hidden_size))

    n_trials = len(seq_idx_list)
    if n_trials < max(n_pcs + 1, 10):
        return

    seq_idx  = np.array(seq_idx_list)
    stim_arr = np.array(stim_label_list)

    # ── PCA fitted on stim-window averages ─────────────────────────────────
    X_stim = np.array(win_avgs['stim'])
    pca = PCA(n_components=n_pcs)
    pca.fit(X_stim)

    projs = {
        'stim':   pca.transform(X_stim),
        'reward': pca.transform(np.array(win_avgs['reward'])),
        'iti':    pca.transform(np.array(win_avgs['iti'])),
    }

    # reversal trial indices for markers
    rev_idxs = _compute_reversal_trial_indices(metrics_numpy['trial_boundaries'])

    window_order = [('stim', 'Stim window'), ('reward', 'Reward window'), ('iti', 'ITI')]

    fig, axes = plt.subplots(n_pcs, 3, figsize=(14, 2.5 * n_pcs), squeeze=False)

    for pc_idx in range(n_pcs):
        var_pct = pca.explained_variance_ratio_[pc_idx] * 100
        for col, (win_key, win_name) in enumerate(window_order):
            ax = axes[pc_idx, col]
            proj = projs[win_key][:, pc_idx]

            for s_idx in plot_stims:
                mask = stim_arr == s_idx
                if mask.sum() == 0:
                    continue
                x = seq_idx[mask]
                y = proj[mask]
                sw = min(smooth_window, len(y))
                y_sm = np.convolve(y, np.ones(sw) / sw, mode='same') if sw > 1 else y
                col_c = stim_colors[s_idx]
                ax.scatter(x, y, color=col_c, s=2, alpha=0.2, zorder=1)
                ax.plot(x, y_sm, color=col_c, lw=1.3, zorder=2,
                        label=stim_names[s_idx] if pc_idx == 0 and col == 0 else None)

            for ri in rev_idxs:
                ax.axvline(ri, color='grey', lw=0.6, ls=':', alpha=0.5)

            if plast_off_trial_idx is not None:
                ax.axvline(plast_off_trial_idx, color='navy', lw=1.4, ls='--')

            ax.axhline(0, color='grey', lw=0.5, ls=':')
            if pc_idx == 0:
                ax.set_title(win_name, fontsize=9)
            ax.set_ylabel(f'PC{pc_idx + 1} ({var_pct:.1f}%)', fontsize=8)
            if pc_idx == n_pcs - 1:
                ax.set_xlabel('Trial index', fontsize=8)

    axes[0, 0].legend(fontsize=7, ncol=3, loc='upper left')
    plt.suptitle(
        f'PCA hidden-state projections — run {run_idx}  seed {seed}\n{variant_label}',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(run_dir / 'pca_hidden_states.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_phase_generalisation_matrices(results, run_dir, variant_label,
                                       max_phases=20, min_trials_per_split=10):
    stim_labels = results["all_stim_labels"]
    phase_idx   = results.get("all_phase_idx")
    if phase_idx is None:
        return
    populations = results.get("populations", {})
    if not populations:
        return

    PRE_HIGH, PRE_LOW   = results["PRE_HIGH"], results["PRE_LOW"]
    POST_HIGH, POST_LOW = results["POST_HIGH"], results["POST_LOW"]
    phases = np.unique(phase_idx)
    phases = phases[phases < max_phases]
    n = len(phases)
    if n < 2:
        return

    def _high_low_for_phase(ph):
        return (PRE_HIGH, PRE_LOW) if int(ph) % 2 == 0 else (POST_HIGH, POST_LOW)

    def _dataset_for_phase(ph, acts):
        high, low = _high_low_for_phase(ph)
        mask = (phase_idx == ph) & np.isin(stim_labels, high + low)
        X = acts[mask]
        y = np.isin(stim_labels[mask], high).astype(int)
        return X, y

    out = {"phases": phases, "by_population": {}}
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for pop_name, acts in populations.items():
        if acts is None:
            continue
        acc = np.full((n, n), np.nan, dtype=float)
        for i, phi in enumerate(phases):
            X_tr, y_tr = _dataset_for_phase(phi, acts)
            if X_tr.shape[0] < min_trials_per_split or len(np.unique(y_tr)) < 2:
                continue
            sc  = StandardScaler()
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
            clf.fit(sc.fit_transform(X_tr), y_tr)
            for j, phj in enumerate(phases):
                if i == j:
                    continue
                X_te, y_te = _dataset_for_phase(phj, acts)
                if X_te.shape[0] < min_trials_per_split or len(np.unique(y_te)) < 2:
                    continue
                acc[i, j] = clf.score(sc.transform(X_te), y_te)
        for i, phi in enumerate(phases):
            X, y = _dataset_for_phase(phi, acts)
            if X.shape[0] < min_trials_per_split or len(np.unique(y)) < 2:
                continue
            clf_cv = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
            ])
            scores = cross_val_score(clf_cv, X, y, cv=cv5, scoring="accuracy")
            acc[i, i] = float(np.mean(scores))

        out["by_population"][pop_name] = {"phase_matrix": acc}

        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
        im = ax.imshow(acc, vmin=0.4, vmax=1.0, cmap="jet", interpolation="nearest")
        ax.set_xticks(range(n)); ax.set_xticklabels([str(int(p)) for p in phases])
        ax.set_yticks(range(n)); ax.set_yticklabels([str(int(p)) for p in phases])
        ax.set_xlabel("Test phase_idx"); ax.set_ylabel("Train phase_idx")
        ax.set_title(
            f"Value decoder phase generalisation\nPopulation={pop_name}  |  [{variant_label}]",
            fontsize=9)
        fig.subplots_adjust(right=0.86)
        cax = fig.add_axes([0.89, 0.15, 0.03, 0.70])
        fig.colorbar(im, cax=cax).set_label("Accuracy")
        fig.savefig(
            run_dir / f"decoder_phase_generalisation_matrix_{pop_name.replace(' ', '_')}.png",
            dpi=200, bbox_inches="tight")
        plt.close(fig)
    return out

# ── Aggregate plots ────────────────────────────────────────────────────────────

def _smooth_in_global_trial_space(y, x, smooth):
    """Rolling mean of length `smooth` applied in per-stimulus trial order.
    Returns (x, smoothed_y) with the same indices — kept for internal use."""
    import pandas as pd
    sm = pd.Series(y).rolling(smooth, min_periods=1).mean().values
    return np.array(x)[:len(sm)], sm


def plot_reward_consumed(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label,
                         plast_off_trial_idx=None):
    """% available reward consumed per trial (sum of received reward / reward_window_length),
    split by stimulus."""
    import pandas as pd
    smooth = 20
    # reward_window_length: infer from first trial that has reward_window data
    rew_win_len = None
    for tb in metrics_numpy['trial_boundaries']:
        ti = tb['trial_idx']
        break
    from cog_nn.tasks.generate_multirev_abef_train_test_phaselen import REWARD_WINDOW
    rew_win_len = REWARD_WINDOW

    stim_names = {0:'A (H→L)',1:'B (H→H)',2:'C (mid)',3:'D (mid)',4:'E (L→H)',5:'F (L→L)'}
    plot_keys  = [k for i, k in enumerate(['A','B','C','D','E','F']) if i in plot_stims]
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}

    rev_indices = _compute_reversal_global_indices(metrics_numpy["trial_boundaries"])
    x_by_stim   = _global_trial_indices_per_stim(metrics_numpy["trial_boundaries"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for k in plot_keys:
        si = stim_idx[k]
        y  = metrics_numpy['trial_reward_consumed'][k]
        if len(y) == 0:
            continue
        y_pct = y / rew_win_len * 100.0
        sm  = pd.Series(y_pct).rolling(smooth, min_periods=1).mean().values
        col = STIM_COLORS_DARK[si]
        ls  = '-' if si in (0, 4) else '--'
        x   = np.array(x_by_stim.get(k, list(range(len(sm)))))[:len(sm)]
        n   = min(len(x), len(sm))
        if n == 0:
            continue
        ax.plot(x[:n], sm[:n], color=col, lw=1.5, ls=ls, label=stim_names[si])
    for ri, rv in enumerate(rev_indices):
        ax.axvline(rv, color='k', ls=':', lw=0.8, alpha=0.7,
                   label='Reversal' if ri == 0 else None)
        ax.text(rv, 1.01, f"R{ri+1}", fontsize=5, ha="center",
                transform=ax.get_xaxis_transform(), color="k")
    if plast_off_trial_idx is not None:
        ax.axvline(plast_off_trial_idx, color=PLAST_OFF_COLOR, ls='--', lw=2.2,
                   label='Plasticity OFF', zorder=6)
    ax.set_xlim(left=0)
    _xs = []
    for k in plot_keys:
        _xs.extend(list(x_by_stim.get(k, [])))
    if _xs:
        ax.set_xlim(0, max(_xs))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Global trial index")
    ax.set_ylabel("% available reward consumed")
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.suptitle(
        f'% Reward consumed — Run {run_idx+1}  seed={seed}  |  {variant_label}\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(run_dir / "reward_consumed_pct.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_reward_lick_count(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label,
                           plast_off_trial_idx=None):
    """Lick count in the reward-availability window per trial, split by stimulus."""
    import pandas as pd
    smooth = 20
    stim_names = {0:'A (H→L)',1:'B (H→H)',2:'C (mid)',3:'D (mid)',4:'E (L→H)',5:'F (L→L)'}
    plot_keys  = [k for i, k in enumerate(['A','B','C','D','E','F']) if i in plot_stims]
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}

    rev_indices = _compute_reversal_global_indices(metrics_numpy["trial_boundaries"])
    x_by_stim   = _global_trial_indices_per_stim(metrics_numpy["trial_boundaries"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for k in plot_keys:
        si = stim_idx[k]
        y  = metrics_numpy['trial_reward_lick_counts'][k]
        if len(y) == 0:
            continue
        sm  = pd.Series(y).rolling(smooth, min_periods=1).mean().values
        col = STIM_COLORS_DARK[si]
        ls  = '-' if si in (0, 4) else '--'
        x   = np.array(x_by_stim.get(k, list(range(len(sm)))))[:len(sm)]
        n   = min(len(x), len(sm))
        if n == 0:
            continue
        ax.plot(x[:n], sm[:n], color=col, lw=1.5, ls=ls, label=stim_names[si])
    for ri, rv in enumerate(rev_indices):
        ax.axvline(rv, color='k', ls=':', lw=0.8, alpha=0.7,
                   label='Reversal' if ri == 0 else None)
        ax.text(rv, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                f"R{ri+1}", fontsize=5, ha="center",
                transform=ax.get_xaxis_transform(), color="k")
    if plast_off_trial_idx is not None:
        ax.axvline(plast_off_trial_idx, color=PLAST_OFF_COLOR, ls='--', lw=2.2,
                   label='Plasticity OFF', zorder=6)
    ax.set_xlim(left=0)
    _xs = []
    for k in plot_keys:
        _xs.extend(list(x_by_stim.get(k, [])))
    if _xs:
        ax.set_xlim(0, max(_xs))
    ax.set_xlabel("Global trial index")
    ax.set_ylabel("Lick count (reward window)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.suptitle(
        f'Reward-window lick count — Run {run_idx+1}  seed={seed}  |  {variant_label}\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(run_dir / "reward_lick_count.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def _agg_per_stim_metric(run_dirs, metric_key, plot_keys, stim_idx, smooth,
                          plast_off_key=None):
    """Load per-stimulus metric from lick_value_data pkls, return aligned mean/sem arrays."""
    import pandas as pd
    all_metrics = []
    for rd in run_dirs:
        pkl = rd / 'lick_value_data.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                all_metrics.append(pickle.load(f))
    return all_metrics


def plot_agg_reward_lick_count(run_dirs, agg_dir, variant_label, plot_stims,
                                plast_off_key='plasticity_off_trial_idx'):
    """Aggregate reward-window lick count across runs."""
    import pandas as pd
    smooth = 20
    stim_names = STIM_NAMES_MAP
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}
    plot_keys  = [k for i, k in enumerate('ABCDEF') if i in plot_stims]

    all_metrics = []
    for rd in run_dirs:
        pkl = rd / 'lick_value_data.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                all_metrics.append(pickle.load(f))
    if not all_metrics:
        _log("  WARNING: no lick_value_data.pkl — skipping agg reward lick count")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    for k in plot_keys:
        si  = stim_idx[k]
        col = STIM_COLORS_DARK[si]
        ls  = '-' if si in (0, 4) else '--'
        series_list = []
        for lv in all_metrics:
            y = lv.get('trial_reward_lick_counts', {}).get(k)
            x = lv.get('global_trial_indices_by_stim', {}).get(k)
            if y is None or len(y) == 0:
                continue
            sm = pd.Series(y).rolling(smooth, min_periods=1).mean().values
            ax.plot(x[:len(sm)] if x is not None and len(x) >= len(sm) else np.arange(len(sm)),
                    sm, color=col, lw=0.6, alpha=0.18, ls=ls)
            idx = np.array(x)[:len(sm)] if x is not None and len(x) >= len(sm) else np.arange(len(sm))
            series_list.append(pd.Series(sm, index=idx))
        if not series_list:
            continue
        all_x = np.unique(np.concatenate([s.index.values for s in series_list])); all_x.sort()
        mat   = np.stack([s.reindex(all_x).to_numpy() for s in series_list])
        mean  = np.nanmean(mat, axis=0)
        ax.plot(all_x, mean, color=col, lw=2.0, ls=ls, label=stim_names[si], zorder=4)

    _add_reversal_markers(ax, all_metrics)
    _add_plasticity_marker(ax, all_metrics, plast_off_key)
    ax.set_xlim(left=0); ax.set_xlabel("Global trial index")
    ax.set_ylabel("Lick count (reward window)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.suptitle(
        f'Reward-window lick count — {len(all_metrics)} runs  [{variant_label}]\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)  |  Light = individual runs',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / "agg_reward_lick_count.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log("  Saved agg_reward_lick_count.png", indent=1)


def plot_agg_reward_consumed(run_dirs, agg_dir, variant_label, plot_stims,
                              plast_off_key='plasticity_off_trial_idx'):
    """Aggregate % available reward consumed across runs."""
    import pandas as pd
    from cog_nn.tasks.generate_multirev_abef_train_test_phaselen import REWARD_WINDOW
    smooth = 20
    stim_names = STIM_NAMES_MAP
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}
    plot_keys  = [k for i, k in enumerate('ABCDEF') if i in plot_stims]

    all_metrics = []
    for rd in run_dirs:
        pkl = rd / 'lick_value_data.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                all_metrics.append(pickle.load(f))
    if not all_metrics:
        _log("  WARNING: no lick_value_data.pkl — skipping agg reward consumed")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    for k in plot_keys:
        si  = stim_idx[k]
        col = STIM_COLORS_DARK[si]
        ls  = '-' if si in (0, 4) else '--'
        series_list = []
        for lv in all_metrics:
            y = lv.get('trial_reward_consumed', {}).get(k)
            x = lv.get('global_trial_indices_by_stim', {}).get(k)
            if y is None or len(y) == 0:
                continue
            y_pct = np.array(y) / REWARD_WINDOW * 100.0
            sm    = pd.Series(y_pct).rolling(smooth, min_periods=1).mean().values
            ax.plot(x[:len(sm)] if x is not None and len(x) >= len(sm) else np.arange(len(sm)),
                    sm, color=col, lw=0.6, alpha=0.18, ls=ls)
            idx = np.array(x)[:len(sm)] if x is not None and len(x) >= len(sm) else np.arange(len(sm))
            series_list.append(pd.Series(sm, index=idx))
        if not series_list:
            continue
        all_x = np.unique(np.concatenate([s.index.values for s in series_list])); all_x.sort()
        mat   = np.stack([s.reindex(all_x).to_numpy() for s in series_list])
        mean  = np.nanmean(mat, axis=0)
        ax.plot(all_x, mean, color=col, lw=2.0, ls=ls, label=stim_names[si], zorder=4)

    _add_reversal_markers(ax, all_metrics)
    _add_plasticity_marker(ax, all_metrics, plast_off_key)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel("Global trial index")
    ax.set_ylabel("% available reward consumed")
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.suptitle(
        f'% Reward consumed — {len(all_metrics)} runs  [{variant_label}]\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)  |  Light = individual runs',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / "agg_reward_consumed_pct.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log("  Saved agg_reward_consumed_pct.png", indent=1)


def _add_reversal_markers(ax, all_metrics):
    rev_lists = [lv.get("rev_indices_global") for lv in all_metrics
                 if lv.get("rev_indices_global") is not None]
    if not rev_lists:
        return
    n_rev = min(len(r) for r in rev_lists)
    for ri in range(n_rev):
        xs  = [r[ri] for r in rev_lists if len(r) > ri]
        med = int(np.median(xs))
        ax.axvline(med, color="k", ls=":", lw=1.2,
                   label="Reversal" if ri == 0 else None, zorder=5, alpha=0.8)
        ax.text(med, 1.01, f"R{ri+1}", fontsize=5, ha="center",
                transform=ax.get_xaxis_transform(), color="k")


def _add_plasticity_marker(ax, all_metrics, key):
    if key is None:
        return
    vals = [lv.get(key) for lv in all_metrics if lv.get(key) is not None]
    if not vals:
        return
    med = int(np.median(vals))
    ax.axvline(med, color=PLAST_OFF_COLOR, ls='--', lw=2.2,
               label='Plasticity OFF', zorder=6)
    ax.text(med, 1.05, "plast. OFF", fontsize=7, ha="center",
            transform=ax.get_xaxis_transform(), color=PLAST_OFF_COLOR, fontweight='bold')


def plot_agg_lick_value(run_dirs, agg_dir, variant_label, plot_stims):
    import pandas as pd
    smooth = 20
    stim_names = STIM_NAMES_MAP
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}

    all_metrics = []
    for rd in run_dirs:
        pkl = rd / 'lick_value_data.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                all_metrics.append(pickle.load(f))
    if not all_metrics:
        _log("  WARNING: no lick_value_data.pkl found — skipping agg lick/value plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    plot_keys = [k for i, k in enumerate('ABCDEF') if i in plot_stims]

    for ax_i, metric in enumerate(['trial_lick_probs', 'trial_values']):
        for k in plot_keys:
            si  = stim_idx[k]
            col = STIM_COLORS_DARK[si]
            ls  = '-' if si in (0, 4) else '--'
            series_list = []
            for lv in all_metrics:
                y = lv.get(metric, {}).get(k)
                x = lv.get("global_trial_indices_by_stim", {}).get(k)
                if y is None or len(y) == 0:
                    continue
                sm = pd.Series(y).rolling(smooth, min_periods=1).mean().values
                axes[ax_i].plot(
                    x[:len(sm)] if x is not None and len(x) >= len(sm) else np.arange(len(sm)),
                    sm, color=col, lw=0.6, alpha=0.18, ls=ls)
                if x is not None and len(x) >= len(sm):
                    series_list.append(pd.Series(sm, index=np.array(x)[:len(sm)]))
                else:
                    series_list.append(pd.Series(sm))
            if not series_list:
                continue
            all_x = np.unique(np.concatenate([s.index.values for s in series_list]))
            all_x.sort()
            mat = np.stack([s.reindex(all_x).to_numpy() for s in series_list])
            mean = np.nanmean(mat, axis=0)
            axes[ax_i].plot(all_x, mean, color=col, lw=2.0, ls=ls, label=stim_names[si], zorder=4)

    rev_lists = [lv.get("rev_indices_global") for lv in all_metrics
                 if lv.get("rev_indices_global") is not None]
    if rev_lists:
        n_rev = min(len(r) for r in rev_lists)
        for ri in range(n_rev):
            xs = [r[ri] for r in rev_lists if len(r) > ri]
            med = int(np.median(xs))
            for ax in axes:
                ax.axvline(med, color="k", ls=":", lw=1.2,
                           label="Reversal" if ri == 0 else None, zorder=5, alpha=0.8)
                ax.text(med, 1.01, f"R{ri+1}", fontsize=6, ha="center",
                        transform=ax.get_xaxis_transform(), color="k")

    axes[0].set_ylabel('Lick probability')
    axes[1].set_ylabel('Value estimate')
    axes[1].set_xlabel("Global trial index")
    for ax in axes:
        ax.set_xlim(left=0)
        if ax.lines:
            xmax = max((np.max(ln.get_xdata()) for ln in ax.lines if len(ln.get_xdata()) > 0), default=None)
            if xmax is not None and np.isfinite(xmax):
                ax.set_xlim(0, float(xmax))
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=1,
               loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.suptitle(
        f'Lick probability and value — {len(all_metrics)} runs  [{variant_label}]\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)  |  Light = individual runs',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / "agg_lick_value.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log("  Saved agg_lick_value.png", indent=1)


def plot_agg_grad_norms(run_dirs, agg_dir, variant_label, phase_boundaries=None):
    import pandas as pd
    all_gn = []
    for rd in run_dirs:
        pkl = rd / 'lick_value_data.pkl'
        if pkl.exists():
            with open(pkl, 'rb') as f:
                lv = pickle.load(f)
            if 'grad_norms' in lv and lv['grad_norms'].get('timestep'):
                all_gn.append(lv['grad_norms'])
    if not all_gn:
        _log("  WARNING: no grad_norms data found — skipping agg grad norm plot")
        return

    phase_boundaries = phase_boundaries or {}
    rev_ts  = phase_boundaries.get("reversal_points", [])
    smooth  = 5000
    max_pts = 50_000
    keys    = ['total', 'rnn', 'actor_fc', 'critic_fc']
    labels  = ['Total', 'RNN', 'Actor FC', 'Critic FC']
    colors_gn = ['#333333', '#1f77b4', '#ff7f00', '#2ca02c']
    min_len = min(len(gn['timestep']) for gn in all_gn)
    ts_ref  = np.array(all_gn[0]['timestep'][:min_len])

    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.5*len(keys)), sharex=True)
    for ax, k, lbl, col in zip(axes, keys, labels, colors_gn):
        mats = []
        for gn in all_gn:
            vals = np.array(gn.get(k, [])[:min_len], dtype=float)
            mats.append(pd.Series(vals).rolling(smooth, min_periods=1).mean().values)
        mat = np.stack(mats)
        mn  = np.maximum(mat.mean(0), 1e-12)
        se  = mat.std(0) / np.sqrt(len(mats))
        lo  = np.maximum(mn - se, 1e-12)
        hi  = np.maximum(mn + se, 1e-12)
        ts_m, mn_s = _subsample_for_plot(ts_ref, mn, max_pts)
        _, lo_s    = _subsample_for_plot(ts_ref, lo, max_pts)
        _, hi_s    = _subsample_for_plot(ts_ref, hi, max_pts)
        ax.semilogy(ts_m, mn_s, color=col, lw=1.5, label=lbl)
        ax.fill_between(ts_m, lo_s, hi_s, color=col, alpha=0.2)
        for ri, rt in enumerate(rev_ts):
            ax.axvline(rt, color="k", ls=":", lw=0.8, alpha=0.5)
            if ri < 12:
                ax.text(rt, 1.01, f"R{ri+1}", fontsize=6, ha="center",
                        transform=ax.get_xaxis_transform(), color="k")
        ref = max(float(np.mean(mn)) if np.all(np.isfinite(mn)) else 1e-12, 1e-12)
        ax.set_ylim(ref / 50.0, ref * 50.0)
        ax.set_ylabel(f'{lbl}\ngrad norm')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Timestep')
    plt.suptitle(f'Gradient norms — {len(all_gn)} runs [{variant_label}]', fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / 'agg_grad_norms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log("  Saved agg_grad_norms.png", indent=1)


def aggregate_plots(run_dirs, agg_dir, cfg, variant_label, plot_stims, phase_boundaries):
    all_results = []
    for rd in run_dirs:
        pkl = rd / "run_results.pkl"
        if not pkl.exists():
            _log(f"WARNING: {pkl} missing, skipping")
            continue
        with open(pkl, 'rb') as f:
            all_results.append(pickle.load(f))
    if not all_results:
        _log("WARNING: No results found for aggregation.")
        return

    n_runs = len(all_results)
    _log(f"Aggregating {n_runs} runs → {agg_dir}")
    agg_dir.mkdir(parents=True, exist_ok=True)

    pop_list  = list(all_results[0]['decoder_results'].keys())
    cond_keys = ['pre_cv', 'post_cv', 'pre_to_post', 'post_to_pre']
    cond_cols = ['#aec6e8', '#1f77b4', '#ffb347', '#d62728']
    cond_lbls = ['Pre CV', 'Post CV', 'Pre→Post', 'Post→Pre']
    n_p = len(pop_list)
    fig, axes = plt.subplots(1, n_p, figsize=(4*n_p, 4.5), sharey=True)
    if n_p == 1:
        axes = [axes]
    for ax, name in zip(axes, pop_list):
        per_run_means = {ck: [] for ck in cond_keys}
        for res in all_results:
            dr = res['decoder_results'].get(name, {})
            for ck in cond_keys:
                val = dr.get(ck, np.nan)
                per_run_means[ck].append(float(np.mean(val)) if hasattr(val, '__len__') else float(val))
        for xi, (ck, col, cl) in enumerate(zip(cond_keys, cond_cols, cond_lbls)):
            vals = np.array(per_run_means[ck])
            m, se = vals.mean(), vals.std() / np.sqrt(len(vals))
            ax.bar(xi, m, color=col, label=cl, alpha=0.85)
            ax.errorbar(xi, m, yerr=se, fmt='none', color='k', capsize=4, lw=1.5)
            jitter = np.random.default_rng(xi).normal(0, 0.07, len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals, s=12, color='k', alpha=0.4, zorder=3)
        ax.axhline(0.5, color='k', ls='--', lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(cond_lbls, rotation=30, ha='right', fontsize=7)
        ax.set_ylim(0.35, 1.05)
        ax.set_title(name, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Accuracy')
    plt.suptitle(f'Value decoder — {n_runs} runs  [{variant_label}]', fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / "agg_value_decoder.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log("  Saved agg_value_decoder.png", indent=1)

    # Aggregate TDR value PSTH
    for axis_key, axis_label in [('pre', 'Pre-defined value axis'),
                                  ('post', 'Post-defined value axis')]:
        valid_pop_names = [n for n in pop_list
                           if all_results[0]['tdr_va_psth'].get(n) is not None]
        if not valid_pop_names:
            continue
        n_vp = len(valid_pop_names)
        fig, axes = plt.subplots(n_vp, 1, figsize=(10, 3.5*n_vp), squeeze=False)
        for row, name in enumerate(valid_pop_names):
            ax = axes[row, 0]
            combo_data = defaultdict(list)
            for res in all_results:
                psth_data = res['tdr_va_psth'].get(name)
                if psth_data is None:
                    continue
                phase_data = psth_data[axis_key]
                for s in plot_stims:
                    for p in [0, 1]:
                        trajs = phase_data.get((s, p), [])
                        if not trajs:
                            continue
                        min_len = min(t.shape[0] for t in trajs)
                        combo_data[(s, p)].append(np.stack([t[:min_len] for t in trajs]).mean(0))
            for s in plot_stims:
                for p in [0, 1]:
                    run_means = combo_data.get((s, p), [])
                    if not run_means:
                        continue
                    min_len  = min(t.shape[0] for t in run_means)
                    mat      = np.stack([t[:min_len] for t in run_means])
                    grand_mean = mat.mean(0)
                    sem        = mat.std(0) / np.sqrt(len(mat))
                    col = PHASE_COLOR_MAP[p][s]
                    ls  = PHASE_LS_MAP[p]
                    ax.plot(np.arange(min_len), grand_mean, color=col, lw=1.5, ls=ls,
                            label=f"{STIM_NAMES_MAP[s]} {'pre' if p==0 else 'post'}")
                    ax.fill_between(np.arange(min_len), grand_mean - sem, grand_mean + sem,
                                    color=col, alpha=0.15)
            ax.axhline(0, color='grey', lw=0.6, ls=':')
            # angle across runs
            angles = [res['value_axes'].get(name, {}).get('angle_deg', float('nan'))
                      for res in all_results
                      if res.get('value_axes', {}).get(name) is not None]
            angle_str = f"{np.nanmean(angles):.1f}°±{np.nanstd(angles):.1f}°" if angles else "N/A"
            ax.set_title(f"{name}  |  angle={angle_str} (mean±SD)", fontsize=9)
            ax.legend(fontsize=6, ncol=4)
            ax.set_ylabel('Proj (a.u.)')
        axes[-1, 0].set_xlabel('Timestep within trial')
        plt.suptitle(f'{axis_label}  [{variant_label}]  ({n_runs} runs)', fontsize=9)
        plt.tight_layout()
        fname = f"agg_tdr_value_psth_{'pre' if axis_key=='pre' else 'post'}_axis.png"
        fig.savefig(agg_dir / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
    _log("  Saved aggregate TDR PSTH plots", indent=1)

    # Aggregate phase generalisation matrices
    objs = []
    for rd in run_dirs:
        pkl = rd / "decoder_phase_generalisation_matrices.pkl"
        if pkl.exists():
            with open(pkl, 'rb') as f:
                objs.append(pickle.load(f))
    if objs:
        phases_ref = objs[0].get("phases")
        if phases_ref is not None and all(
            o.get("phases") is not None
            and np.array_equal(o.get("phases"), phases_ref) for o in objs
        ):
            pop_names = sorted(objs[0].get("by_population", {}).keys())
            for pop in pop_names:
                mats_all = [o["by_population"][pop]["phase_matrix"]
                            for o in objs if o.get("by_population", {}).get(pop)]
                if not mats_all:
                    continue
                mean_mat = np.nanmean(np.stack(mats_all), axis=0)
                n = mean_mat.shape[0]
                fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
                im = ax.imshow(mean_mat, vmin=0.4, vmax=1.0, cmap="viridis", interpolation="nearest")
                ax.set_xticks(range(n)); ax.set_xticklabels([str(int(p)) for p in phases_ref])
                ax.set_yticks(range(n)); ax.set_yticklabels([str(int(p)) for p in phases_ref])
                ax.set_xlabel("Test phase_idx"); ax.set_ylabel("Train phase_idx")
                ax.set_title(
                    f"Value decoder phase generalisation (mean across runs)\n"
                    f"Population={pop}  |  [{variant_label}]", fontsize=9)
                fig.subplots_adjust(right=0.86)
                cax = fig.add_axes([0.89, 0.15, 0.03, 0.70])
                fig.colorbar(im, cax=cax).set_label("Accuracy")
                fig.savefig(
                    agg_dir / f"agg_decoder_phase_generalisation_matrix_{pop.replace(' ', '_')}.png",
                    dpi=200, bbox_inches="tight")
                plt.close(fig)
            _log("  Saved aggregate phase generalisation matrices", indent=1)

    # Aggregate lick/value and grad norms
    plot_agg_lick_value(run_dirs, agg_dir, variant_label, plot_stims)
    plot_agg_reward_lick_count(run_dirs, agg_dir, variant_label, plot_stims)
    plot_agg_reward_consumed(run_dirs, agg_dir, variant_label, plot_stims)
    plot_agg_grad_norms(run_dirs, agg_dir, variant_label, phase_boundaries)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_runs',    type=int, default=10)
    parser.add_argument('--n_reversals',     type=int, default=10,
                        help='Number of reversals (= n_phases/2). Each reversal adds 2 phases.')
    parser.add_argument('--trials_per_phase', type=int, default=400)
    parser.add_argument('--task_seed', type=int, default=42,
                        help='Seed used to generate task data.')
    parser.add_argument('--freeze_mode', default='none',
                        choices=['none', 'readout_only', 'rnn_only', 'all'])
    parser.add_argument('--hidden_size',      type=int,   default=128)
    parser.add_argument('--readout_fraction', type=float, default=0.5,
                        help='Fraction of hidden units projecting to readout')
    parser.add_argument('--learning_rate',    type=float, default=5e-4)
    parser.add_argument('--gamma',            type=float, default=0.0)
    parser.add_argument('--policy_clip',      type=float, default=0.25)
    parser.add_argument('--tdr_pre_start',    type=int,   default=50)
    parser.add_argument('--tdr_pre_end',      type=int,   default=-1)
    parser.add_argument('--tdr_post_start',   type=int,   default=50)
    parser.add_argument('--tdr_post_end',     type=int,   default=-1)
    parser.add_argument('--exclude_mid_stim', type=int,   default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--run_indices', type=str, default='',
                        help='Comma-separated run indices to execute (e.g. "0,3,7")')
    parser.add_argument('--results_dir',  type=str, default='../results/08_04_2026_context_signal_abef')
    parser.add_argument('--task_data_dir', type=str, default='../task_data')
    args = parser.parse_args()

    freeze_mode  = args.freeze_mode
    n_runs       = args.n_runs
    hidden_size  = args.hidden_size
    readout_size = max(1, int(hidden_size * args.readout_fraction))
    n_phases     = args.n_reversals * 2  # each reversal = 2 phases (pre → post)

    tdr_pre_end  = None if args.tdr_pre_end  < 0 else args.tdr_pre_end
    tdr_post_end = None if args.tdr_post_end < 0 else args.tdr_post_end

    cfg = {
        'freeze_mode':    freeze_mode,
        'hidden_size':    hidden_size,
        'readout_size':   readout_size,
        'n_contexts':     N_CONTEXTS,
        'learning_rate':  args.learning_rate,
        'gamma':          args.gamma,
        'policy_clip':    args.policy_clip,
        'tdr_pre_block':  (args.tdr_pre_start,  tdr_pre_end),
        'tdr_post_block': (args.tdr_post_start, tdr_post_end),
        'use_context_signal': True,
    }

    _freeze_labels = {
        'none': 'full plasticity', 'readout_only': 'readout only',
        'rnn_only': 'RNN only', 'all': 'frozen',
    }
    freeze_label = _freeze_labels[freeze_mode]
    _ro_str      = f'partialro{readout_size}'
    variant_name = (f'context_signal_abef_{args.n_reversals}rev_'
                    f'tpp{args.trials_per_phase}_{_ro_str}_frozen{freeze_mode}')
    variant_label = f'{variant_name}  [{freeze_label}]'

    results_dir = Path(args.results_dir)
    variant_dir = results_dir / variant_name
    agg_dir     = variant_dir / 'aggregate'
    variant_dir.mkdir(parents=True, exist_ok=True)

    plot_stims = [s for s in range(6) if not (args.exclude_mid_stim and s in (2, 3))]

    # ── Generate / load task data ──────────────────────────────────────────────
    task_path = ensure_task_data(
        args.task_data_dir, n_phases, args.trials_per_phase, args.task_seed)
    (state_sequence, reward_sequence, reversal_mask,
     phase_boundaries, trial_structure, state_map) = load_reversal_abcdef_multitimestep_multirev_data(
        task_path)
    _log(f"Task loaded: {state_sequence.shape[0]} timesteps, {len(trial_structure)} trials")

    # ── Save config CSV ────────────────────────────────────────────────────────
    with open(variant_dir / 'config.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'value'])
        for k, v in cfg.items():
            w.writerow([k, v])
        w.writerow(['n_runs', n_runs])
        w.writerow(['n_reversals', args.n_reversals])
        w.writerow(['trials_per_phase', args.trials_per_phase])
        w.writerow(['variant_name', variant_name])

    post_end    = phase_boundaries['post_reversal']['end']
    n_trainable = sum(p.numel() for p in build_model(10, 2, cfg).parameters() if p.requires_grad)
    _log(f"Variant: {variant_name}")
    _log(f"  freeze={freeze_mode}  hidden={hidden_size}  readout={readout_size}"
         f"  n_contexts={N_CONTEXTS}  lr={cfg['learning_rate']}")
    _log(f"  {n_runs} runs × {post_end:,} timesteps  |  {n_trainable:,} trainable params")
    _log(f"  Results → {variant_dir}")

    # Parse optional run_indices filter
    if args.run_indices.strip():
        run_indices = sorted({int(x) for x in args.run_indices.split(",") if x.strip()})
    else:
        run_indices = list(range(n_runs))

    run_dirs = []
    for run_idx in run_indices:
        seed    = run_idx * 7 + 42
        run_dir = variant_dir / f'run_{run_idx:02d}_seed{seed}'
        run_dir.mkdir(exist_ok=True)
        run_dirs.append(run_dir)

        _log("")
        _log(f"{'─'*56}")
        _log(f"Run {run_idx+1}/{n_runs}  seed={seed}  [{freeze_label}]")
        _log(f"{'─'*56}")

        if not args.overwrite and (run_dir / 'run_results.pkl').exists():
            _log("  Skipping — run_results.pkl exists (use --overwrite to re-run)", indent=1)
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        _t_run = time.time()

        env = ReversalABCDEFMultiTimestepEnv(
            state_sequence, reward_sequence, reversal_mask, trial_structure,
            reward_lick=1.0, lick_no_reward=-1.0, no_lick=0.0)
        state_size  = env.observation_space.shape[0]
        action_size = env.action_space.n
        model = build_model(state_size, action_size, cfg)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        _log(f"  Model: {n_train:,} / {n_total:,} params trainable", indent=1)

        _log(f"  Training ({post_end:,} steps)...", indent=1)
        _t_train = time.time()
        metrics_numpy, agent = train(env, model, cfg, trial_structure, phase_boundaries, seed)
        n_trials    = len(metrics_numpy['trial_boundaries'])
        pre_trials  = sum(1 for tb in metrics_numpy['trial_boundaries'] if tb['reversal_phase'] == 0)
        post_trials = n_trials - pre_trials
        _log(f"  Training done in {time.time()-_t_train:.0f}s  "
             f"({n_trials} trials: {pre_trials} pre, {post_trials} post)", indent=1)

        _log("  Running analysis...", indent=1)
        results = analyse(metrics_numpy, model, cfg, trial_structure)

        _log("  Saving per-run plots...", indent=1)
        plot_lick_value(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label)
        _log("    lick_value.png", indent=2)
        plot_grad_norms(metrics_numpy, run_dir, cfg, variant_label, run_idx, seed, phase_boundaries or {})
        _log("    grad_norms.png", indent=2)
        plot_tdr_value_psth(results, run_dir, cfg, trial_structure, plot_stims, variant_label)
        _log("    tdr_value_psth_*.png", indent=2)
        plot_decoder(results['decoder_results'], run_dir, variant_label)
        _log("    value_decoder.png", indent=2)
        plot_reward_lick_count(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label)
        _log("    reward_lick_count.png", indent=2)
        plot_reward_consumed(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label)
        _log("    reward_consumed_pct.png", indent=2)
        plot_pca_hidden_states(metrics_numpy, trial_structure, run_dir, cfg,
                               plot_stims, run_idx, seed, variant_label, n_pcs=5)
        _log("    pca_hidden_states.png", indent=2)
        mat_obj = plot_phase_generalisation_matrices(results, run_dir, variant_label)
        if mat_obj is not None:
            with open(run_dir / "decoder_phase_generalisation_matrices.pkl", "wb") as f:
                pickle.dump(mat_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            _log("    decoder_phase_generalisation_matrix_*.png", indent=2)

        # Save lightweight lick/value data for aggregate plots
        lick_value_data = {
            'trial_lick_probs':        {k: metrics_numpy['trial_lick_probs'][k]         for k in 'ABCDEF'},
            'trial_values':            {k: metrics_numpy['trial_values'][k]              for k in 'ABCDEF'},
            'trial_reward_lick_counts': {k: metrics_numpy['trial_reward_lick_counts'][k] for k in 'ABCDEF'},
            'trial_reward_consumed':    {k: metrics_numpy['trial_reward_consumed'][k]    for k in 'ABCDEF'},
            'rev_idx':          sum(1 for tb in metrics_numpy['trial_boundaries']
                                    if tb['stimulus'] == 0 and tb['reversal_phase'] == 0),
            'rev_indices':      _compute_reversal_trial_indices(metrics_numpy['trial_boundaries']),
            'rev_indices_global': _compute_reversal_global_indices(metrics_numpy['trial_boundaries']),
            'global_trial_indices_by_stim': _global_trial_indices_per_stim(metrics_numpy['trial_boundaries']),
            'grad_norms': metrics_numpy.get('grad_norms', {}),
        }
        with open(run_dir / 'lick_value_data.pkl', 'wb') as f:
            pickle.dump(lick_value_data, f)

        # Save full metrics for later analysis
        with open(run_dir / "metrics_numpy.pkl", "wb") as f:
            pickle.dump(metrics_numpy, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save run results
        run_results = {
            'decoder_results':  results['decoder_results'],
            'value_axes':       results['value_axes'],
            'tdr_projs':        results['tdr_projs'],
            'tdr_va_psth':      results['tdr_va_psth'],
            'all_stim_labels':  results['all_stim_labels'],
            'all_phase_labels': results['all_phase_labels'],
            'all_phase_idx':    results.get('all_phase_idx'),
            'pre_conv_mask':    results['pre_conv_mask'],
            'post_conv_mask':   results['post_conv_mask'],
            'PRE_HIGH':  results['PRE_HIGH'],  'PRE_LOW':  results['PRE_LOW'],
            'POST_HIGH': results['POST_HIGH'], 'POST_LOW': results['POST_LOW'],
            'readout_weights': {
                'actor_fc_weight':  model.actor_fc.weight.detach().cpu().numpy(),
                'actor_fc_bias':    model.actor_fc.bias.detach().cpu().numpy(),
                'critic_fc_weight': model.critic_fc.weight.detach().cpu().numpy(),
                'critic_fc_bias':   model.critic_fc.bias.detach().cpu().numpy(),
                'readout_indices':  list(range(cfg['readout_size'])),
            },
            'params': {**cfg, 'seed': seed},
        }
        with open(run_dir / 'run_results.pkl', 'wb') as f:
            pickle.dump(run_results, f)
        _log(f"  Run {run_idx+1} complete in {time.time()-_t_run:.0f}s  →  {run_dir.name}", indent=1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    _log("")
    _log(f"{'─'*56}")
    _log(f"Computing aggregate plots ({len(run_dirs)} runs)...")
    aggregate_plots(run_dirs, agg_dir, cfg, variant_label, plot_stims, phase_boundaries)
    total = time.time() - _script_start
    _log(f"All done in {total/60:.1f} min.  Results in: {variant_dir}")


if __name__ == '__main__':
    main()

"""
Multi-run training + analysis script for Meta-AC reversal ABCDEF partial-reversal.

Trains multiple seeds, saves per-run plots in subfolders, then produces
aggregate TDR value-projection and decoder cross-generalisation plots.

Usage:
    python run_meta_ac_abcdef_partial_multirun.py \
        --freeze_mode none \
        --use_prev_action_reward 0 \
        --n_runs 10 \
        --results_dir ../results/20_03_26_value_subspace_experiments

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
matplotlib.use("Agg")  # non-interactive backend for scripts
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

# ── Logging helper ────────────────────────────────────────────────────────────

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
    load_reversal_abcdef_multitimestep_data,
    load_reversal_abcdef_multitimestep_multirev_data,
)
from cog_nn.models import RNNActorCriticPartialReadout
from cog_nn.agents import MetaA2CAgent

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _lighten(hex_color, amount=0.55):
    c = np.array(mc.to_rgb(hex_color))
    return tuple(c + amount * (np.ones(3) - c))

STIM_COLORS_DARK  = {0:'#e41a1c',1:'#ff7f00',2:'#4daf4a',3:'#377eb8',4:'#984ea3',5:'#a65628'}
STIM_COLORS_LIGHT = {k: _lighten(v) for k, v in STIM_COLORS_DARK.items()}  # kept for compat
# pre=dashed, post=solid — same hue for both phases, linestyle encodes phase
PHASE_COLOR_MAP   = {0: STIM_COLORS_DARK, 1: STIM_COLORS_DARK}
PHASE_LS_MAP      = {0: '--', 1: '-'}
STIM_NAMES_MAP    = {0:'A (H→L)',1:'B (H→H)',2:'C (mid)',3:'D (mid)',4:'E (L→H)',5:'F (L→L)'}
POP_COLORS        = {'Full':'#444444','Projecting':'#1f77b4',
                     'Non-proj':'#d62728','Readout act':'#2ca02c'}

# ── Build model ────────────────────────────────────────────────────────────────

def build_model(state_size, action_size, cfg):
    hidden_size   = cfg['hidden_size']
    readout_size  = cfg['readout_size']
    freeze_mode   = cfg['freeze_mode']
    learning_rate = cfg['learning_rate']
    gamma         = cfg['gamma']

    readout_indices = list(range(readout_size))
    model = RNNActorCriticPartialReadout(
        state_size=state_size,
        action_size=action_size,
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

def train(env, model, cfg, trial_structure, phase_boundaries, seed):
    hidden_size    = cfg['hidden_size']
    action_size    = env.action_space.n
    state_size     = env.observation_space.shape[0]
    freeze_mode    = cfg['freeze_mode']
    use_prev_ar    = cfg['use_prev_action_reward']
    clip           = cfg['policy_clip']

    stim_keys   = ['A', 'B', 'C', 'D', 'E', 'F']
    idx_to_stim = {i: k for i, k in enumerate(stim_keys)}

    metrics = {
        'lick_probs':  {k: [] for k in stim_keys},
        'values':      {k: [] for k in stim_keys},
        'timesteps':   {k: [] for k in stim_keys},
        'trial_lick_probs': {k: [] for k in stim_keys},
        'trial_values':     {k: [] for k in stim_keys},
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
        'grad_norms': {'rnn': [], 'actor_fc': [], 'critic_fc': [], 'total': [], 'timestep': []},
    }

    agent = MetaA2CAgent(state_size=state_size, action_size=action_size, hidden_size=hidden_size)
    agent.model = model

    pre_start  = 0
    post_end   = phase_boundaries['post_reversal']['end']

    model.train()
    agent.reset_hidden_state()
    obs, info = env.reset()
    prev_action = torch.zeros(action_size)
    prev_reward = torch.tensor(0.0)

    states_b, pa_b, pr_b, act_b, rew_b, ns_b, npa_b, npr_b, done_b = [[] for _ in range(9)]

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
                    'trial_idx': trial_info, 'trial_start': td['trial_start'],
                    'trial_end': td['trial_end'], 'stimulus': td['stimulus'],
                    'reversal_phase': td['reversal_phase'],
                    'phase_idx': td.get('phase_idx', td['reversal_phase']),
                    'reward_available': td['reward_available'],
                })
                for key in ('within_trial_lick_probs','within_trial_values',
                            'within_trial_timesteps','within_trial_states','hidden_states'):
                    metrics[key][trial_info] = []
            if trial_info in metrics['within_trial_lick_probs']:
                metrics['within_trial_lick_probs'][trial_info].append(lick_p)
                metrics['within_trial_values'][trial_info].append(float(value))
                metrics['within_trial_timesteps'][trial_info].append(t_idx)
                metrics['within_trial_states'][trial_info].append(info.get('state_name','?'))
                metrics['hidden_states'][trial_info].append(rnn_out.cpu().numpy())
            if t_idx == td['stim_window'][-1]:
                k = idx_to_stim.get(td['stimulus'])
                if k:
                    metrics['trial_lick_probs'][k].append(lick_p)
                    metrics['trial_values'][k].append(float(value))
                metrics['trial_timesteps'].append(t_idx)
                metrics['trial_indices'].append(trial_info)
                metrics['trial_reversal_phases'].append(td['reversal_phase'])
        if info.get('reward_available', False):
            metrics['rewards'].append(reward)
            metrics['reward_timesteps'].append(t_idx)

    LOG_EVERY = 10_000
    _win_rewards, _win_trials, _win_lick_p = [], [], []  # rolling window for log
    _last_log_t = pre_start

    # Build phase lookup for logging (supports multi-reversal)
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

    for t_idx in range(pre_start, post_end):
        state = torch.from_numpy(obs).float()
        _pa = prev_action if use_prev_ar else torch.zeros(action_size)
        _pr = prev_reward if use_prev_ar else torch.tensor(0.0)
        action, action_prob, value, rnn_out = agent.select_action(
            state, _pa, _pr, deterministic=False, policy_clip=clip, return_rnn_out=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track reward & lick prob for progress logging
        _win_rewards.append(reward)
        _win_lick_p.append(float(action_prob if action == 0 else 1.0 - action_prob))
        if info.get('trial_idx') is not None:
            td = trial_structure[info['trial_idx']]
            if t_idx == td.get('trial_start', -1):
                _win_trials.append(td.get('reversal_phase', 0))

        record_step(t_idx, obs, action, action_prob, value, rnn_out, reward, info)
        next_prev_action = torch.zeros(action_size)
        next_prev_action[action] = 1.0
        if freeze_mode != "all":
            states_b.append(obs)
            pa_b.append(_pa.numpy())
            pr_b.append(_pr.item())
            act_b.append(action)
            rew_b.append(reward)
            ns_b.append(next_obs)
            npa_b.append(next_prev_action.numpy() if use_prev_ar else np.zeros(action_size))
            npr_b.append(reward if use_prev_ar else 0.0)
            done_b.append(done)
            if len(states_b) >= 1:
                _upd = agent.update(
                    torch.from_numpy(np.array(states_b)).float(),
                    torch.from_numpy(np.array(pa_b)).float(),
                    torch.from_numpy(np.array(pr_b)).float(),
                    torch.from_numpy(np.array(act_b)).long(),
                    torch.from_numpy(np.array(rew_b)).float(),
                    torch.from_numpy(np.array(ns_b)).float(),
                    torch.from_numpy(np.array(npa_b)).float(),
                    torch.from_numpy(np.array(npr_b)).float(),
                    torch.from_numpy(np.array(done_b)).float())
                if _upd and 'grad_norms' in _upd:
                    for gk in ('rnn','actor_fc','critic_fc','total'):
                        metrics['grad_norms'][gk].append(_upd['grad_norms'][gk])
                    metrics['grad_norms']['timestep'].append(t_idx)
                states_b,pa_b,pr_b,act_b,rew_b,ns_b,npa_b,npr_b,done_b = [[] for _ in range(9)]
        obs = next_obs
        prev_action = next_prev_action
        prev_reward = torch.tensor(reward, dtype=torch.float32)

        if (t_idx - _last_log_t) >= LOG_EVERY or t_idx == pre_start:
            pct      = 100 * (t_idx - pre_start) / max(post_end - pre_start, 1)
            phase    = _phase_label(t_idx)
            rew_rate = np.mean(_win_rewards) if _win_rewards else 0.0
            lick_p   = np.mean(_win_lick_p)  if _win_lick_p  else 0.0
            n_trials = len(metrics['trial_boundaries'])
            _log(f"  step {t_idx:>7d}/{post_end}  ({pct:4.1f}%)  phase={phase}"
                 f"  trials={n_trials:5d}"
                 f"  rew_rate={rew_rate:+.3f}  lick_p={lick_p:.3f}", indent=1)
            _win_rewards.clear(); _win_lick_p.clear(); _win_trials.clear()
            _last_log_t = t_idx

    # Convert to numpy
    metrics_numpy = {
        'lick_probs':  {k: np.array(v) for k, v in metrics['lick_probs'].items()},
        'values':      {k: np.array(v) for k, v in metrics['values'].items()},
        'timesteps':   {k: np.array(v) for k, v in metrics['timesteps'].items()},
        'trial_lick_probs': {k: np.array(v) for k, v in metrics['trial_lick_probs'].items()},
        'trial_values':     {k: np.array(v) for k, v in metrics['trial_values'].items()},
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
    all_activations, all_stim_labels, all_phase_labels = [], [], []
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

    all_activations  = np.array(all_activations)
    all_stim_labels  = np.array(all_stim_labels)
    all_phase_labels = np.array(all_phase_labels)
    _log(f"  Analysis: {len(all_activations)} trials extracted  "
         f"(pre={int((all_phase_labels==0).sum())}, post={int((all_phase_labels==1).sum())})",
         indent=1)

    act_proj    = all_activations[:, readout_indices]
    act_nonproj = all_activations[:, non_readout_indices] if non_readout_indices else None
    readout_all = np.hstack([act_proj @ W_actor.T  + b_actor,
                             act_proj @ W_critic.T + b_critic])

    populations = OrderedDict()
    populations['Full']        = all_activations
    populations['Projecting']  = act_proj
    if non_readout_indices:
        populations['Non-proj'] = act_nonproj
    populations['Readout act'] = readout_all

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

    def _convergence_mask(stim_labels, phase_labels, phase, block):
        start, end = block
        mask = np.zeros(len(stim_labels), dtype=bool)
        for s in range(6):
            idx = np.where((stim_labels == s) & (phase_labels == phase))[0]
            mask[idx[start:end]] = True
        return mask

    pre_conv_mask  = _convergence_mask(all_stim_labels, all_phase_labels, 0, tdr_pre_blk)
    post_conv_mask = _convergence_mask(all_stim_labels, all_phase_labels, 1, tdr_post_blk)
    _log(f"  Analysis: converged trials — pre={pre_conv_mask.sum()}, post={post_conv_mask.sum()}", indent=1)

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
    # Collect per-trial PSTH projections onto value axes
    def _get_pop_hidden(h, name):
        h = h.squeeze()
        if name == 'Projecting':
            return h[readout_indices]
        elif name == 'Non-proj':
            return h[non_readout_indices]
        elif name == 'Readout act':
            al = h[readout_indices] @ W_actor.T + b_actor
            cv = h[readout_indices] @ W_critic.T + b_critic
            return np.concatenate([al, cv])
        return h

    _log("  Analysis: TDR value PSTH (time-resolved projections)...", indent=1)
    tdr_va_psth = {}   # name -> axis_key ('pre'/'post') -> (s, p) -> list[traj arrays]
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

    decoder_results = {}
    for name, acts in populations.items():
        X_pre,  y_pre  = _value_dataset(acts, 0, PRE_HIGH,  PRE_LOW,  pre_conv_mask)
        X_post, y_post = _value_dataset(acts, 1, POST_HIGH, POST_LOW, post_conv_mask)
        clf = Pipeline([('sc', StandardScaler()),
                        ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=42))])
        cv_pre  = cross_val_score(clf, X_pre,  y_pre,  cv=cv5, scoring='accuracy')
        cv_post = cross_val_score(clf, X_post, y_post, cv=cv5, scoring='accuracy')
        sc1 = StandardScaler(); lr1 = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        lr1.fit(sc1.fit_transform(X_pre), y_pre)
        p2p = lr1.score(sc1.transform(X_post), y_post)
        sc2 = StandardScaler(); lr2 = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        lr2.fit(sc2.fit_transform(X_post), y_post)
        p2pr = lr2.score(sc2.transform(X_pre), y_pre)
        decoder_results[name] = {
            'pre_cv': cv_pre, 'post_cv': cv_post,
            'pre_to_post': p2p, 'post_to_pre': p2pr,
        }
        _log(f"  [{name:15s}] pre={cv_pre.mean():.3f}±{cv_pre.std():.3f}  "
             f"post={cv_post.mean():.3f}±{cv_post.std():.3f}  "
             f"pre→post={p2p:.3f}  post→pre={p2pr:.3f}", indent=1)

    _log(f"  Analysis complete  ({time.time()-_t0:.1f}s)", indent=1)
    return {
        'populations':       populations,
        'all_stim_labels':   all_stim_labels,
        'all_phase_labels':  all_phase_labels,
        'pre_conv_mask':     pre_conv_mask,
        'post_conv_mask':    post_conv_mask,
        'tdr_projs':         tdr_projs,
        'tdr_va_psth':       tdr_va_psth,
        'value_axes':        value_axes,
        'decoder_results':   decoder_results,
        'all_coefs':         all_coefs,
        'trial_structure':   trial_structure,
        # also pass through key arrays for aggregate
        'PRE_HIGH': PRE_HIGH, 'PRE_LOW': PRE_LOW,
        'POST_HIGH': POST_HIGH, 'POST_LOW': POST_LOW,
    }

# ── Per-run plots ──────────────────────────────────────────────────────────────

def _compute_reversal_trial_indices(trial_boundaries):
    """
    Return list of per-stimulus-A trial indices at which each phase change occurs.

    For a single-reversal task this returns a 1-element list [rev_idx].
    For a multi-reversal task it returns one entry per phase boundary.
    """
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


def plot_lick_value(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, freeze_label):
    """Lick probability and value estimate over trials, per stimulus."""
    import pandas as pd
    smooth = 20
    stim_names = {0:'A (H→L)',1:'B (H→H)',2:'C (mid)',3:'D (mid)',4:'E (L→H)',5:'F (L→L)'}
    plot_keys  = [k for i, k in enumerate(['A','B','C','D','E','F']) if i in plot_stims]
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}

    rev_indices = _compute_reversal_trial_indices(metrics_numpy['trial_boundaries'])

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
            ax.plot(np.arange(len(sm)), sm, color=col, lw=1.5, ls=ls,
                    label=stim_names[si])
        for ri, rv in enumerate(rev_indices):
            ax.axvline(rv, color='k', ls=':', lw=1.0,
                       label='Reversal' if ri == 0 else None)
            ax.text(rv, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1.0,
                    f'R{ri+1}', fontsize=6, ha='center', va='bottom', color='k')
        ax.set_xlim(left=0)
        ax.set_ylabel('Lick probability' if metric=='trial_lick_probs' else 'Value estimate')
        ax.legend(fontsize=7, ncol=4)
    axes[1].set_xlabel('Trial (per stimulus)')
    _ro_str = f"partialro{cfg['readout_size']}"
    _pra    = '' if cfg['use_prev_action_reward'] else ', no prev A/R'
    n_rev   = len(rev_indices)
    plt.suptitle(
        f'Run {run_idx+1}  seed={seed}  |  {freeze_label}  |  {_ro_str}{_pra}  '
        f'|  {n_rev} reversals\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)',
        fontsize=9
    )
    plt.tight_layout()
    fig.savefig(run_dir / "lick_value.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_grad_norms(metrics_numpy, run_dir, cfg, freeze_label, run_idx, seed, phase_boundaries):
    """Plot gradient norms over training timesteps (log-scale, smoothed)."""
    import pandas as pd
    gn = metrics_numpy['grad_norms']
    if not gn.get('timestep'):
        return
    ts   = np.array(gn['timestep'])
    smooth = 2000

    rev_ts = phase_boundaries.get('reversal_points', [])

    keys   = ['total', 'rnn', 'actor_fc', 'critic_fc']
    labels = ['Total', 'RNN', 'Actor FC', 'Critic FC']
    colors_gn = ['#333333', '#1f77b4', '#ff7f00', '#2ca02c']

    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.5 * len(keys)), sharex=True)
    for ax, k, lbl, col in zip(axes, keys, labels, colors_gn):
        vals = np.array(gn[k])
        if len(vals) == 0:
            continue
        sm = pd.Series(vals).rolling(smooth, min_periods=1).mean().values
        ax.semilogy(ts, vals, color=col, lw=0.4, alpha=0.2)
        ax.semilogy(ts, sm,   color=col, lw=1.5, label=lbl)
        for rt in rev_ts:
            ax.axvline(rt, color='k', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylabel(f'{lbl}\ngrad norm')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Timestep')
    _ro_str = f"partialro{cfg['readout_size']}"
    _pra    = '' if cfg['use_prev_action_reward'] else ', no prev A/R'
    plt.suptitle(
        f'Gradient norms — Run {run_idx+1} seed={seed}  |  {freeze_label}  |  {_ro_str}{_pra}',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(run_dir / 'grad_norms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_tdr_value_psth(results, run_dir, cfg, trial_structure, plot_stims, freeze_label):
    """Time-resolved TDR projections onto pre- and post-reversal value axes."""
    value_axes  = results['value_axes']
    tdr_va_psth = results['tdr_va_psth']

    # Reference trial length from first valid trial
    ref_info  = None
    for tb in results.get('trial_boundaries_ref', []):
        ref_info = trial_structure.get(tb['trial_idx'])
        if ref_info:
            break
    # fallback: just use timestep count from PSTH
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
            ax  = axes[row, 0]
            psth_data = tdr_va_psth.get(name)
            if psth_data is None:
                continue
            phase_data = psth_data[axis_key]
            for s in plot_stims:
                for p in [0, 1]:
                    trajs = phase_data.get((s, p), [])
                    if not trajs:
                        continue
                    min_len  = min(t.shape[0] for t in trajs)
                    stacked  = np.stack([t[:min_len] for t in trajs])
                    mean_tr  = stacked.mean(0)
                    sem_tr   = stacked.std(0) / np.sqrt(len(trajs))
                    col = PHASE_COLOR_MAP[p][s]
                    ls  = PHASE_LS_MAP[p]
                    ax.plot(np.arange(min_len), mean_tr, color=col, lw=1.5, ls=ls,
                            label=f"{STIM_NAMES_MAP[s]} {'pre' if p==0 else 'post'}")
                    ax.fill_between(np.arange(min_len),
                                    mean_tr - sem_tr, mean_tr + sem_tr,
                                    color=col, alpha=0.15)
            ax.axhline(0, color='grey', lw=0.6, ls=':')
            ax.set_title(f"{name}  |  angle={ax_info['angle_deg']:.1f}°", fontsize=9)
            ax.legend(fontsize=6, ncol=4)
            ax.set_ylabel('Proj (a.u.)')
        axes[-1, 0].set_xlabel('Timestep within trial')
        plt.suptitle(f'{axis_label}  [{freeze_label}]', fontsize=9)
        plt.tight_layout()
        fname = f"tdr_value_psth_{'pre' if axis_col==0 else 'post'}_axis.png"
        fig.savefig(run_dir / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_decoder(decoder_results, run_dir, freeze_label):
    """Bar plot of decoder cross-generalisation results."""
    pop_list   = list(decoder_results.keys())
    cond_keys  = ['pre_cv', 'post_cv', 'pre_to_post', 'post_to_pre']
    cond_cols  = ['#aec6e8', '#1f77b4', '#ffb347', '#d62728']
    cond_lbls  = ['Pre CV', 'Post CV', 'Pre→Post', 'Post→Pre']
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
    plt.suptitle(f'Value decoder  [{freeze_label}]', fontsize=9)
    plt.tight_layout()
    fig.savefig(run_dir / "value_decoder.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

# ── Aggregate lick/value plot ─────────────────────────────────────────────────

def plot_agg_lick_value(run_dirs, agg_dir, freeze_label, plot_stims):
    """Plot lick probability and value estimate for all runs on shared axes."""
    import pandas as pd
    smooth = 20
    stim_names = STIM_NAMES_MAP
    stim_idx   = {k: i for i, k in enumerate('ABCDEF')}

    # Load metrics from each run's pkl
    all_metrics = []
    for rd in run_dirs:
        pkl = rd / "run_results.pkl"
        # metrics_numpy is not in run_results.pkl; load separately if saved,
        # otherwise reconstruct from the pkl's stim_labels for reversal point
        # We save a lightweight lick/value pkl alongside run_results
        lv_pkl = rd / "lick_value_data.pkl"
        if lv_pkl.exists():
            with open(lv_pkl, 'rb') as f:
                all_metrics.append(pickle.load(f))

    if not all_metrics:
        _log("  WARNING: no lick_value_data.pkl files found — skipping agg lick/value plot")
        return

    stim_keys = [k for k in 'ABCDEF' if stim_idx[k] in plot_stims]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    for run_i, lv in enumerate(all_metrics):
        rev_idx   = lv['rev_idx']
        for k in stim_keys:
            si  = stim_idx[k]
            col = STIM_COLORS_DARK[si]
            for ax_i, metric in enumerate(['trial_lick_probs', 'trial_values']):
                y = lv[metric][k]
                if len(y) == 0:
                    continue
                sm = pd.Series(y).rolling(smooth, min_periods=1).mean().values
                axes[ax_i].plot(np.arange(len(sm)), sm, color=col, lw=0.7,
                                alpha=0.25, zorder=2)

    # Mean across runs per stimulus
    for k in stim_keys:
        si  = stim_idx[k]
        col = STIM_COLORS_DARK[si]
        ls  = '-' if si in (0, 4) else '--'  # solid = reversed, dashed = unchanged
        for ax_i, metric in enumerate(['trial_lick_probs', 'trial_values']):
            series_list = []
            for lv in all_metrics:
                y = lv[metric][k]
                if len(y) > 0:
                    series_list.append(pd.Series(y).rolling(smooth, min_periods=1).mean().values)
            if not series_list:
                continue
            min_len = min(len(s) for s in series_list)
            mat = np.stack([s[:min_len] for s in series_list])
            axes[ax_i].plot(np.arange(min_len), mat.mean(0), color=col, lw=2.0,
                            ls=ls, label=stim_names[si], zorder=4)

    # Reversal line (use median reversal index across runs)
    rev_indices = [lv['rev_idx'] for lv in all_metrics if lv['rev_idx'] is not None]
    if rev_indices:
        med_rev = int(np.median(rev_indices))
        for ax in axes:
            ax.axvline(med_rev, color='k', ls=':', lw=1.5, label='Reversal', zorder=5)

    axes[0].set_ylabel('Lick probability')
    axes[1].set_ylabel('Value estimate')
    axes[1].set_xlabel('Trial (per stimulus)')
    for ax in axes:
        ax.set_xlim(left=0)
        ax.legend(fontsize=7, ncol=4)
    plt.suptitle(
        f'Lick probability and value — {len(all_metrics)} runs  [{freeze_label}]\n'
        'Solid = reversed (A, E)  |  Dashed = unchanged (B, F)  |  '
        'Light traces = individual runs',
        fontsize=9
    )
    plt.tight_layout()
    fig.savefig(agg_dir / "agg_lick_value.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log(f"  Saved agg_lick_value.png", indent=1)


def plot_agg_grad_norms(run_dirs, agg_dir, freeze_label):
    """Aggregate gradient norms across runs: mean ± SEM per component."""
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

    smooth = 5000
    keys   = ['total', 'rnn', 'actor_fc', 'critic_fc']
    labels = ['Total', 'RNN', 'Actor FC', 'Critic FC']
    colors_gn = ['#333333', '#1f77b4', '#ff7f00', '#2ca02c']

    # Use shortest run as common x-axis
    min_len = min(len(gn['timestep']) for gn in all_gn)
    ts_ref  = np.array(all_gn[0]['timestep'][:min_len])

    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.5 * len(keys)), sharex=True)
    for ax, k, lbl, col in zip(axes, keys, labels, colors_gn):
        mats = []
        for gn in all_gn:
            vals = np.array(gn[k][:min_len], dtype=float)
            sm   = pd.Series(vals).rolling(smooth, min_periods=1).mean().values
            mats.append(sm)
        mat = np.stack(mats)
        mn  = mat.mean(0)
        se  = mat.std(0) / np.sqrt(len(mats))
        ax.semilogy(ts_ref, mn,          color=col, lw=1.5, label=lbl)
        ax.fill_between(ts_ref, np.maximum(mn - se, 1e-12), mn + se,
                        color=col, alpha=0.2)
        # Individual runs
        for row in mats:
            ax.semilogy(ts_ref, row, color=col, lw=0.5, alpha=0.15)
        ax.set_ylabel(f'{lbl}\ngrad norm')
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Timestep')
    plt.suptitle(
        f'Gradient norms — {len(all_gn)} runs [{freeze_label}]\n'
        'Mean ± SEM  |  light traces = individual runs',
        fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / 'agg_grad_norms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log(f"  Saved agg_grad_norms.png", indent=1)


# ── Aggregate plots ────────────────────────────────────────────────────────────

def aggregate_plots(run_dirs, agg_dir, cfg, freeze_label, plot_stims):
    """Load per-run results and make aggregate decoder + TDR plots."""
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

    pop_list = list(all_results[0]['decoder_results'].keys())

    # ── Aggregate decoder ─────────────────────────────────────────────────────
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
            ax.scatter(np.full(len(vals), xi) + jitter, vals,
                       s=12, color='k', alpha=0.4, zorder=3)
        ax.axhline(0.5, color='k', ls='--', lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(cond_lbls, rotation=30, ha='right', fontsize=7)
        ax.set_ylim(0.35, 1.05)
        ax.set_title(name, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Accuracy')
    plt.suptitle(f'Value decoder — {n_runs} runs  [{freeze_label}]', fontsize=9)
    plt.tight_layout()
    fig.savefig(agg_dir / "agg_value_decoder.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _log(f"  Saved agg_value_decoder.png", indent=1)

    # ── Aggregate TDR value PSTH ───────────────────────────────────────────────
    # For each population and axis (pre/post), average trial-mean trajectories across runs
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
            # For each (stim, phase), collect per-run mean trajectories
            combo_data = defaultdict(list)   # (s,p) -> list of 1D arrays (mean traj per run)
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
                        mean_tr = np.stack([t[:min_len] for t in trajs]).mean(0)
                        combo_data[(s, p)].append(mean_tr)
            for s in plot_stims:
                for p in [0, 1]:
                    run_trajs = combo_data.get((s, p), [])
                    if not run_trajs:
                        continue
                    min_len = min(t.shape[0] for t in run_trajs)
                    stacked  = np.stack([t[:min_len] for t in run_trajs])
                    gm = stacked.mean(0)
                    gs = stacked.std(0) / np.sqrt(len(run_trajs))
                    col = PHASE_COLOR_MAP[p][s]
                    ls  = PHASE_LS_MAP[p]
                    ax.plot(np.arange(min_len), gm, color=col, lw=1.5, ls=ls,
                            label=f"{STIM_NAMES_MAP[s]} {'pre' if p==0 else 'post'}")
                    ax.fill_between(np.arange(min_len), gm-gs, gm+gs, color=col, alpha=0.15)
            ax.axhline(0, color='grey', lw=0.6, ls=':')
            # Collect angles across runs
            angles = [r['value_axes'].get(name, {}).get('angle_deg', np.nan)
                      if r['value_axes'].get(name) else np.nan
                      for r in all_results]
            ang_str = f"{np.nanmean(angles):.1f}±{np.nanstd(angles):.1f}°" if any(~np.isnan(angles)) else '?'
            ax.set_title(f"{name}  |  angle={ang_str} (mean±std)", fontsize=9)
            ax.legend(fontsize=6, ncol=4)
            ax.set_ylabel('Proj (a.u.)')
        axes[-1, 0].set_xlabel('Timestep within trial')
        plt.suptitle(f'{axis_label} — {n_runs} runs  [{freeze_label}]', fontsize=9)
        plt.tight_layout()
        fname = f"agg_tdr_value_psth_{'pre' if axis_key=='pre' else 'post'}_axis.png"
        fig.savefig(agg_dir / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        _log(f"  Saved {fname}", indent=1)

    # ── Aggregate lick/value ───────────────────────────────────────────────────
    plot_agg_lick_value(run_dirs, agg_dir, freeze_label, plot_stims)
    plot_agg_grad_norms(run_dirs, agg_dir, freeze_label)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze_mode', default='none',
                        choices=['none','readout_only','rnn_only','all'])
    parser.add_argument('--use_prev_action_reward', type=int, default=0,
                        help='1=True, 0=False')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--readout_fraction', type=float, default=0.5,
                        help='Fraction of hidden units projecting to readout')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--policy_clip', type=float, default=0.25)
    parser.add_argument('--tdr_pre_start',  type=int, default=50)
    parser.add_argument('--tdr_pre_end',    type=int, default=-1,
                        help='-1 means None (use all remaining)')
    parser.add_argument('--tdr_post_start', type=int, default=50)
    parser.add_argument('--tdr_post_end',   type=int, default=-1)
    parser.add_argument('--exclude_mid_stim', type=int, default=1,
                        help='1=exclude C/D from plots')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run seeds that already have run_results.pkl')
    parser.add_argument('--results_dir', type=str,
                        default='../results/20_03_26_value_subspace_experiments')
    parser.add_argument('--task_data_dir', type=str,
                        default='../task_data')
    parser.add_argument('--task', type=str,
                        default='reversal_abcdef_multitimestep_partial')
    args = parser.parse_args()

    freeze_mode   = args.freeze_mode
    use_prev_ar   = bool(args.use_prev_action_reward)
    n_runs        = args.n_runs
    hidden_size   = args.hidden_size
    readout_size  = max(1, int(hidden_size * args.readout_fraction))

    tdr_pre_end  = None if args.tdr_pre_end  < 0 else args.tdr_pre_end
    tdr_post_end = None if args.tdr_post_end < 0 else args.tdr_post_end

    cfg = {
        'freeze_mode':           freeze_mode,
        'use_prev_action_reward': use_prev_ar,
        'hidden_size':           hidden_size,
        'readout_size':          readout_size,
        'learning_rate':         args.learning_rate,
        'gamma':                 args.gamma,
        'policy_clip':           args.policy_clip,
        'tdr_pre_block':         (args.tdr_pre_start,  tdr_pre_end),
        'tdr_post_block':        (args.tdr_post_start, tdr_post_end),
    }

    _freeze_labels = {
        'none': 'full plasticity', 'readout_only': 'readout only',
        'rnn_only': 'RNN only', 'all': 'frozen',
    }
    freeze_label = _freeze_labels[freeze_mode]
    _pra_str = '' if use_prev_ar else '_noprevAR'
    _ro_str  = f'partialro{readout_size}'
    variant_name = f'{args.task}_{_ro_str}_frozen{freeze_mode}{_pra_str}'

    results_dir  = Path(args.results_dir)
    variant_dir  = results_dir / variant_name
    agg_dir      = variant_dir / 'aggregate'
    variant_dir.mkdir(parents=True, exist_ok=True)

    plot_stims = [s for s in range(6) if not (args.exclude_mid_stim and s in (2, 3))]

    # ── Load task data once ────────────────────────────────────────────────────
    data_path = Path(args.task_data_dir) / f"{args.task}.pkl"
    if "multirev" in args.task:
        (state_sequence, reward_sequence, reversal_mask,
         phase_boundaries, trial_structure, state_map) = load_reversal_abcdef_multitimestep_multirev_data(
            data_path
        )
    else:
        (state_sequence, reward_sequence, reversal_mask,
         phase_boundaries, trial_structure, state_map) = load_reversal_abcdef_multitimestep_data(
            data_path
        )
    _log(f"Task loaded: {state_sequence.shape[0]} timesteps, "
         f"{len(trial_structure)} trials")

    # ── Save config CSV ────────────────────────────────────────────────────────
    with open(variant_dir / 'config.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'value'])
        for k, v in cfg.items():
            w.writerow([k, v])
        w.writerow(['n_runs', n_runs])
        w.writerow(['variant_name', variant_name])

    n_trainable = sum(p.numel() for p in build_model(
        10, 2, cfg).parameters() if p.requires_grad)   # quick probe for log
    post_end = phase_boundaries['post_reversal']['end']
    _log(f"Variant: {variant_name}")
    _log(f"  freeze={freeze_mode}  prevAR={use_prev_ar}  "
         f"hidden={hidden_size}  readout={readout_size}  lr={cfg['learning_rate']}")
    _log(f"  {n_runs} runs × {post_end:,} timesteps each")
    _log(f"  Results → {variant_dir}")

    run_dirs = []
    # ── Per-seed training & analysis ──────────────────────────────────────────
    for run_idx in range(n_runs):
        seed = run_idx * 7 + 42
        run_dir = variant_dir / f'run_{run_idx:02d}_seed{seed}'
        run_dir.mkdir(exist_ok=True)
        run_dirs.append(run_dir)

        _log("")
        _log(f"{'─'*56}")
        _log(f"Run {run_idx+1}/{n_runs}  seed={seed}  [{freeze_label}]")
        _log(f"{'─'*56}")

        # Skip if already complete (unless --overwrite)
        if not args.overwrite and (run_dir / 'run_results.pkl').exists():
            _log(f"  Skipping — run_results.pkl already exists (use --overwrite to re-run)",
                 indent=1)
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)

        _t_run = time.time()

        # Build fresh env + model each run
        env = ReversalABCDEFMultiTimestepEnv(
            state_sequence, reward_sequence, reversal_mask, trial_structure,
            reward_lick=1.0, lick_no_reward=-1.0, no_lick=0.0)
        state_size  = env.observation_space.shape[0]
        action_size = env.action_space.n
        model = build_model(state_size, action_size, cfg)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        _log(f"  Model: {n_train:,} / {n_total:,} params trainable", indent=1)

        # Train
        _log(f"  Training ({post_end:,} steps)...", indent=1)
        _t_train = time.time()
        metrics_numpy, agent = train(env, model, cfg, trial_structure, phase_boundaries, seed)
        n_trials = len(metrics_numpy['trial_boundaries'])
        pre_trials  = sum(1 for tb in metrics_numpy['trial_boundaries'] if tb['reversal_phase']==0)
        post_trials = n_trials - pre_trials
        _log(f"  Training done in {time.time()-_t_train:.0f}s  "
             f"({n_trials} trials: {pre_trials} pre, {post_trials} post)", indent=1)

        # Analyse
        _log(f"  Running analysis...", indent=1)
        results = analyse(metrics_numpy, model, cfg, trial_structure)

        # Per-run plots
        _log(f"  Saving per-run plots...", indent=1)
        plot_lick_value(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, freeze_label)
        _log(f"    lick_value.png", indent=2)
        plot_grad_norms(metrics_numpy, run_dir, cfg, freeze_label, run_idx, seed, phase_boundaries or {})
        _log(f"    grad_norms.png", indent=2)
        plot_tdr_value_psth(results, run_dir, cfg, trial_structure, plot_stims, freeze_label)
        _log(f"    tdr_value_psth_*.png", indent=2)
        plot_decoder(results['decoder_results'], run_dir, freeze_label)
        _log(f"    value_decoder.png", indent=2)

        # Save lightweight lick/value data for aggregate plot
        rev_trial_idx = sum(1 for tb in metrics_numpy['trial_boundaries']
                            if tb['stimulus'] == 0 and tb['reversal_phase'] == 0)
        lick_value_data = {
            'trial_lick_probs': {k: metrics_numpy['trial_lick_probs'][k] for k in 'ABCDEF'},
            'trial_values':     {k: metrics_numpy['trial_values'][k]     for k in 'ABCDEF'},
            'rev_idx': rev_trial_idx,
            'grad_norms': metrics_numpy.get('grad_norms', {}),
        }
        with open(run_dir / 'lick_value_data.pkl', 'wb') as f:
            pickle.dump(lick_value_data, f)

        # Save full metrics (incl. hidden states) for later PCA/TDR analysis
        with open(run_dir / "metrics_numpy.pkl", "wb") as f:
            pickle.dump(metrics_numpy, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save results pkl
        run_results = {
            'decoder_results': results['decoder_results'],
            'value_axes':      results['value_axes'],
            'tdr_projs':       results['tdr_projs'],
            'tdr_va_psth':     results['tdr_va_psth'],
            'all_stim_labels': results['all_stim_labels'],
            'all_phase_labels': results['all_phase_labels'],
            'pre_conv_mask':   results['pre_conv_mask'],
            'post_conv_mask':  results['post_conv_mask'],
            'params':          {**cfg, 'seed': seed},
        }
        with open(run_dir / 'run_results.pkl', 'wb') as f:
            pickle.dump(run_results, f)
        _log(f"  Run {run_idx+1} complete in {time.time()-_t_run:.0f}s  →  {run_dir.name}",
             indent=1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    _log("")
    _log(f"{'─'*56}")
    _log(f"Computing aggregate plots ({len(run_dirs)} runs)...")
    aggregate_plots(run_dirs, agg_dir, cfg, freeze_label, plot_stims)
    total = time.time() - _script_start
    _log(f"All done in {total/60:.1f} min.  Results in: {variant_dir}")


if __name__ == '__main__':
    main()

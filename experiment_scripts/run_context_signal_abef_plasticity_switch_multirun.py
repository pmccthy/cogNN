"""
Multi-run training + analysis script for Meta-AC with context-signal input on the
ABEF partial-reversal task, with a mid-run plasticity switch.

The model first trains (plasticity ON) for n_train_reversals reversals, then
continues running (plasticity OFF, weights frozen) for n_test_reversals more
reversals.  This isolates what the network has learned from online adaptation.

Usage:
    python run_context_signal_abef_plasticity_switch_multirun.py \\
        --n_runs 10 \\
        --n_train_reversals 10 \\
        --n_test_reversals 10 \\
        --trials_per_phase 400 \\
        --results_dir ../results/08_04_26_context_signal_abef_plasticity_switch
"""

import argparse
import pickle
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cog_nn.tasks.reversal_envs import (
    ReversalABCDEFMultiTimestepEnv,
    load_reversal_abcdef_multitimestep_multirev_data,
)
from cog_nn.agents import MetaA2CContextAgent
from cog_nn.analysis_utils import (
    _log, _script_start,
    N_CONTEXTS,
    ensure_task_data, build_model, build_ts_to_context,
    compute_reversal_global_indices, compute_reversal_trial_indices,
    global_trial_indices_per_stim,
    analyse,
    plot_lick_value, plot_grad_norms, plot_tdr_value_psth, plot_decoder,
    plot_reward_lick_count, plot_reward_consumed,
    plot_pca_hidden_states, plot_phase_generalisation_matrices,
    aggregate_plots,
)


# ── Training ───────────────────────────────────────────────────────────────────

def train(env, model, cfg, trial_structure, phase_boundaries, seed):
    hidden_size = cfg['hidden_size']
    action_size = env.action_space.n
    state_size  = env.observation_space.shape[0]
    clip        = cfg['policy_clip']

    stim_keys   = ['A', 'B', 'C', 'D', 'E', 'F']
    idx_to_stim = {i: k for i, k in enumerate(stim_keys)}

    metrics = {
        'lick_probs':  {k: [] for k in stim_keys},
        'values':      {k: [] for k in stim_keys},
        'timesteps':   {k: [] for k in stim_keys},
        'trial_lick_probs':         {k: [] for k in stim_keys},
        'trial_values':             {k: [] for k in stim_keys},
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
        'plasticity_off_trial_idx': None,
        'plasticity_off_ts':        None,
    }

    n_contexts_eff = cfg.get('n_contexts', N_CONTEXTS)
    agent = MetaA2CContextAgent(
        state_size=state_size, action_size=action_size,
        n_contexts=max(n_contexts_eff, 1), hidden_size=hidden_size,
        readout_indices=list(range(cfg['readout_size'])),
    )
    agent.model = model

    ts_to_ctx = build_ts_to_context(trial_structure)

    pre_start  = 0
    post_end   = phase_boundaries['post_reversal']['end']
    train_end  = int(phase_boundaries.get('train_end', post_end))

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
    plasticity_on   = True

    states_b, ctx_b, act_b, rew_b, ns_b, nctx_b, done_b = [[] for _ in range(7)]
    _rew_lick_buf     = {}
    _rew_consumed_buf = {}

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
                    'trial_idx':        trial_info,
                    'trial_start':      td['trial_start'],
                    'trial_end':        td['trial_end'],
                    'stimulus':         td['stimulus'],
                    'reversal_phase':   td['reversal_phase'],
                    'phase_idx':        td.get('phase_idx', td['reversal_phase']),
                    'reward_available': td['reward_available'],
                    'plasticity_on':    plasticity_on,
                })
                for key in ('within_trial_lick_probs', 'within_trial_values',
                            'within_trial_timesteps', 'within_trial_states',
                            'hidden_states'):
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
                _rew_lick_buf[trial_info]     = 0
                _rew_consumed_buf[trial_info] = 0.0
            if trial_info in _rew_lick_buf and t_idx in td['reward_window']:
                if action == 0:
                    _rew_lick_buf[trial_info] += 1
                _rew_consumed_buf[trial_info] += max(0.0, reward)
            if len(td['reward_window']) > 0 and t_idx == td['reward_window'][-1]:
                k = idx_to_stim.get(td['stimulus'])
                if k:
                    if trial_info in _rew_lick_buf:
                        metrics['trial_reward_lick_counts'][k].append(
                            _rew_lick_buf.pop(trial_info))
                    if trial_info in _rew_consumed_buf:
                        metrics['trial_reward_consumed'][k].append(
                            _rew_consumed_buf.pop(trial_info))
        if info.get('reward_available', False):
            metrics['rewards'].append(reward)
            metrics['reward_timesteps'].append(t_idx)

    LOG_EVERY = 10_000
    _win_rewards, _win_lick_p = [], []
    _last_log_t = pre_start

    for t_idx in range(pre_start, post_end):

        if plasticity_on and t_idx >= train_end:
            model.eval()
            plasticity_on = False
            n_trials_so_far = len(metrics['trial_boundaries'])
            metrics['plasticity_off_trial_idx'] = n_trials_so_far
            metrics['plasticity_off_ts']        = t_idx
            _log(f"  *** Plasticity OFF at t={t_idx}  "
                 f"(global trial {n_trials_so_far}) ***", indent=1)

        current_ctx_idx = ts_to_ctx.get(t_idx, current_ctx_idx)
        context = torch.zeros(n_contexts_eff)
        if n_contexts_eff > 0:
            context[current_ctx_idx] = 1.0

        state = torch.from_numpy(obs).float()
        action, action_prob, value, rnn_out = agent.select_action(
            state, context, deterministic=False, policy_clip=clip, return_rnn_out=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        _win_rewards.append(reward)
        _win_lick_p.append(float(action_prob if action == 0 else 1.0 - action_prob))

        record_step(t_idx, obs, action, action_prob, value, rnn_out, reward, info)

        next_ctx_idx = ts_to_ctx.get(t_idx + 1, current_ctx_idx)
        next_context = torch.zeros(n_contexts_eff)
        if n_contexts_eff > 0:
            next_context[next_ctx_idx] = 1.0

        if plasticity_on:
            states_b.append(obs); ctx_b.append(context.numpy())
            act_b.append(action); rew_b.append(reward)
            ns_b.append(next_obs); nctx_b.append(next_context.numpy())
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
            plast    = "ON " if plasticity_on else "OFF"
            rew_rate = np.mean(_win_rewards) if _win_rewards else 0.0
            lick_p   = np.mean(_win_lick_p)  if _win_lick_p  else 0.0
            n_trials = len(metrics['trial_boundaries'])
            _log(f"  step {t_idx:>7d}/{post_end}  ({pct:4.1f}%)  phase={phase}"
                 f"  plast={plast}  trials={n_trials:5d}"
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
        'plasticity_off_trial_idx': metrics['plasticity_off_trial_idx'],
        'plasticity_off_ts':        metrics['plasticity_off_ts'],
    }
    return metrics_numpy, agent


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_runs',              type=int,   default=10)
    parser.add_argument('--n_reversals',         type=int,   default=None,
                        help='Shorthand: sets both n_train_reversals and n_test_reversals.')
    parser.add_argument('--n_train_reversals',   type=int,   default=10)
    parser.add_argument('--n_test_reversals',    type=int,   default=10)
    parser.add_argument('--trials_per_phase',    type=int,   default=400)
    parser.add_argument('--task_seed',           type=int,   default=42)
    parser.add_argument('--hidden_size',         type=int,   default=128)
    parser.add_argument('--readout_fraction',    type=float, default=0.5)
    parser.add_argument('--learning_rate',       type=float, default=5e-4)
    parser.add_argument('--gamma',               type=float, default=0.0)
    parser.add_argument('--policy_clip',         type=float, default=0.25)
    parser.add_argument('--tdr_pre_start',       type=int,   default=50)
    parser.add_argument('--tdr_pre_end',         type=int,   default=-1)
    parser.add_argument('--tdr_post_start',      type=int,   default=50)
    parser.add_argument('--tdr_post_end',        type=int,   default=-1)
    parser.add_argument('--stim_window',         type=int,   default=None)
    parser.add_argument('--reward_window',       type=int,   default=None)
    parser.add_argument('--exclude_mid_stim',    type=int,   default=1)
    parser.add_argument('--include_mid_stim',    action='store_true', default=False)
    parser.add_argument('--task_pkl',            type=str,   default=None)
    parser.add_argument('--plot_stims',          type=str,   default=None)
    parser.add_argument('--no_context',          action='store_true', default=False)
    parser.add_argument('--overwrite',           action='store_true')
    parser.add_argument('--run_indices',         type=str,   default='')
    parser.add_argument('--results_dir',  type=str,
                        default='../results/08_04_26_context_signal_abef_plasticity_switch')
    parser.add_argument('--task_data_dir', type=str, default='../task_data')
    args = parser.parse_args()

    if args.n_reversals is not None:
        args.n_train_reversals = args.n_reversals
        args.n_test_reversals  = args.n_reversals

    n_runs         = args.n_runs
    hidden_size    = args.hidden_size
    readout_size   = max(1, int(hidden_size * args.readout_fraction))
    n_train_phases = args.n_train_reversals * 2
    n_test_phases  = args.n_test_reversals  * 2

    tdr_pre_end  = None if args.tdr_pre_end  < 0 else args.tdr_pre_end
    tdr_post_end = None if args.tdr_post_end < 0 else args.tdr_post_end

    n_contexts_eff = 0 if args.no_context else N_CONTEXTS

    cfg = {
        'freeze_mode':       'none',
        'hidden_size':       hidden_size,
        'readout_size':      readout_size,
        'n_contexts':        n_contexts_eff,
        'n_train_phases':    n_train_phases,
        'n_test_phases':     n_test_phases,
        'learning_rate':     args.learning_rate,
        'gamma':             args.gamma,
        'policy_clip':       args.policy_clip,
        'tdr_pre_block':     (args.tdr_pre_start,  tdr_pre_end),
        'tdr_post_block':    (args.tdr_post_start, tdr_post_end),
        'use_context_signal': not args.no_context,
        'plasticity_switch':  True,
    }

    _ro_str  = f'partialro{readout_size}'
    _win_str = (f'_sw{args.stim_window}_rw{args.reward_window}'
                if args.stim_window is not None or args.reward_window is not None else '')
    _ctx_str = '_nocontext' if args.no_context else '_context'
    variant_name = (f'abef_plswitch{_ctx_str}'
                    f'_train{args.n_train_reversals}rev'
                    f'_test{args.n_test_reversals}rev'
                    f'_tpp{args.trials_per_phase}{_win_str}_{_ro_str}')
    variant_label = variant_name

    results_dir = Path(args.results_dir)
    variant_dir = results_dir / variant_name
    agg_dir     = variant_dir / 'aggregate'
    variant_dir.mkdir(parents=True, exist_ok=True)

    if args.include_mid_stim:
        args.exclude_mid_stim = 0
    if args.plot_stims is not None:
        plot_stims = [int(s.strip()) for s in args.plot_stims.split(',')]
    else:
        plot_stims = [s for s in range(6) if not (args.exclude_mid_stim and s in (2, 3))]

    if args.task_pkl is not None:
        task_path = Path(args.task_pkl)
        _log(f"Using pre-built task pkl: {task_path}")
    else:
        task_path = ensure_task_data(
            args.task_data_dir, n_train_phases, n_test_phases,
            args.trials_per_phase, args.task_seed,
            stim_window=args.stim_window, reward_window=args.reward_window,
            include_mid_stim=args.include_mid_stim)

    (state_sequence, reward_sequence, reversal_mask,
     phase_boundaries, trial_structure, _) = load_reversal_abcdef_multitimestep_multirev_data(
        task_path)

    phase_boundaries = dict(phase_boundaries)
    phase_boundaries['plasticity_off_ts'] = phase_boundaries.get('train_end')
    _log(f"Task loaded: {state_sequence.shape[0]} timesteps, {len(trial_structure)} trials")
    _log(f"  Plasticity switch at timestep {phase_boundaries.get('train_end')}")

    with open(variant_dir / 'config.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'value'])
        for k, v in cfg.items():
            w.writerow([k, v])
        w.writerow(['n_runs', n_runs])
        w.writerow(['n_train_reversals', args.n_train_reversals])
        w.writerow(['n_test_reversals',  args.n_test_reversals])
        w.writerow(['trials_per_phase',  args.trials_per_phase])
        if args.stim_window   is not None: w.writerow(['stim_window',   args.stim_window])
        if args.reward_window is not None: w.writerow(['reward_window', args.reward_window])
        w.writerow(['variant_name', variant_name])
        w.writerow(['include_mid_stim', int(args.include_mid_stim)])
        w.writerow(['no_context', int(args.no_context)])

    post_end    = phase_boundaries['post_reversal']['end']
    n_trainable = sum(p.numel() for p in build_model(10, 2, cfg).parameters()
                      if p.requires_grad)
    _log(f"Variant: {variant_name}")
    _log(f"  hidden={hidden_size}  readout={readout_size}  "
         f"n_contexts={n_contexts_eff}  lr={cfg['learning_rate']}")
    _log(f"  train: {n_train_phases} phases ({args.n_train_reversals} reversals)")
    _log(f"  test:  {n_test_phases} phases ({args.n_test_reversals} reversals, plasticity OFF)")
    _log(f"  {n_runs} runs × {post_end:,} timesteps  |  {n_trainable:,} trainable params")
    _log(f"  Results → {variant_dir}")

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
        _log(f"Run {run_idx+1}/{n_runs}  seed={seed}")
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

        _log(f"  Training ({post_end:,} steps, plasticity OFF after "
             f"{phase_boundaries['train_end']:,})...", indent=1)
        _t_train = time.time()
        metrics_numpy, _ = train(env, model, cfg, trial_structure, phase_boundaries, seed)
        n_trials     = len(metrics_numpy['trial_boundaries'])
        train_trials = sum(1 for tb in metrics_numpy['trial_boundaries']
                           if tb.get('plasticity_on', True))
        test_trials  = n_trials - train_trials
        _log(f"  Done in {time.time()-_t_train:.0f}s  "
             f"({n_trials} trials: {train_trials} train, {test_trials} test/frozen)", indent=1)

        _log("  Running analysis...", indent=1)
        results = analyse(metrics_numpy, model, cfg, trial_structure)
        results['n_train_phases'] = n_train_phases

        _log("  Saving per-run plots...", indent=1)
        plot_lick_value(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed, variant_label)
        _log("    lick_value.png", indent=2)
        plot_grad_norms(metrics_numpy, run_dir, cfg, variant_label, run_idx, seed,
                        phase_boundaries)
        _log("    grad_norms.png", indent=2)
        plot_tdr_value_psth(results, run_dir, cfg, trial_structure, plot_stims, variant_label)
        _log("    tdr_value_psth_*.png", indent=2)
        plot_decoder(results['decoder_results'], run_dir, variant_label)
        _log("    value_decoder.png", indent=2)
        plot_reward_lick_count(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed,
                               variant_label,
                               plast_off_trial_idx=metrics_numpy.get('plasticity_off_trial_idx'))
        _log("    reward_lick_count.png", indent=2)
        plot_reward_consumed(metrics_numpy, run_dir, cfg, plot_stims, run_idx, seed,
                             variant_label,
                             plast_off_trial_idx=metrics_numpy.get('plasticity_off_trial_idx'))
        _log("    reward_consumed.png", indent=2)
        plot_pca_hidden_states(metrics_numpy, trial_structure, run_dir, cfg,
                               plot_stims, run_idx, seed, variant_label, n_pcs=5,
                               plast_off_trial_idx=metrics_numpy.get('plasticity_off_trial_idx'))
        _log("    pca_hidden_states.png", indent=2)

        mat_obj = plot_phase_generalisation_matrices(results, run_dir, variant_label)
        if mat_obj is not None:
            with open(run_dir / "decoder_phase_generalisation_matrices.pkl", "wb") as f:
                pickle.dump(mat_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            _log("    decoder_phase_generalisation_matrix_*.png", indent=2)

        mat_obj_po = plot_phase_generalisation_matrices(
            results, run_dir, variant_label, plast_off_only=True)
        if mat_obj_po is not None:
            with open(run_dir / "decoder_phase_gen_plast_off.pkl", "wb") as f:
                pickle.dump(mat_obj_po, f, protocol=pickle.HIGHEST_PROTOCOL)
            _log("    decoder_phase_generalisation_matrix_plast_off_*.png", indent=2)

        lick_value_data = {
            'trial_lick_probs':        {k: metrics_numpy['trial_lick_probs'][k]
                                        for k in 'ABCDEF'},
            'trial_values':            {k: metrics_numpy['trial_values'][k]
                                        for k in 'ABCDEF'},
            'trial_reward_lick_counts':{k: metrics_numpy['trial_reward_lick_counts'][k]
                                        for k in 'ABCDEF'},
            'trial_reward_consumed':   {k: metrics_numpy['trial_reward_consumed'][k]
                                        for k in 'ABCDEF'},
            'rev_indices':       compute_reversal_trial_indices(metrics_numpy['trial_boundaries']),
            'rev_indices_global':compute_reversal_global_indices(metrics_numpy['trial_boundaries']),
            'global_trial_indices_by_stim': global_trial_indices_per_stim(
                metrics_numpy['trial_boundaries']),
            'grad_norms':        metrics_numpy.get('grad_norms', {}),
            'plasticity_off_trial_idx': metrics_numpy.get('plasticity_off_trial_idx'),
            'plasticity_off_ts':        metrics_numpy.get('plasticity_off_ts'),
        }
        with open(run_dir / 'lick_value_data.pkl', 'wb') as f:
            pickle.dump(lick_value_data, f)

        with open(run_dir / "metrics_numpy.pkl", "wb") as f:
            pickle.dump(metrics_numpy, f, protocol=pickle.HIGHEST_PROTOCOL)

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
            'n_train_phases': n_train_phases,
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
        _log(f"  Run {run_idx+1} complete in {time.time()-_t_run:.0f}s  →  {run_dir.name}",
             indent=1)

    _log("")
    _log(f"{'─'*56}")
    _log(f"Computing aggregate plots ({len(run_dirs)} runs)...")
    aggregate_plots(run_dirs, agg_dir, cfg, variant_label, plot_stims, phase_boundaries)
    _log(f"All done in {(time.time()-_script_start)/60:.1f} min.  Results in: {variant_dir}")


if __name__ == '__main__':
    main()

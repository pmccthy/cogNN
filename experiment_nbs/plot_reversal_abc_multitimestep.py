"""
Plotting functions for Reversal ABC Multi-Timestep task results.

Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))
import cog_nn.plot_style


def plot_trial_structure_results(metrics_path, trial_structure_path, save_dir=None):
    """
    Plot results for multi-timestep trial structure.
    
    Creates:
    1. Standard plots (lick probs, values over time)
    2. Trial-level plots (last timestep of stim window)
    3. Within-trial dynamics plots (early/mid/late learning, pre/post reversal)
    """
    # Load metrics
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    # Load trial structure
    with open(trial_structure_path, 'rb') as f:
        data = pickle.load(f)
        trial_structure = data.get('trial_structure', [])
        phase_boundaries = data.get('phase_boundaries', {})
    
    reversal_points = phase_boundaries.get('reversal_points', [])
    pre_end = phase_boundaries.get('pre_reversal', {}).get('end', len(metrics['trial_timesteps']))
    
    # Plot 1: Standard time series plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Lick probabilities over time
    ax = axes[0, 0]
    if len(metrics['timesteps_A']) > 0:
        ax.plot(metrics['timesteps_A'], metrics['lick_probs']['A'], 
               label='Stimulus A', linewidth=1.5, color='darkblue', alpha=0.7)
    if len(metrics['timesteps_B']) > 0:
        ax.plot(metrics['timesteps_B'], metrics['lick_probs']['B'], 
               label='Stimulus B', linewidth=1.5, color='darkred', alpha=0.7)
    if len(metrics['timesteps_C']) > 0:
        ax.plot(metrics['timesteps_C'], metrics['lick_probs']['C'], 
               label='Stimulus C', linewidth=1.5, color='darkgreen', alpha=0.7)
    
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Lick Probability')
    ax.set_title('Lick Probability Through Training')
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Value estimates over time
    ax = axes[0, 1]
    if len(metrics['timesteps_A']) > 0:
        ax.plot(metrics['timesteps_A'], metrics['values']['A'], 
               label='Stimulus A', linewidth=1.5, color='darkblue', alpha=0.7)
    if len(metrics['timesteps_B']) > 0:
        ax.plot(metrics['timesteps_B'], metrics['values']['B'], 
               label='Stimulus B', linewidth=1.5, color='darkred', alpha=0.7)
    if len(metrics['timesteps_C']) > 0:
        ax.plot(metrics['timesteps_C'], metrics['values']['C'], 
               label='Stimulus C', linewidth=1.5, color='darkgreen', alpha=0.7)
    
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value Estimate')
    ax.set_title('Value Estimates Through Training')
    ax.legend()
    
    # Trial-level lick probabilities (last timestep of stim window)
    ax = axes[1, 0]
    trial_timesteps = metrics['trial_timesteps']
    reversal_phases = metrics['trial_reversal_phases']
    
    pre_mask = np.array(reversal_phases) == 0
    post_mask = np.array(reversal_phases) == 1
    
    if np.any(pre_mask):
        pre_timesteps = np.array(trial_timesteps)[pre_mask]
        if len(metrics['trial_lick_probs']['A']) > 0:
            pre_A = np.array(metrics['trial_lick_probs']['A'])[pre_mask]
            ax.plot(pre_timesteps, pre_A, 'o', label='A (pre)', color='darkblue', alpha=0.5, markersize=2)
        if len(metrics['trial_lick_probs']['B']) > 0:
            pre_B = np.array(metrics['trial_lick_probs']['B'])[pre_mask]
            ax.plot(pre_timesteps, pre_B, 'o', label='B (pre)', color='darkred', alpha=0.5, markersize=2)
        if len(metrics['trial_lick_probs']['C']) > 0:
            pre_C = np.array(metrics['trial_lick_probs']['C'])[pre_mask]
            ax.plot(pre_timesteps, pre_C, 'o', label='C (pre)', color='darkgreen', alpha=0.5, markersize=2)
    
    if np.any(post_mask):
        post_timesteps = np.array(trial_timesteps)[post_mask]
        if len(metrics['trial_lick_probs']['A']) > 0:
            post_A = np.array(metrics['trial_lick_probs']['A'])[post_mask]
            ax.plot(post_timesteps, post_A, 's', label='A (post)', color='lightblue', alpha=0.5, markersize=2)
        if len(metrics['trial_lick_probs']['B']) > 0:
            post_B = np.array(metrics['trial_lick_probs']['B'])[post_mask]
            ax.plot(post_timesteps, post_B, 's', label='B (post)', color='lightcoral', alpha=0.5, markersize=2)
        if len(metrics['trial_lick_probs']['C']) > 0:
            post_C = np.array(metrics['trial_lick_probs']['C'])[post_mask]
            ax.plot(post_timesteps, post_C, 's', label='C (post)', color='lightgreen', alpha=0.5, markersize=2)
    
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Lick Probability (End of Stim Window)')
    ax.set_title('Trial-Level Lick Probabilities')
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Rewards over time
    ax = axes[1, 1]
    if len(metrics['reward_timesteps']) > 0:
        window_size = 200
        if len(metrics['rewards']) > window_size:
            smoothed_rewards = np.convolve(metrics['rewards'], 
                                          np.ones(window_size)/window_size, mode='valid')
            start_idx = window_size // 2
            smoothed_timesteps = metrics['reward_timesteps'][start_idx:start_idx + len(smoothed_rewards)]
            ax.plot(smoothed_timesteps, smoothed_rewards, linewidth=1.5, color='purple')
        else:
            ax.plot(metrics['reward_timesteps'], metrics['rewards'], linewidth=1, color='purple')
    
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title('Reward Consumption Through Training')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "standard_plots.png", dpi=300, bbox_inches='tight')
    
    # Plot 2: Within-trial dynamics
    plot_within_trial_dynamics(metrics, trial_structure, phase_boundaries, save_dir)
    
    return fig


def plot_within_trial_dynamics(metrics, trial_structure, phase_boundaries, save_dir=None):
    """
    Plot within-trial dynamics for early/mid/late learning, pre/post reversal.
    """
    reversal_points = phase_boundaries.get('reversal_points', [])
    pre_end = phase_boundaries.get('pre_reversal', {}).get('end', len(metrics['trial_timesteps']))
    
    # Define learning phases
    num_pre_trials = np.sum(np.array(metrics['trial_reversal_phases']) == 0)
    num_post_trials = np.sum(np.array(metrics['trial_reversal_phases']) == 1)
    
    # Split into early/mid/late
    pre_early_end = num_pre_trials // 3
    pre_mid_end = 2 * num_pre_trials // 3
    
    post_early_end = num_pre_trials + num_post_trials // 3
    post_mid_end = num_pre_trials + 2 * num_post_trials // 3
    
    # Get trial structure parameters
    stim_window_len = trial_structure[0]['stim_window'][-1] - trial_structure[0]['stim_window'][0] + 1
    reward_window_len = trial_structure[0]['reward_window'][-1] - trial_structure[0]['reward_window'][0] + 1
    
    # Create figure with subplots for each learning phase and reversal phase
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    learning_phases = [
        ('Early', 0, pre_early_end),
        ('Mid', pre_early_end, pre_mid_end),
        ('Late', pre_mid_end, num_pre_trials)
    ]
    
    reversal_phases = [
        ('Pre-reversal', 0),
        ('Post-reversal', 1)
    ]
    
    for learn_idx, (learn_name, learn_start, learn_end) in enumerate(learning_phases):
        for rev_idx, (rev_name, rev_phase) in enumerate(reversal_phases):
            ax = axes[learn_idx, rev_idx]
            
            # Get trials in this phase
            trial_indices = metrics['trial_indices']
            reversal_phases_arr = np.array(metrics['trial_reversal_phases'])
            
            # Filter by reversal phase
            rev_mask = reversal_phases_arr == rev_phase
            
            # Adjust trial indices for post-reversal
            if rev_phase == 1:
                trial_offset = num_pre_trials
                phase_start = learn_start + trial_offset
                phase_end = learn_end + trial_offset
            else:
                phase_start = learn_start
                phase_end = learn_end
            
            # Filter by learning phase
            phase_mask = (np.array(trial_indices) >= phase_start) & (np.array(trial_indices) < phase_end)
            combined_mask = rev_mask & phase_mask
            
            if not np.any(combined_mask):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{learn_name} {rev_name}')
                continue
            
            # Get trial data for these trials
            selected_trials = np.array(trial_indices)[combined_mask]
            
            # For each selected trial, get within-trial data
            # This would require storing more detailed metrics during training
            # For now, we'll plot what we have: last timestep of stim window
            
            # Plot lick probabilities for each stimulus
            for stim_idx, (stim_name, stim_color) in enumerate([('A', 'darkblue'), ('B', 'darkred'), ('C', 'darkgreen')]):
                stim_trials = []
                stim_probs = []
                
                for trial_idx in selected_trials:
                    trial_data = trial_structure[trial_idx]
                    if trial_data['stimulus'] == stim_idx:
                        # Get the last timestep of stim window
                        last_stim_timestep = trial_data['stim_window'][-1]
                        # Find corresponding metric
                        # This is simplified - in practice, you'd need to track all timesteps
                        stim_trials.append(trial_idx)
                        # Use trial-level metrics
                        trial_metric_idx = np.where(np.array(trial_indices) == trial_idx)[0]
                        if len(trial_metric_idx) > 0:
                            prob = metrics['trial_lick_probs'][stim_name][trial_metric_idx[0]]
                            stim_probs.append(prob)
                
                if len(stim_probs) > 0:
                    ax.plot(stim_trials, stim_probs, 'o', label=stim_name, 
                           color=stim_color, alpha=0.5, markersize=3)
            
            ax.set_xlabel('Trial Index')
            ax.set_ylabel('Lick Probability')
            ax.set_title(f'{learn_name} {rev_name}')
            ax.legend()
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "within_trial_dynamics.png", dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    metrics_path = Path("/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/reversal_abc_multitimestep/metrics.pkl")
    trial_structure_path = Path("/Users/pmccarthy/Documents/cogNN/task_data/reversal_abc_multitimestep.pkl")
    save_dir = Path("/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/reversal_abc_multitimestep")
    
    plot_trial_structure_results(metrics_path, trial_structure_path, save_dir)
    plt.show()



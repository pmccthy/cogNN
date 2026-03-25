"""
Analyse grid search results and create visualizations.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import cog_nn.plot_style

# Configuration
# results_dir = Path("/Users/pmccarthy/Documents/modelling_results/january_2026_metarl/meta_ac_reversal_singletimestep_abc_gridsearch")
results_dir = Path("/Users/pmccarthy/Documents/modelling_results/january_2026_meta_ac/meta_ac_reversal_abc_multitimestep_gridsearch")
vis_dir = results_dir / "vis"
vis_dir.mkdir(exist_ok=True)

# Check if this is a multitimestep task by looking at the first run's metrics
is_multitimestep = False
task_data_path = Path("/Users/pmccarthy/Documents/cogNN/task_data")

# Load grid search configuration
config_file = results_dir / "gridsearch_config.json"
with open(config_file, "r") as f:
    config = json.load(f)

reversal_points = config["task_config"]["reversal_points"]
state_sequence_length = config["task_config"]["post_end"]
pre_end = config["task_config"]["pre_end"]
post_end = config["task_config"]["post_end"]

print(f"Loaded grid search configuration")
print(f"Total combinations: {config['num_combinations']}")
print(f"Runs per combination: {config.get('num_runs_per_combination', 1)}")
print(f"Reversal points: {reversal_points}")

# Load summary
summary_file = results_dir / "gridsearch_summary.json"
with open(summary_file, "r") as f:
    summary = json.load(f)

print(f"\nFound {len(summary)} completed runs")

# Group runs by parameter combination
# If combo_id exists, use it; otherwise group by parameter values
runs_by_combo = defaultdict(list)
for run_info in summary:
    # Try to get combo_id, fallback to grouping by params
    if "combo_id" in run_info:
        combo_id = run_info["combo_id"]
    else:
        # For backward compatibility: group by parameter values
        params_str = "_".join([f"{k}_{v}" for k, v in sorted(run_info["params"].items())])
        combo_id = hash(params_str) % 10000  # Simple hash for grouping
    
    runs_by_combo[combo_id].append(run_info)

print(f"Found {len(runs_by_combo)} unique parameter combinations")

# Check if this is a multitimestep task
if len(summary) > 0:
    first_run_metrics_file = Path(summary[0]["metrics_file"])
    with open(first_run_metrics_file, "rb") as f:
        first_metrics = pickle.load(f)
    is_multitimestep = 'trial_boundaries' in first_metrics or 'trial_lick_probs' in first_metrics
    
    if is_multitimestep:
        print("Detected multitimestep task - will create learning stages plot")
        # Load trial structure if available
        task_name = config["task_config"]["task"]
        if task_name == "reversal_abc_multitimestep":
            try:
                from cog_nn.tasks.reversal_envs import load_reversal_abc_multitimestep_data
                data_path = task_data_path / f"{task_name}.pkl"
                _, _, _, _, trial_structure, state_map = load_reversal_abc_multitimestep_data(data_path)
                print(f"Loaded trial structure with {len(trial_structure)} trials")
            except Exception as e:
                print(f"Warning: Could not load trial structure: {e}")
                trial_structure = None
                state_map = None
        else:
            trial_structure = None
            state_map = None
    else:
        trial_structure = None
        state_map = None
else:
    trial_structure = None
    state_map = None

# Function to create plot for a parameter combination (with all runs)
def plot_combination_results(runs_list, combo_id, save_path):
    """Create visualization plot for a parameter combination with all runs overlaid."""
    if len(runs_list) == 0:
        print(f"No runs found for combination {combo_id}")
        return
    
    # Get parameters from first run (all should be the same)
    params = runs_list[0]["params"]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Create parameter string for suptitle
    param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())])
    param_str = param_str.replace(".", "_")  # Replace dots with underscores for display
    fig.suptitle(param_str, fontsize=12, y=0.995)
    
    # Plot 1: Lick probabilities over training (all runs)
    ax = axes[0, 0]
    for run_info in runs_list:
        metrics_file = Path(run_info["metrics_file"])
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        
        # For multitimestep tasks, use trial_lick_probs; otherwise use lick_probs
        if 'trial_lick_probs' in metrics and is_multitimestep and trial_structure is not None:
            lick_probs_A = metrics['trial_lick_probs']['A']
            lick_probs_B = metrics['trial_lick_probs']['B']
            lick_probs_C = metrics['trial_lick_probs']['C']
            # Get trial timesteps filtered by stimulus
            all_trial_indices = metrics.get('trial_indices', [])
            all_trial_timesteps = metrics.get('trial_timesteps', [])
            # Filter timesteps for each stimulus
            mask_A = np.array([(idx < len(trial_structure) and trial_structure[idx]['stimulus'] == 0) 
                              for idx in all_trial_indices], dtype=bool)
            mask_B = np.array([(idx < len(trial_structure) and trial_structure[idx]['stimulus'] == 1) 
                              for idx in all_trial_indices], dtype=bool)
            mask_C = np.array([(idx < len(trial_structure) and trial_structure[idx]['stimulus'] == 2) 
                              for idx in all_trial_indices], dtype=bool)
            timesteps_A = all_trial_timesteps[mask_A] if len(mask_A) > 0 else np.array([])
            timesteps_B = all_trial_timesteps[mask_B] if len(mask_B) > 0 else np.array([])
            timesteps_C = all_trial_timesteps[mask_C] if len(mask_C) > 0 else np.array([])
        else:
            lick_probs_A = metrics['lick_probs']['A']
            lick_probs_B = metrics['lick_probs']['B']
            lick_probs_C = metrics['lick_probs']['C']
            timesteps_A = metrics.get('timesteps_A', np.arange(len(lick_probs_A)))
            timesteps_B = metrics.get('timesteps_B', np.arange(len(lick_probs_B)))
            timesteps_C = metrics.get('timesteps_C', np.arange(len(lick_probs_C)))
        
        alpha = 0.3 if len(runs_list) > 1 else 0.7
        
        if len(timesteps_A) == len(lick_probs_A) and len(timesteps_A) > 0:
            ax.plot(timesteps_A, lick_probs_A, label='Stimulus A' if run_info == runs_list[0] else '', 
                   linewidth=1.5, color='darkblue', alpha=alpha)
        if len(timesteps_B) == len(lick_probs_B) and len(timesteps_B) > 0:
            ax.plot(timesteps_B, lick_probs_B, label='Stimulus B' if run_info == runs_list[0] else '', 
                   linewidth=1.5, color='darkred', alpha=alpha)
        if len(timesteps_C) == len(lick_probs_C) and len(timesteps_C) > 0:
            ax.plot(timesteps_C, lick_probs_C, label='Stimulus C' if run_info == runs_list[0] else '', 
                   linewidth=1.5, color='darkgreen', alpha=alpha)
    
    # Mark reversal points
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                   label='Reversal' if rev_point == reversal_points[0] else '')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Lick Probability')
    # Check if first run has trial_lick_probs
    first_metrics_file = Path(runs_list[0]["metrics_file"])
    with open(first_metrics_file, "rb") as f:
        first_metrics = pickle.load(f)
    title = 'Trial-Level Lick Probabilities' if (is_multitimestep and 'trial_lick_probs' in first_metrics) else 'Lick Probability Through Training'
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.set_xlim(0, state_sequence_length)
    
    # Plot 2: Value estimates over training (all runs)
    ax = axes[0, 1]
    for run_info in runs_list:
        metrics_file = Path(run_info["metrics_file"])
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        
        # For multitimestep tasks, use trial_values; otherwise use values
        if 'trial_values' in metrics and is_multitimestep and trial_structure is not None:
            values_A = metrics['trial_values']['A']
            values_B = metrics['trial_values']['B']
            values_C = metrics['trial_values']['C']
            # Get trial timesteps filtered by stimulus
            all_trial_indices = metrics.get('trial_indices', [])
            all_trial_timesteps = metrics.get('trial_timesteps', [])
            # Filter timesteps for each stimulus
            mask_A = np.array([(idx < len(trial_structure) and trial_structure[idx]['stimulus'] == 0) 
                              for idx in all_trial_indices], dtype=bool)
            mask_B = np.array([(idx < len(trial_structure) and trial_structure[idx]['stimulus'] == 1) 
                              for idx in all_trial_indices], dtype=bool)
            mask_C = np.array([(idx < len(trial_structure) and trial_structure[idx]['stimulus'] == 2) 
                              for idx in all_trial_indices], dtype=bool)
            timesteps_A = all_trial_timesteps[mask_A] if len(mask_A) > 0 else np.array([])
            timesteps_B = all_trial_timesteps[mask_B] if len(mask_B) > 0 else np.array([])
            timesteps_C = all_trial_timesteps[mask_C] if len(mask_C) > 0 else np.array([])
        else:
            values_A = metrics['values']['A']
            values_B = metrics['values']['B']
            values_C = metrics['values']['C']
            timesteps_A = metrics.get('timesteps_A', np.arange(len(values_A)))
            timesteps_B = metrics.get('timesteps_B', np.arange(len(values_B)))
            timesteps_C = metrics.get('timesteps_C', np.arange(len(values_C)))
        
        alpha = 0.3 if len(runs_list) > 1 else 0.7
        
        if len(timesteps_A) == len(values_A) and len(timesteps_A) > 0:
            ax.plot(timesteps_A, values_A, label='Stimulus A' if run_info == runs_list[0] else '', 
                   linewidth=1.5, color='darkblue', alpha=alpha)
        if len(timesteps_B) == len(values_B) and len(timesteps_B) > 0:
            ax.plot(timesteps_B, values_B, label='Stimulus B' if run_info == runs_list[0] else '', 
                   linewidth=1.5, color='darkred', alpha=alpha)
        if len(timesteps_C) == len(values_C) and len(timesteps_C) > 0:
            ax.plot(timesteps_C, values_C, label='Stimulus C' if run_info == runs_list[0] else '', 
                   linewidth=1.5, color='darkgreen', alpha=alpha)
    
    # Mark reversal points
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                   label='Reversal' if rev_point == reversal_points[0] else '')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value Estimate')
    # Check if first run has trial_values
    first_metrics_file = Path(runs_list[0]["metrics_file"])
    with open(first_metrics_file, "rb") as f:
        first_metrics = pickle.load(f)
    title = 'Trial-Level Value Estimates' if (is_multitimestep and 'trial_values' in first_metrics) else 'Value Estimates Through Training'
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, state_sequence_length)
    
    # Plot 3: Rewards over time (smoothed, all runs)
    ax = axes[1, 0]
    window_size = 200
    for run_info in runs_list:
        metrics_file = Path(run_info["metrics_file"])
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        
        rewards = metrics['rewards']
        reward_timesteps = metrics.get('reward_timesteps', np.arange(len(rewards)))
        
        alpha = 0.3 if len(runs_list) > 1 else 0.7
        
        if len(rewards) > window_size:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            start_idx = window_size // 2
            smoothed_timesteps = reward_timesteps[start_idx:start_idx + len(smoothed_rewards)]
            ax.plot(smoothed_timesteps, smoothed_rewards, linewidth=1.5, color='purple', alpha=alpha)
        else:
            ax.plot(reward_timesteps, rewards, linewidth=1, color='purple', alpha=alpha)
    
    # Mark reversal points
    for rev_point in reversal_points:
        ax.axvline(x=rev_point, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                   label='Reversal' if rev_point == reversal_points[0] else '')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward Amount (smoothed)')
    ax.set_title('Consumption of Available Reward Through Training')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, state_sequence_length)
    
    # Plot 4: Lick probabilities at end of each phase (all runs)
    ax = axes[1, 1]
    
    def get_value_at_timestep(timesteps, values, target_timestep):
        """Get the last value before or at target_timestep."""
        mask = timesteps <= target_timestep
        if np.any(mask):
            idx = np.where(mask)[0][-1]  # Last index before or at target
            return values[idx]
        return None
    
    # Collect data from all runs
    all_pre_end_vals = {'A': [], 'B': [], 'C': []}
    all_post_end_vals = {'A': [], 'B': [], 'C': []}
    
    for run_info in runs_list:
        metrics_file = Path(run_info["metrics_file"])
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        
        lick_probs_A = metrics['lick_probs']['A']
        lick_probs_B = metrics['lick_probs']['B']
        lick_probs_C = metrics['lick_probs']['C']
        timesteps_A = metrics.get('timesteps_A', np.arange(len(lick_probs_A)))
        timesteps_B = metrics.get('timesteps_B', np.arange(len(lick_probs_B)))
        timesteps_C = metrics.get('timesteps_C', np.arange(len(lick_probs_C)))
        
        if len(timesteps_A) > 0 and len(lick_probs_A) > 0:
            val = get_value_at_timestep(timesteps_A, lick_probs_A, pre_end)
            if val is not None:
                all_pre_end_vals['A'].append(val)
            val = get_value_at_timestep(timesteps_A, lick_probs_A, post_end)
            if val is not None:
                all_post_end_vals['A'].append(val)
        
        if len(timesteps_B) > 0 and len(lick_probs_B) > 0:
            val = get_value_at_timestep(timesteps_B, lick_probs_B, pre_end)
            if val is not None:
                all_pre_end_vals['B'].append(val)
            val = get_value_at_timestep(timesteps_B, lick_probs_B, post_end)
            if val is not None:
                all_post_end_vals['B'].append(val)
        
        if len(timesteps_C) > 0 and len(lick_probs_C) > 0:
            val = get_value_at_timestep(timesteps_C, lick_probs_C, pre_end)
            if val is not None:
                all_pre_end_vals['C'].append(val)
            val = get_value_at_timestep(timesteps_C, lick_probs_C, post_end)
            if val is not None:
                all_post_end_vals['C'].append(val)
    
    # Prepare data for grouped bar chart with error bars
    stimuli = []
    pre_means = []
    pre_stds = []
    post_means = []
    post_stds = []
    colors_map = {'A': 'darkblue', 'B': 'darkred', 'C': 'darkgreen'}
    colors = []
    
    for stim in ['A', 'B', 'C']:
        if len(all_pre_end_vals[stim]) > 0 or len(all_post_end_vals[stim]) > 0:
            stimuli.append(stim)
            pre_means.append(np.mean(all_pre_end_vals[stim]) if len(all_pre_end_vals[stim]) > 0 else 0)
            pre_stds.append(np.std(all_pre_end_vals[stim]) if len(all_pre_end_vals[stim]) > 1 else 0)
            post_means.append(np.mean(all_post_end_vals[stim]) if len(all_post_end_vals[stim]) > 0 else 0)
            post_stds.append(np.std(all_post_end_vals[stim]) if len(all_post_end_vals[stim]) > 1 else 0)
            colors.append(colors_map[stim])
    
    if len(stimuli) > 0:
        x = np.arange(len(stimuli))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-reversal', color=colors, alpha=0.7, yerr=pre_stds, capsize=5)
        bars2 = ax.bar(x + width/2, post_means, width, label='Post-reversal', color=colors, alpha=0.5, yerr=post_stds, capsize=5)
        
        ax.set_ylabel('Lick Probability')
        ax.set_title('Lick Probabilities at End of Each Phase')
        ax.set_xticks(x)
        ax.set_xticklabels(stimuli)
        ax.legend()
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {save_path}")

# Function to create learning stages plot for multitimestep tasks
def plot_learning_stages(runs_list, combo_id, save_path, trial_structure=None):
    """Create learning stages plot showing within-trial dynamics across training stages."""
    if len(runs_list) == 0:
        print(f"No runs found for combination {combo_id}")
        return
    
    if trial_structure is None:
        print(f"Warning: No trial structure available for learning stages plot")
        return
    
    # Configuration for trial extraction
    num_trials_per_stage = 5
    early_window = (0.0, 0.2)
    mid_window = (0.4, 0.6)
    late_window = (0.8, 1.0)
    
    # Get parameters from first run
    params = runs_list[0]["params"]
    
    # Load metrics from first run (we'll average across runs)
    metrics_file = Path(runs_list[0]["metrics_file"])
    with open(metrics_file, "rb") as f:
        metrics = pickle.load(f)
    
    if 'trial_boundaries' not in metrics:
        print(f"Warning: No trial_boundaries in metrics for learning stages plot")
        return
    
    # Get trial boundaries separated by phase
    pre_trials = [tb for tb in metrics['trial_boundaries'] if tb['reversal_phase'] == 0]
    post_trials = [tb for tb in metrics['trial_boundaries'] if tb['reversal_phase'] == 1]
    
    def select_trials_by_position(trial_list, position_range, stimulus, num_trials):
        """Select trials from a specific position range and stimulus."""
        start_idx = int(len(trial_list) * position_range[0])
        end_idx = int(len(trial_list) * position_range[1])
        candidate_trials = trial_list[start_idx:end_idx]
        
        # Filter by stimulus (stimulus is stored as integer: 0=A, 1=B, 2=C)
        stimulus_trials = [t for t in candidate_trials if t['stimulus'] == stimulus]
        
        # Select evenly spaced trials
        if len(stimulus_trials) == 0:
            return []
        if len(stimulus_trials) <= num_trials:
            return [t['trial_idx'] for t in stimulus_trials]
        
        indices = np.linspace(0, len(stimulus_trials) - 1, num_trials, dtype=int)
        return [stimulus_trials[i]['trial_idx'] for i in indices]
    
    # Select trials for each stage and stimulus
    selected_trials = {}
    for phase_name, position_range in [('early', early_window), ('mid', mid_window), ('late', late_window)]:
        selected_trials[phase_name] = {}
        for phase_type in ['pre', 'post']:
            selected_trials[phase_name][phase_type] = {}
            trial_list = pre_trials if phase_type == 'pre' else post_trials
            
            for stim_int, stim_name in [(0, 'A'), (1, 'B'), (2, 'C')]:
                trial_indices = select_trials_by_position(trial_list, position_range, stim_int, num_trials_per_stage)
                selected_trials[phase_name][phase_type][stim_name] = trial_indices
    
    # Create grid plot: 4 rows (policy pre, value pre, policy post, value post) × 3 columns (early, mid, late)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    # Color map for stimuli
    stim_colors = {'A': 'darkblue', 'B': 'darkred', 'C': 'darkgreen'}
    stage_labels = ['Early', 'Mid', 'Late']
    
    # Plot each combination
    for row_idx, (row_label, phase_type, metric_type) in enumerate([
        ('Policy Pre-reversal', 'pre', 'lick_probs'),
        ('Value Pre-reversal', 'pre', 'values'),
        ('Policy Post-reversal', 'post', 'lick_probs'),
        ('Value Post-reversal', 'post', 'values')
    ]):
        for col_idx, (stage_name, stage_range) in enumerate([
            ('early', early_window),
            ('mid', mid_window),
            ('late', late_window)
        ]):
            ax = axes[row_idx, col_idx]
            
            # Get phase boundaries from first trial
            first_trial_idx = None
            for stim_name in ['A', 'B', 'C']:
                if selected_trials[stage_name][phase_type][stim_name]:
                    first_trial_idx = selected_trials[stage_name][phase_type][stim_name][0]
                    break
            
            # Calculate max ITI end for this subplot
            subplot_max_iti_end = 0
            for stim_name in ['A', 'B', 'C']:
                for trial_idx in selected_trials[stage_name][phase_type][stim_name]:
                    if trial_idx < len(trial_structure):
                        trial_info = trial_structure[trial_idx]
                        if 'iti_window' in trial_info and trial_info['iti_window'] and len(trial_info['iti_window']) > 0:
                            trial_start = trial_info['trial_start']
                            iti_end_rel = trial_info['iti_window'][-1] - trial_start
                            subplot_max_iti_end = max(subplot_max_iti_end, iti_end_rel)
                    break
            
            # Initialize variables for phase boundaries
            stim_start_rel = 0
            stim_end_rel = 0
            reward_start_rel = 0
            reward_end_rel = 0
            
            if first_trial_idx is not None and first_trial_idx < len(trial_structure):
                first_trial_info = trial_structure[first_trial_idx]
                trial_start = first_trial_info['trial_start']
                stim_start_rel = first_trial_info['stim_window'][0] - trial_start
                stim_end_rel = first_trial_info['stim_window'][-1] - trial_start
                reward_start_rel = first_trial_info['reward_window'][0] - trial_start
                reward_end_rel = first_trial_info['reward_window'][-1] - trial_start
                
                # Shade phase regions
                if row_idx == 0 and col_idx == 0:
                    ax.axvspan(stim_start_rel, stim_end_rel, alpha=0.1, color='blue', label='Stim Window')
                    ax.axvspan(reward_start_rel-1, reward_end_rel, alpha=0.1, color='orange', label='Reward Window')
                else:
                    ax.axvspan(stim_start_rel, stim_end_rel, alpha=0.1, color='blue')
                    ax.axvspan(reward_start_rel-1, reward_end_rel, alpha=0.1, color='orange')
                
                # Apply ITI shading
                if subplot_max_iti_end > 0:
                    iti_start = reward_end_rel + 1
                    if row_idx == 0 and col_idx == 0:
                        ax.axvspan(iti_start, subplot_max_iti_end, alpha=0.1, color='gray', label='ITI')
                    else:
                        ax.axvspan(iti_start, subplot_max_iti_end, alpha=0.1, color='gray')
            
            # Collect data for mean calculation
            stim_data_for_mean = {'A': [], 'B': [], 'C': []}
            plotted_stimuli = set()
            
            # Plot trials for each stimulus (average across runs)
            for stim_name in ['A', 'B', 'C']:
                trial_indices = selected_trials[stage_name][phase_type][stim_name]
                color = stim_colors[stim_name]
                
                for trial_idx in trial_indices:
                    if trial_idx < len(trial_structure):
                        # Collect data from all runs for this trial
                        all_trial_data = []
                        all_trial_timesteps = []
                        
                        for run_info in runs_list:
                            run_metrics_file = Path(run_info["metrics_file"])
                            with open(run_metrics_file, "rb") as f:
                                run_metrics = pickle.load(f)
                            
                            if trial_idx in run_metrics[f'within_trial_{metric_type}']:
                                trial_data = run_metrics[f'within_trial_{metric_type}'][trial_idx]
                                trial_timesteps = run_metrics['within_trial_timesteps'][trial_idx]
                                trial_info = trial_structure[trial_idx]
                                
                                # Normalize timesteps to start at 0
                                trial_start = trial_info['trial_start']
                                trial_timesteps_norm = np.array(trial_timesteps) - trial_start
                                
                                all_trial_data.append((trial_timesteps_norm, trial_data))
                                if len(all_trial_timesteps) == 0:
                                    all_trial_timesteps = trial_timesteps_norm
                        
                        # Average across runs
                        if len(all_trial_data) > 0:
                            # Find common timestep range
                            all_timesteps = []
                            for timesteps, _ in all_trial_data:
                                all_timesteps.extend(timesteps.tolist())
                            
                            if len(all_timesteps) > 0:
                                # Create common timestep grid
                                common_timesteps = np.linspace(0, max(all_timesteps), 100)
                                interpolated_data = []
                                
                                for timesteps, data in all_trial_data:
                                    if len(timesteps) > 1 and len(data) == len(timesteps):
                                        try:
                                            interp_data = np.interp(common_timesteps, timesteps, data)
                                            interpolated_data.append(interp_data)
                                        except:
                                            pass
                                
                                if len(interpolated_data) > 0:
                                    # Calculate mean across interpolated traces
                                    interpolated_array = np.array(interpolated_data)
                                    mean_data = np.nanmean(interpolated_array, axis=0)
                                    
                                    # Plot mean trace
                                    label = None
                                    if row_idx == 0 and col_idx == 0 and stim_name not in plotted_stimuli:
                                        label = f'Stimulus {stim_name}'
                                        plotted_stimuli.add(stim_name)
                                    
                                    ax.plot(common_timesteps, mean_data, '-', 
                                           color=color, alpha=0.6, linewidth=2, label=label)
                                    
                                    # Collect for overall mean
                                    stim_data_for_mean[stim_name].append((common_timesteps, mean_data))
            
            # Plot overall mean traces for each stimulus
            for stim_name in ['A', 'B', 'C']:
                if len(stim_data_for_mean[stim_name]) > 0:
                    # Find common timestep range
                    all_timesteps = []
                    for timesteps, _ in stim_data_for_mean[stim_name]:
                        all_timesteps.extend(timesteps.tolist())
                    
                    if len(all_timesteps) > 0:
                        common_timesteps = np.linspace(0, max(all_timesteps), 100)
                        interpolated_data = []
                        
                        for timesteps, data in stim_data_for_mean[stim_name]:
                            if len(timesteps) > 1 and len(data) == len(timesteps):
                                try:
                                    interp_data = np.interp(common_timesteps, timesteps, data)
                                    interpolated_data.append(interp_data)
                                except:
                                    pass
                        
                        if len(interpolated_data) > 0:
                            interpolated_array = np.array(interpolated_data)
                            mean_data = np.nanmean(interpolated_array, axis=0)
                            
                            color = stim_colors[stim_name]
                            label = f'Mean {stim_name}' if row_idx == 0 and col_idx == 0 else None
                            ax.plot(common_timesteps, mean_data, '-', color=color, alpha=1.0, 
                                   linewidth=3, label=label, zorder=10)
            
            # Set labels and limits
            ax.set_xlabel('Timestep Within Trial')
            ax.set_xlim(left=0)
            
            # Set title
            if row_idx == 0:
                ax.set_title(f'{stage_labels[col_idx]} Learning - Pre-reversal')
            elif row_idx == 1:
                ax.set_title(f'{stage_labels[col_idx]} Learning - Pre-reversal')
            elif row_idx == 2:
                ax.set_title(f'{stage_labels[col_idx]} Learning - Post-reversal')
            else:
                ax.set_title(f'{stage_labels[col_idx]} Learning - Post-reversal')
            
            if col_idx == 0:
                if metric_type == 'lick_probs':
                    ax.set_ylabel('Lick Probability')
                    ax.set_ylim([0, 1])
                else:
                    ax.set_ylabel('Value Estimate')
    
    # Add parameter string for suptitle
    param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())])
    param_str = param_str.replace(".", "_")
    fig.suptitle(param_str, fontsize=12, y=0.995)
    
    # Collect legend handles and labels from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Add global legend
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                   frameon=True, fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved learning stages plot to {save_path}")

# Process each parameter combination
print(f"\nGenerating plots...")
for combo_id, runs_list in sorted(runs_by_combo.items()):
    # Get parameters from first run
    params = runs_list[0]["params"]
    
    # Create filename for plot
    param_str = "_".join([f"{k}_{v}" for k, v in sorted(params.items())])
    param_str = param_str.replace(".", "_")  # Replace dots with underscores for filenames
    
    if len(runs_list) > 1:
        plot_filename = f"combo_{combo_id:04d}_{param_str}_all_runs.png"
    else:
        plot_filename = f"combo_{combo_id:04d}_{param_str}.png"
    plot_path = vis_dir / plot_filename
    
    # Create standard plot
    try:
        plot_combination_results(runs_list, combo_id, plot_path)
    except Exception as e:
        print(f"Error creating plot for combination {combo_id}: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # Create learning stages plot for multitimestep tasks
    if is_multitimestep and trial_structure is not None:
        learning_stages_filename = f"combo_{combo_id:04d}_{param_str}_learning_stages.png"
        learning_stages_path = vis_dir / learning_stages_filename
        
        try:
            plot_learning_stages(runs_list, combo_id, learning_stages_path, trial_structure)
        except Exception as e:
            print(f"Error creating learning stages plot for combination {combo_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

print(f"\n{'='*60}")
print(f"Analysis complete!")
print(f"Plots saved to: {vis_dir}")
print(f"{'='*60}")

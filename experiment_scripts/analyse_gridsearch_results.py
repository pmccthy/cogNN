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
results_dir = Path("/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/ff_a2c_reversal_abc_gridsearch")
vis_dir = results_dir / "vis"
vis_dir.mkdir(exist_ok=True)

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
    ax.set_title('Lick Probability Through Training')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, state_sequence_length)
    
    # Plot 2: Value estimates over training (all runs)
    ax = axes[0, 1]
    for run_info in runs_list:
        metrics_file = Path(run_info["metrics_file"])
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        
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
    ax.set_title('Value Estimates Through Training')
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
    
    # Create plot
    try:
        plot_combination_results(runs_list, combo_id, plot_path)
    except Exception as e:
        print(f"Error creating plot for combination {combo_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(f"Analysis complete!")
print(f"Plots saved to: {vis_dir}")
print(f"{'='*60}")

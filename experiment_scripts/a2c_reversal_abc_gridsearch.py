"""
Perform grid search over parameters for A2C agent in Reversal ABC environment.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import sys
from pathlib import Path
import pickle
import torch
from torch.optim import Adam
import numpy as np
from itertools import product
import json
from datetime import datetime
import random
from pprint import pprint

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cog_nn.tasks.reversal_envs import ReversalABCEnv, load_reversal_abc_data
from cog_nn.models import Actor, Critic
from cog_nn.agents import A2CAgent

# All parameters - if a list has only one element, it's treated as fixed (not searched)
# If a list has multiple elements, all combinations will be searched
all_params = {
    "learning_rate": [0.0005, 0.001],
    "gamma": [0, 0.5],
    "model_size": [4, 8, 16],
    "batch_size": [1],  # Fixed - only one value
    "reward_lick": [1.0],  # Reward when reward is available and agent licks
    "lick_no_reward": [0., -0.5, -1.0],  # Reward when no reward available but agent licks
    "no_lick": [0.0],  # Reward when agent doesn't lick
    "policy_clip": [None, 0.25],  # Policy clipping parameter for select_action
}

# Number of runs per parameter combination
num_runs_per_combination = 5

# Paths and task configuration
task_data_path = Path("/Users/pmccarthy/Documents/cogNN/task_data")
task = "reversal_abc"
results_dir = Path("/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/ff_a2c_reversal_abc_gridsearch")

# Create results directory
results_dir.mkdir(parents=True, exist_ok=True)

# Load task data
print("Loading task data...")
data_path = Path(task_data_path, f"{task}.pkl")
state_sequence, reward_sequence, reversal_mask, phase_boundaries, state_map = load_reversal_abc_data(data_path)
print(f"Loaded data from {data_path}")
print(f"State sequence shape: {state_sequence.shape}")
print(f"Reward sequence shape: {reward_sequence.shape}")
if phase_boundaries:
    print(f"Phase boundaries: {phase_boundaries}")
    print(f"Reversal points (timesteps): {phase_boundaries.get('reversal_points', [])}")

original_state_sequence = state_sequence.copy()
original_reward_sequence = reward_sequence.copy()
original_reversal_mask = reversal_mask.copy() if reversal_mask is not None else None

# Get state and action sizes
state_size = state_sequence.shape[1]
action_size = 2

# Get phase boundaries
pre_start = phase_boundaries['pre_reversal']['start']
pre_end = phase_boundaries['pre_reversal']['end']
post_start = phase_boundaries['post_reversal']['start']
post_end = phase_boundaries['post_reversal']['end']
reversal_points = phase_boundaries.get('reversal_points', [])

print(f"\nTraining setup:")
print(f"Pre-reversal phase: timesteps {pre_start} to {pre_end}")
print(f"Post-reversal phase: timesteps {post_start} to {post_end}")
print(f"Reversal points: {reversal_points}")

# Separate grid search params (multiple values) from fixed params (single value)
grid_params = {k: v for k, v in all_params.items() if len(v) > 1}
fixed_params = {k: v[0] for k, v in all_params.items() if len(v) == 1}

print(f"\nAll parameters: {all_params}")
print(f"Grid search parameters (multiple values): {grid_params}")
print(f"Fixed parameters (single value): {fixed_params}")

# Generate all parameter combinations for grid search params
if grid_params:
    grid_param_names = list(grid_params.keys())
    grid_param_values = list(grid_params.values())
    grid_combinations = list(product(*grid_param_values))
    
    # Create full parameter combinations by adding fixed params to each grid combination
    param_combinations = []
    for grid_combo in grid_combinations:
        full_combo = dict(zip(grid_param_names, grid_combo))
        full_combo.update(fixed_params)  # Add fixed params
        param_combinations.append(full_combo)
else:
    # No grid search, just use fixed params
    param_combinations = [fixed_params.copy()]

print(f"\nTotal parameter combinations: {len(param_combinations)}")

# Save grid search configuration
config = {
    "all_params": all_params,
    "grid_params": grid_params,
    "fixed_params": fixed_params,
    "task_config": {
        "task": task,
        "state_size": state_size,
        "action_size": action_size,
        "pre_start": pre_start,
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_end,
        "reversal_points": reversal_points,
    },
    "num_combinations": len(param_combinations),
    "num_runs_per_combination": num_runs_per_combination,
    "timestamp": datetime.now().isoformat(),
}

with open(results_dir / "gridsearch_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Training function
def train_model(params_dict, run_id, total_runs):
    """Train a single model with given parameters."""
    print(f"\n{'='*60}")
    print(f"Training model {run_id}/{total_runs}")
    print("Parameters:")
    pprint(params_dict)
    print(f"{'='*60}")
    
    # Extract parameters
    learning_rate = params_dict["learning_rate"]
    gamma = params_dict["gamma"]
    model_size = params_dict["model_size"]
    batch_size = params_dict.get("batch_size", 1)
    reward_lick = params_dict.get("reward_lick", 1.0)
    lick_no_reward = params_dict.get("lick_no_reward", -2.0)
    no_lick = params_dict.get("no_lick", 0.0)
    policy_clip = params_dict.get("policy_clip", None)
    
    # Create Actor and Critic networks
    actor = Actor(state_size=state_size,
                  action_size=action_size,
                  hidden_size=model_size)
    
    critic = Critic(state_size=state_size,
                    hidden_size=model_size)
    
    # Create optimizers
    actor_optimizer = Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = Adam(critic.parameters(), lr=learning_rate)
    
    # Create A2C agent
    agent = A2CAgent(actor, critic, actor_optimizer, critic_optimizer, 
                     gamma=gamma, device='cpu')
    
    # Set networks to training mode
    actor.train()
    critic.train()
    
    if not np.array_equal(state_sequence, original_state_sequence):
        print(f"WARNING: state_sequence has been modified!")
    if not np.array_equal(reward_sequence, original_reward_sequence):
        print(f"WARNING: reward_sequence has been modified!")
    if reversal_mask is not None and original_reversal_mask is not None:
        if not np.array_equal(reversal_mask, original_reversal_mask):
            print(f"WARNING: reversal_mask has been modified!")
    
    # Reset environment
    env = ReversalABCEnv(
        state_sequence, 
        reward_sequence, 
        reversal_mask,
        reward_lick=reward_lick,
        lick_no_reward=lick_no_reward,
        no_lick=no_lick,
        state_map=state_map
    )
    obs, info = env.reset()
    
    
    # Track metrics during training
    metrics = {
        'lick_probs': {'A': [], 'B': [], 'C': []},
        'values': {'A': [], 'B': [], 'C': []},
        'rewards': [],
        'reward_timesteps': [],
        'timesteps_A': [],
        'timesteps_B': [],
        'timesteps_C': []
    }
    
    # Track states, actions, rewards for batch update
    states_batch = []
    actions_batch = []
    rewards_batch = []
    next_states_batch = []
    dones_batch = []
    
    # Train on pre-reversal phase
    print(f"Phase 1: Pre-reversal (timesteps {pre_start} to {pre_end})")
    
    pre_reversal_step_count = 0
    for t_idx in range(pre_start, pre_end):
        if t_idx % 1000 == 0:
            print(f"  Timestep {t_idx}/{pre_end}")
        
        # Get current state
        state = torch.from_numpy(obs).float()
        
        # Check if we're at a stimulus state BEFORE stepping
        state_idx_before_step = np.argmax(obs)
        is_stimulus_state = state_idx_before_step in [0, 1, 2]  # A, B, or C
        
        # Select action
        action, action_prob, value = agent.select_action(state, deterministic=False, policy_clip=policy_clip)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Get actual environment timestep
        env_timestep = info.get('timestep', t_idx)
        pre_reversal_step_count += 1
        
        # Store for batch update
        states_batch.append(obs)
        actions_batch.append(action)
        rewards_batch.append(reward)
        next_states_batch.append(next_obs)
        dones_batch.append(done)
        
        # Track metrics by stimulus (use state BEFORE step)
        state_idx = state_idx_before_step
        if state_idx == 0:  # A
            if action == 0:  # lick
                metrics['lick_probs']['A'].append(action_prob)
            elif action == 1:  # no-lick
                metrics['lick_probs']['A'].append(1 - action_prob)
            metrics['values']['A'].append(value)
            metrics['timesteps_A'].append(env_timestep)
        elif state_idx == 1:  # B
            if action == 0:  # lick
                metrics['lick_probs']['B'].append(action_prob)
            elif action == 1:  # no-lick
                metrics['lick_probs']['B'].append(1 - action_prob)
            metrics['values']['B'].append(value)
            metrics['timesteps_B'].append(env_timestep)
        elif state_idx == 2:  # C
            if action == 0:  # lick
                metrics['lick_probs']['C'].append(action_prob)
            elif action == 1:  # no-lick
                metrics['lick_probs']['C'].append(1 - action_prob)
            metrics['values']['C'].append(value)
            metrics['timesteps_C'].append(env_timestep)
        
        # Track rewards only when reward is available
        # Note: reward_available in info indicates if reward was available at the stimulus state we just left
        # But it's only True when we step FROM a stimulus state, not FROM an outcome state
        # So we need to check it only when we were at a stimulus state before stepping
        reward_available = info.get('reward_available', False) if is_stimulus_state else False
        if reward_available:
            metrics['rewards'].append(reward)
            metrics['reward_timesteps'].append(env_timestep)
        
        obs = next_obs
        
        # Batch update
        if len(states_batch) >= batch_size:
            agent.update(
                np.array(states_batch),
                np.array(actions_batch),
                np.array(rewards_batch),
                np.array(next_states_batch),
                np.array(dones_batch)
            )
            # Clear batch
            states_batch = []
            actions_batch = []
            rewards_batch = []
            next_states_batch = []
            dones_batch = []
    
    # Train on post-reversal phase
    print(f"Phase 2: Post-reversal (timesteps {post_start} to {post_end})")
    
    post_reversal_step_count = 0
    for t_idx in range(post_start, post_end):
        if (t_idx - post_start) % 1000 == 0:
            print(f"  Timestep {t_idx}/{post_end}")
        
        # Get current state
        state = torch.from_numpy(obs).float()
        
        # Check if we're at a stimulus state BEFORE stepping
        state_idx_before_step = np.argmax(obs)
        is_stimulus_state = state_idx_before_step in [0, 1, 2]  # A, B, or C
        
        # Select action
        action, action_prob, value = agent.select_action(state, deterministic=False, policy_clip=policy_clip)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Get actual environment timestep
        env_timestep = info.get('timestep', t_idx)
        post_reversal_step_count += 1
        
        # Store for batch update
        states_batch.append(obs)
        actions_batch.append(action)
        rewards_batch.append(reward)
        next_states_batch.append(next_obs)
        dones_batch.append(done)
        
        # Track metrics by stimulus (use state BEFORE step)
        state_idx = state_idx_before_step
        if state_idx == 0:  # A
            if action == 0:  # lick
                metrics['lick_probs']['A'].append(action_prob)
            elif action == 1:  # no-lick
                metrics['lick_probs']['A'].append(1 - action_prob)
            metrics['values']['A'].append(value)
            metrics['timesteps_A'].append(env_timestep)
        elif state_idx == 1:  # B
            if action == 0:  # lick
                metrics['lick_probs']['B'].append(action_prob)
            elif action == 1:  # no-lick
                metrics['lick_probs']['B'].append(1 - action_prob)
            metrics['values']['B'].append(value)
            metrics['timesteps_B'].append(env_timestep)
        elif state_idx == 2:  # C
            if action == 0:  # lick
                metrics['lick_probs']['C'].append(action_prob)
            elif action == 1:  # no-lick
                metrics['lick_probs']['C'].append(1 - action_prob)
            metrics['values']['C'].append(value)
            metrics['timesteps_C'].append(env_timestep)
        
        # Track rewards only when reward is available
        # Note: reward_available in info indicates if reward was available at the stimulus state we just left
        # But it's only True when we step FROM a stimulus state, not FROM an outcome state
        # So we need to check it only when we were at a stimulus state before stepping
        reward_available = info.get('reward_available', False) if is_stimulus_state else False
        if reward_available:
            metrics['rewards'].append(reward)
            metrics['reward_timesteps'].append(env_timestep)
        
        obs = next_obs
        
        # Batch update
        if len(states_batch) >= batch_size:
            agent.update(
                np.array(states_batch),
                np.array(actions_batch),
                np.array(rewards_batch),
                np.array(next_states_batch),
                np.array(dones_batch)
            )
            # Clear batch
            states_batch = []
            actions_batch = []
            rewards_batch = []
            next_states_batch = []
            dones_batch = []
    
    # Final update with remaining batch
    if len(states_batch) > 0:
        agent.update(
            np.array(states_batch),
            np.array(actions_batch),
            np.array(rewards_batch),
            np.array(next_states_batch),
            np.array(dones_batch)
        )
    
    print(f"Training complete!")
    print(f"Total rewards collected: {len(metrics['rewards'])}")
    
    # Convert metrics to numpy arrays for saving
    metrics_numpy = {
        'lick_probs': {
            'A': np.array(metrics['lick_probs']['A']),
            'B': np.array(metrics['lick_probs']['B']),
            'C': np.array(metrics['lick_probs']['C'])
        },
        'values': {
            'A': np.array(metrics['values']['A']),
            'B': np.array(metrics['values']['B']),
            'C': np.array(metrics['values']['C'])
        },
        'rewards': np.array(metrics['rewards']),
        'reward_timesteps': np.array(metrics['reward_timesteps']),
        'timesteps_A': np.array(metrics['timesteps_A']),
        'timesteps_B': np.array(metrics['timesteps_B']),
        'timesteps_C': np.array(metrics['timesteps_C']),
    }
    
    return metrics_numpy, params_dict, actor, critic

# Run grid search
print(f"\n{'='*60}")
print("Starting grid search...")
print(f"{'='*60}")

# Calculate total number of runs
total_runs = len(param_combinations) * num_runs_per_combination
print(f"Total runs to execute: {total_runs} ({len(param_combinations)} combinations × {num_runs_per_combination} runs each)")

results = []
total_run_id = 0
for combo_id, params_dict in enumerate(param_combinations, 1):
    print(f"\n{'='*60}")
    print(f"Parameter combination {combo_id}/{len(param_combinations)}")
    print("Parameters:")
    pprint(params_dict)
    print(f"Running {num_runs_per_combination} run(s) for this combination")
    print(f"{'='*60}")
    
    # Run multiple times for this parameter combination
    for run_idx in range(num_runs_per_combination):
        total_run_id += 1
        
        # Reset random seed for each run to get different initializations
        # Use combination ID and run index to create unique seed
        run_seed = RANDOM_SEED + combo_id * 1000 + run_idx
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)
        
        # Create unique filename for this run
        param_str = "_".join([f"{k}_{v}" for k, v in sorted(params_dict.items())])
        param_str = param_str.replace(".", "_")  # Replace dots with underscores for filenames
        
        if num_runs_per_combination > 1:
            run_dir = results_dir / f"combo_{combo_id:04d}_run_{run_idx+1:02d}_{param_str}"
        else:
            run_dir = results_dir / f"run_{total_run_id:04d}_{param_str}"
        run_dir.mkdir(exist_ok=True)
        
        try:
            # Train model
            print(f"\n  Run {run_idx+1}/{num_runs_per_combination} (total run {total_run_id}/{total_runs}, seed={run_seed})")
            metrics, params, actor, critic = train_model(params_dict, total_run_id, total_runs)
            
            # Save metrics
            metrics_file = run_dir / "metrics.pkl"
            with open(metrics_file, "wb") as f:
                pickle.dump(metrics, f)
            
            # Save parameters
            params_file = run_dir / "params.json"
            with open(params_file, "w") as f:
                json.dump(params, f, indent=2)
            
            # Save model state dicts
            torch.save(actor.state_dict(), run_dir / "actor.pth")
            torch.save(critic.state_dict(), run_dir / "critic.pth")
            
            results.append({
                "combo_id": combo_id,
                "run_idx": run_idx,
                "total_run_id": total_run_id,
                "params": params,
                "metrics_file": str(metrics_file),
                "params_file": str(params_file),
                "run_dir": str(run_dir),
            })
            
            print(f"  Saved results to {run_dir}")
            
        except Exception as e:
            print(f"  Error training run {run_idx+1} for combination {combo_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

# Save summary of all runs
summary_file = results_dir / "gridsearch_summary.json"
with open(summary_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Grid search complete!")
print(f"Total parameter combinations: {len(param_combinations)}")
print(f"Runs per combination: {num_runs_per_combination}")
print(f"Total runs completed: {len(results)}/{len(param_combinations) * num_runs_per_combination}")
print(f"Results saved to: {results_dir}")
print(f"Summary saved to: {summary_file}")
print(f"{'='*60}")

"""
Gymnasium environments for reversal learning tasks (single-timestep versions).

This module contains:
- ReversalABEnv: Reversal learning with A and B stimuli (uses pre-generated sequence)
- ReversalABCEnv: Reversal learning with A, B, and C stimuli (uses pre-generated sequence)
- ReversalABCDynamicEnv: Reversal ABC environment that generates trials dynamically
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from pathlib import Path


class ReversalABEnv(gym.Env):
    """
    Reversal AB task environment matching train_a2c_rl.ipynb structure.
    
    Uses a pre-generated sequence of states. When reward is available:
    - If agent licks: gets reward and follows sequence
    - If agent doesn't lick: goes to unrewarded state
    
    States (one-hot encoded):
    - A: Stimulus A [1, 0, 0, 0]
    - B: Stimulus B [0, 1, 0, 0]
    - unrewarded: No reward state [0, 0, 1, 0]
    - rewarded: Reward state [0, 0, 0, 1]
    
    Actions:
    - 0: lick
    - 1: no_lick
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, state_sequence=None, reward_sequence=None, reversal_mask=None, 
                 reward_lick=1.0, lick_no_reward=-1.0, no_lick=0.0, render_mode=None,
                 state_map=None):
        """
        Initialize the environment.
        
        Args:
            state_sequence: Pre-generated sequence of states (N x 4 array).
                          If None, generates a simple sequence.
            reward_sequence: Pre-generated sequence of rewards (N x 1 array).
                           If None, generates based on state sequence.
            reversal_mask: Array indicating reversal phase (0=pre, 1=post) for each trial.
                          If None, creates default mask.
            reward_lick: Reward when reward is available and agent licks (default: 1.0)
            lick_no_reward: Reward when no reward available but agent licks (default: -1.0)
            no_lick: Reward when agent doesn't lick (default: 0.0)
            render_mode: Rendering mode
            state_map: Dict mapping state names to numpy arrays. If None, uses default encodings.
        """
        super().__init__()
        
        # Store reward values
        self.reward_lick = reward_lick
        self.lick_no_reward = lick_no_reward
        self.no_lick = no_lick
        
        # State space: 4D one-hot encoding
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
        # Action space: lick (0) or no_lick (1)
        self.action_space = spaces.Discrete(2)
        
        # Default state encodings (one-hot) - matching train_a2c_rl.ipynb
        default_state_encodings = {
            'A': np.array([1, 0, 0, 0], dtype=np.float32),
            'B': np.array([0, 1, 0, 0], dtype=np.float32),
            'unrewarded': np.array([0, 0, 1, 0], dtype=np.float32),
            'rewarded': np.array([0, 0, 0, 1], dtype=np.float32),
        }
        
        # Use state_map if provided, otherwise use defaults
        if state_map is not None:
            # Convert state_map values to numpy arrays if needed
            self.state_encodings = {}
            for key, value in state_map.items():
                if isinstance(value, (list, tuple)):
                    self.state_encodings[key] = np.array(value, dtype=np.float32)
                elif isinstance(value, np.ndarray):
                    self.state_encodings[key] = value.astype(np.float32)
                else:
                    self.state_encodings[key] = np.array(value, dtype=np.float32)
            # Merge with defaults for any missing keys
            for key, value in default_state_encodings.items():
                if key not in self.state_encodings:
                    self.state_encodings[key] = value
        else:
            self.state_encodings = default_state_encodings
        
        # Identify stimulus and outcome state names from state_encodings
        self.stimulus_states = ['A', 'B']  # For AB env
        self.outcome_states = ['rewarded', 'unrewarded']
        
        # Store sequences
        if state_sequence is None:
            # Generate a simple default sequence for testing
            self.state_sequence = self._generate_default_sequence()
        else:
            self.state_sequence = np.array(state_sequence, dtype=np.float32)
        
        if reward_sequence is None:
            # Generate reward sequence based on states
            self.reward_sequence = self._generate_reward_sequence()
        else:
            self.reward_sequence = np.array(reward_sequence, dtype=np.float32)
            # Ensure reward_sequence is 1D
            if self.reward_sequence.ndim > 1:
                self.reward_sequence = self.reward_sequence.flatten()
        
        if len(self.state_sequence) != len(self.reward_sequence):
            raise ValueError("state_sequence and reward_sequence must have same length")
        
        # Store reversal mask (maps trial index to reversal phase)
        if reversal_mask is not None:
            self.reversal_mask = np.array(reversal_mask)
            # Create a mapping from timestep to reversal phase
            # In simplified structure: stimulus at even indices (0, 2, 4...), outcome at odd (1, 3, 5...)
            self.timestep_to_reversal = {}
            trial_idx = 0
            for t_idx, state in enumerate(self.state_sequence):
                # Map stimulus states (even indices) to reversal phase
                if self._is_stimulus_state(state) and t_idx % 2 == 0:  # Stimulus state at even index
                    if trial_idx < len(self.reversal_mask):
                        self.timestep_to_reversal[t_idx] = self.reversal_mask[trial_idx]
                        trial_idx += 1
        else:
            self.reversal_mask = None
            self.timestep_to_reversal = {}
        
        self.render_mode = render_mode
        
        # Internal state
        self.current_timestep = 0
        self.max_timesteps = len(self.state_sequence) - 1
        
    def _generate_default_sequence(self, num_trials=10):
        """Generate a default state sequence for testing."""
        sequence = []
        for _ in range(num_trials):
            # Randomly choose A or B
            stim = self.np_random.choice(['A', 'B'])
            sequence.append(self.state_encodings[stim])
            # Add reward availability state (same as stimulus for simplicity)
            sequence.append(self.state_encodings[stim])
            # Add ITI states (unrewarded)
            sequence.append(self.state_encodings['unrewarded'])
            sequence.append(self.state_encodings['unrewarded'])
        return np.array(sequence)
    
    def _generate_reward_sequence(self):
        """Generate reward sequence based on state sequence."""
        rewards = []
        reward_contingency = {'A': True, 'B': False}  # Default: A rewarded, B not
        
        for state in self.state_sequence:
            # Check which stimulus state this is using state_encodings
            if self._is_state_equal(state, 'A'):
                rewards.append([1.0] if reward_contingency['A'] else [0.0])
            elif self._is_state_equal(state, 'B'):
                rewards.append([1.0] if reward_contingency['B'] else [0.0])
            else:
                rewards.append([0.0])
        
        return np.array(rewards)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start of sequence."""
        super().reset(seed=seed)
        
        self.current_timestep = 0
        
        # Get initial state from sequence
        self.current_state = self.state_sequence[0]
        
        info = {
            'timestep': self.current_timestep,
            'state_idx': int(np.argmax(self.current_state))
        }
        
        return self.current_state.copy(), info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (lick) or 1 (no_lick)
            
        Returns:
            observation: Next state (4D one-hot)
            reward: Reward received
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Get next state from sequence
        next_state = self.state_sequence[self.current_timestep + 1]
        
        # Check current state type by comparing with state_encodings
        is_stimulus_state = self._is_stimulus_state(self.current_state)
        is_outcome_state = self._is_outcome_state(self.current_state)
        
        if is_stimulus_state:
            # At stimulus state: agent takes action, outcome depends on action
            # The outcome state in state_sequence represents available reward
            # Check what outcome state is in the sequence at the next timestep
            outcome_timestep = self.current_timestep + 1
            if outcome_timestep < len(self.state_sequence):
                outcome_state_in_sequence = self.state_sequence[outcome_timestep]
                # Check if outcome state is 'rewarded'
                reward_available = self._is_state_equal(outcome_state_in_sequence, 'rewarded')
            else:
                reward_available = False
            
            # Calculate available reward value (what reward would be if agent licks)
            available_reward = self.reward_lick if reward_available else 0.0
            
            if action == 0:  # lick
                # Animal licks: outcome depends on whether reward is available
                if reward_available:
                    # Reward available and animal licks -> rewarded state
                    reward = self.reward_lick
                    self.current_state = self.state_encodings['rewarded']
                else:
                    # No reward available but animal licks -> unrewarded state
                    reward = self.lick_no_reward
                    self.current_state = self.state_encodings['unrewarded']
            else:  # no_lick
                # Animal doesn't lick: always unrewarded state (overrides sequence)
                reward = self.no_lick
                self.current_state = self.state_encodings['unrewarded']
            
            # Update timestep
            self.current_timestep += 1
            
        elif is_outcome_state:
            # At outcome state: no reward available (we're past the decision point)
            reward_available = False
            available_reward = 0.0
            # At outcome state: agent takes action to proceed to next stimulus
            # Action is required to transition (any action works, but we track it)
            reward = 0.0
            # Find next stimulus state in sequence
            next_stimulus_idx = None
            for idx in range(self.current_timestep + 1, len(self.state_sequence)):
                if self._is_stimulus_state(self.state_sequence[idx]):
                    next_stimulus_idx = idx
                    break
            
            if next_stimulus_idx is not None:
                self.current_state = self.state_sequence[next_stimulus_idx]
                self.current_timestep = next_stimulus_idx
            else:
                # No more stimuli, episode should end
                self.current_timestep += 1
        else:
            # Fallback: follow sequence normally
            reward = 0.0
            reward_available = False
            available_reward = 0.0
            if self.current_timestep + 1 < len(self.state_sequence):
                next_state = self.state_sequence[self.current_timestep + 1]
                self.current_state = next_state
            self.current_timestep += 1
        
        # Check if episode is done
        if self.current_timestep >= self.max_timesteps:
            terminated = True
        
        # Get reversal phase for current timestep if available
        reversal_phase = self.timestep_to_reversal.get(self.current_timestep - 1, None)
        
        info = {
            'timestep': self.current_timestep,
            'action': 'lick' if action == 0 else 'no_lick',
            'reward': reward,
            'reward_available': reward_available,
            'available_reward': available_reward,
            'state_idx': int(np.argmax(self.current_state)),
            'reversal_phase': reversal_phase
        }
        
        return self.current_state.copy(), reward, terminated, truncated, info
    
    def _is_stimulus_state(self, state):
        """Check if state matches any stimulus state encoding."""
        for stim_name in self.stimulus_states:
            if stim_name in self.state_encodings:
                if self._is_state_equal(state, stim_name):
                    return True
        return False
    
    def _is_outcome_state(self, state):
        """Check if state matches any outcome state encoding."""
        for outcome_name in self.outcome_states:
            if outcome_name in self.state_encodings:
                if self._is_state_equal(state, outcome_name):
                    return True
        return False
    
    def _is_state_equal(self, state, state_name):
        """Check if state vector matches a named state encoding."""
        if state_name not in self.state_encodings:
            return False
        return np.allclose(state, self.state_encodings[state_name])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            state_names = ['A', 'B', 'unrewarded', 'rewarded']
            state_idx = np.argmax(self.current_state)
            print(f"Timestep: {self.current_timestep}, "
                  f"State: {state_names[state_idx]}, "
                  f"State vector: {self.current_state}")


class ReversalABCEnv(gym.Env):
    """
    Reversal ABC task environment (Sandra's task).
    
    Uses a pre-generated sequence of states. When reward is available:
    - If agent licks: gets reward and follows sequence
    - If agent doesn't lick: goes to unrewarded state
    
    States (one-hot encoded):
    - A: Stimulus A [1, 0, 0, 0, 0]
    - B: Stimulus B [0, 1, 0, 0, 0]
    - C: Stimulus C [0, 0, 1, 0, 0]
    - unrewarded: No reward state [0, 0, 0, 1, 0]
    - rewarded: Reward state [0, 0, 0, 0, 1]
    
    Actions:
    - 0: lick
    - 1: no_lick
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, state_sequence=None, reward_sequence=None, reversal_mask=None,
                 reward_lick=1.0, lick_no_reward=-1.0, no_lick=0.0, render_mode=None,
                 state_map=None):
        """
        Initialize the environment.
        
        Args:
            state_sequence: Pre-generated sequence of states (N x 5 array).
                          If None, generates a simple sequence.
            reward_sequence: Pre-generated sequence of rewards (N x 1 array).
                           If None, generates based on state sequence.
            reversal_mask: Array indicating reversal phase (0=pre, 1=post) for each trial.
                          If None, creates default mask.
            reward_lick: Reward when reward is available and agent licks (default: 1.0)
            lick_no_reward: Reward when no reward available but agent licks (default: -1.0)
            no_lick: Reward when agent doesn't lick (default: 0.0)
            render_mode: Rendering mode
            state_map: Dict mapping state names to numpy arrays. If None, uses default encodings.
        """
        super().__init__()
        
        # Store reward values
        self.reward_lick = reward_lick
        self.lick_no_reward = lick_no_reward
        self.no_lick = no_lick
        
        # State space: 5D one-hot encoding (A, B, C, unrewarded, rewarded)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        # Action space: lick (0) or no_lick (1)
        self.action_space = spaces.Discrete(2)
        
        # Default state encodings (one-hot)
        default_state_encodings = {
            'A': np.array([1, 0, 0, 0, 0], dtype=np.float32),
            'B': np.array([0, 1, 0, 0, 0], dtype=np.float32),
            'C': np.array([0, 0, 1, 0, 0], dtype=np.float32),
            'unrewarded': np.array([0, 0, 0, 1, 0], dtype=np.float32),
            'rewarded': np.array([0, 0, 0, 0, 1], dtype=np.float32),
        }
        
        # Use state_map if provided, otherwise use defaults
        if state_map is not None:
            # Convert state_map values to numpy arrays if needed
            self.state_encodings = {}
            for key, value in state_map.items():
                if isinstance(value, (list, tuple)):
                    self.state_encodings[key] = np.array(value, dtype=np.float32)
                elif isinstance(value, np.ndarray):
                    self.state_encodings[key] = value.astype(np.float32)
                else:
                    self.state_encodings[key] = np.array(value, dtype=np.float32)
            # Merge with defaults for any missing keys
            for key, value in default_state_encodings.items():
                if key not in self.state_encodings:
                    self.state_encodings[key] = value
        else:
            self.state_encodings = default_state_encodings
        
        # Identify stimulus and outcome state names from state_encodings
        self.stimulus_states = ['A', 'B', 'C']  # For ABC env
        self.outcome_states = ['rewarded', 'unrewarded']
        
        # Track C stimulus rewards (for random 50% reward)
        # Initialize before generating sequences
        self.c_reward_outcomes = {}  # Maps timestep -> reward outcome for C
        
        # Store sequences
        if state_sequence is None:
            self.state_sequence = self._generate_default_sequence()
        else:
            self.state_sequence = np.array(state_sequence, dtype=np.float32)
            # Update observation space if state_dim differs
            state_dim = self.state_sequence.shape[-1]
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(state_dim,), dtype=np.float32
            )
        
        if reward_sequence is None:
            self.reward_sequence = self._generate_reward_sequence()
        else:
            self.reward_sequence = np.array(reward_sequence, dtype=np.float32)
            # Ensure reward_sequence is 1D
            if self.reward_sequence.ndim > 1:
                self.reward_sequence = self.reward_sequence.flatten()
        
        if len(self.state_sequence) != len(self.reward_sequence):
            raise ValueError("state_sequence and reward_sequence must have same length")
        
        # Store reversal mask (maps trial index to reversal phase)
        # Create a mapping from state sequence timestep to reversal phase
        if reversal_mask is not None:
            self.reversal_mask = np.array(reversal_mask)
            # Create a mapping from timestep to reversal phase
            # In simplified structure: stimulus at even indices (0, 2, 4...), outcome at odd (1, 3, 5...)
            self.timestep_to_reversal = {}
            trial_idx = 0
            for t_idx, state in enumerate(self.state_sequence):
                # Map stimulus states (even indices) to reversal phase
                if self._is_stimulus_state(state) and t_idx % 2 == 0:  # Stimulus state at even index
                    if trial_idx < len(self.reversal_mask):
                        self.timestep_to_reversal[t_idx] = self.reversal_mask[trial_idx]
                        trial_idx += 1
        else:
            self.reversal_mask = None
            self.timestep_to_reversal = {}
        
        self.render_mode = render_mode
        
        # Internal state
        self.current_timestep = 0
        self.max_timesteps = len(self.state_sequence) - 1
        
    def _generate_default_sequence(self, num_trials=10):
        """Generate a default state sequence for testing (simplified: no ITI)."""
        sequence = []
        reward_contingency = {'A': True, 'B': False}  # Default: A rewarded, B not
        
        for _ in range(num_trials):
            # Randomly choose A, B, or C
            stim = self.np_random.choice(['A', 'B', 'C'], p=[1/3, 1/3, 1/3])
            # Add stimulus state
            sequence.append(self.state_encodings[stim])
            
            # Add outcome state (rewarded or unrewarded based on contingency)
            if stim == 'A' and reward_contingency['A']:
                sequence.append(self.state_encodings['rewarded'])
            elif stim == 'B' and reward_contingency['B']:
                sequence.append(self.state_encodings['rewarded'])
            elif stim == 'C':
                # C has random 50% reward
                if self.np_random.rand() < 0.5:
                    sequence.append(self.state_encodings['rewarded'])
                else:
                    sequence.append(self.state_encodings['unrewarded'])
            else:
                sequence.append(self.state_encodings['unrewarded'])
        
        return np.array(sequence)
    
    def _generate_reward_sequence(self):
        """Generate reward sequence based on state sequence."""
        rewards = []
        reward_contingency = {'A': True, 'B': False}  # Default: A rewarded, B not
        
        for t_idx, state in enumerate(self.state_sequence):
            # Check which stimulus state this is using state_encodings
            if self._is_state_equal(state, 'A'):
                rewards.append([1.0] if reward_contingency['A'] else [0.0])
            elif self._is_state_equal(state, 'B'):
                rewards.append([1.0] if reward_contingency['B'] else [0.0])
            elif self._is_state_equal(state, 'C'):
                # C - random 50% reward
                # Pre-determine reward outcome for consistency
                if t_idx not in self.c_reward_outcomes:
                    self.c_reward_outcomes[t_idx] = 1.0 if self.np_random.rand() < 0.5 else 0.0
                rewards.append([self.c_reward_outcomes[t_idx]])
            else:
                rewards.append([0.0])
        
        return np.array(rewards)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start of sequence."""
        super().reset(seed=seed)
        
        self.current_timestep = 0
        self.c_reward_outcomes = {}  # Reset C reward outcomes
        
        # Get initial state from sequence
        self.current_state = self.state_sequence[0]
        
        state_idx = int(np.argmax(self.current_state))
        state_names = ['A', 'B', 'C', 'unrewarded', 'rewarded']
        
        info = {
            'timestep': self.current_timestep,
            'state_idx': state_idx,
            'state_name': state_names[state_idx] if state_idx < len(state_names) else 'unknown'
        }
        
        return self.current_state.copy(), info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Simplified structure: observation -> outcome -> observation -> outcome
        - At stimulus states (A, B, C): agent takes action
        - At outcome states (rewarded/unrewarded): automatically transition to next stimulus
        
        Args:
            action: 0 (lick) or 1 (no_lick)
            
        Returns:
            observation: Next state (5D one-hot)
            reward: Reward received
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Check current state type by comparing with state_encodings
        is_stimulus_state = self._is_stimulus_state(self.current_state)
        is_outcome_state = self._is_outcome_state(self.current_state)
        
        if is_stimulus_state:
            # At stimulus state: agent takes action, outcome depends on action
            # The outcome state in state_sequence represents available reward
            # Check what outcome state is in the sequence at the next timestep
            outcome_timestep = self.current_timestep + 1
            if outcome_timestep < len(self.state_sequence):
                outcome_state_in_sequence = self.state_sequence[outcome_timestep]
                # Check if outcome state is 'rewarded'
                reward_available = self._is_state_equal(outcome_state_in_sequence, 'rewarded')
            else:
                reward_available = False
            
            # Calculate available reward value (what reward would be if agent licks)
            available_reward = self.reward_lick if reward_available else 0.0
            
            if action == 0:  # lick
                # Animal licks: outcome depends on whether reward is available
                if reward_available:
                    # Reward available and animal licks -> rewarded state
                    reward = self.reward_lick
                    self.current_state = self.state_encodings['rewarded']
                else:
                    # No reward available but animal licks -> unrewarded state
                    reward = self.lick_no_reward
                    self.current_state = self.state_encodings['unrewarded']
            else:  # no_lick
                # Animal doesn't lick: always unrewarded state (overrides sequence)
                reward = self.no_lick
                self.current_state = self.state_encodings['unrewarded']
            
            # Update timestep
            self.current_timestep += 1
            
        elif is_outcome_state:
            # At outcome state: no reward available (we're past the decision point)
            reward_available = False
            available_reward = 0.0
            # At outcome state: agent takes action to proceed to next stimulus
            # Action is required to transition (any action works, but we track it)
            reward = 0.0
            # Find next stimulus state in sequence
            next_stimulus_idx = None
            for idx in range(self.current_timestep + 1, len(self.state_sequence)):
                if self._is_stimulus_state(self.state_sequence[idx]):
                    next_stimulus_idx = idx
                    break
            
            if next_stimulus_idx is not None:
                self.current_state = self.state_sequence[next_stimulus_idx]
                self.current_timestep = next_stimulus_idx
            else:
                # No more stimuli, episode should end
                self.current_timestep += 1
        else:
            # Fallback: follow sequence normally
            reward = 0.0
            reward_available = False
            available_reward = 0.0
            if self.current_timestep + 1 < len(self.state_sequence):
                next_state = self.state_sequence[self.current_timestep + 1]
                self.current_state = next_state
            self.current_timestep += 1
        
        # Check if episode is done
        if self.current_timestep >= self.max_timesteps:
            terminated = True
        
        state_idx = int(np.argmax(self.current_state))
        state_names = ['A', 'B', 'C', 'unrewarded', 'rewarded']
        
        # Get reversal phase for current timestep if available
        # In simplified structure: stimulus at even indices (0, 2, 4...), outcome at odd (1, 3, 5...)
        # Trial index = timestep // 2 (each trial has 2 timesteps: stimulus + outcome)
        trial_idx = self.current_timestep // 2
        reversal_phase = self.timestep_to_reversal.get(trial_idx * 2, None)  # Map to stimulus timestep
        
        info = {
            'timestep': self.current_timestep,
            'action': 'lick' if action == 0 else 'no_lick',
            'reward': reward,
            'reward_available': reward_available,
            'available_reward': available_reward,
            'state_idx': state_idx,
            'state_name': state_names[state_idx] if state_idx < len(state_names) else 'unknown',
            'reversal_phase': reversal_phase
        }
        
        return self.current_state.copy(), reward, terminated, truncated, info
    
    def _is_stimulus_state(self, state):
        """Check if state matches any stimulus state encoding."""
        for stim_name in self.stimulus_states:
            if stim_name in self.state_encodings:
                if self._is_state_equal(state, stim_name):
                    return True
        return False
    
    def _is_outcome_state(self, state):
        """Check if state matches any outcome state encoding."""
        for outcome_name in self.outcome_states:
            if outcome_name in self.state_encodings:
                if self._is_state_equal(state, outcome_name):
                    return True
        return False
    
    def _is_state_equal(self, state, state_name):
        """Check if state vector matches a named state encoding."""
        if state_name not in self.state_encodings:
            return False
        return np.allclose(state, self.state_encodings[state_name])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            state_names = ['A', 'B', 'C', 'unrewarded', 'rewarded']
            state_idx = np.argmax(self.current_state)
            state_name = state_names[state_idx] if state_idx < len(state_names) else 'unknown'
            print(f"Timestep: {self.current_timestep}, "
                  f"State: {state_name}, "
                  f"State vector: {self.current_state}")


def load_reversal_data(data_path):
    """
    Load reversal AB task data from pickle file and prepare for environment.
    
    Args:
        data_path: Path to pickle file containing task data
        
    Returns:
        state_sequence: Array of states (N x 4)
        reward_sequence: Array of rewards (N x 1)
        reversal_mask: Array indicating reversal phase (0=pre, 1=post)
        phase_boundaries: Dict with phase boundaries and reversal points
    """
    with open(data_path, "rb") as fid:
        data = pickle.load(fid)
    
    # Extract state sequence (one-hot encoded)
    state_sequence = data.get("state_sequence_ohe", data.get("sequence_ohe"))
    
    # Extract sequence data with rewards and reversal mask
    sequence = data.get("sequence", {})
    rewards_list = sequence.get("rewards", [])
    reversal_mask = np.array(sequence.get("masks", {}).get("reversal", []))
    
    # Extract phase boundaries if available
    phase_boundaries = data.get("phase_boundaries", None)
    
    # Create reward sequence aligned with state sequence
    # Simplified structure: stimulus (even indices) -> outcome (odd indices)
    # Rewards should be available at stimulus states (A or B) when agent licks
    reward_sequence = []
    reward_idx = 0
    
    for t_idx, state in enumerate(state_sequence):
        state_idx = np.argmax(state)
        # If this is a stimulus state (A=0 or B=1) at even index, check if reward is available
        if state_idx in [0, 1] and t_idx % 2 == 0 and reward_idx < len(rewards_list):
            # Reward is available if the corresponding trial has reward=1
            reward_value = rewards_list[reward_idx] if rewards_list[reward_idx] == 1 else 0.0
            reward_sequence.append([reward_value])
            reward_idx += 1
        else:
            # No reward available at this timestep (outcome states or non-stimulus states)
            reward_sequence.append([0.0])
    
    return state_sequence, np.array(reward_sequence), reversal_mask, phase_boundaries

def load_reversal_abc_data(data_path):
    with open(data_path, "rb") as fid:
        data = pickle.load(fid)

    # state_sequence_ohe is interleaved: [stimulus_0, outcome_0, stimulus_1, outcome_1, ...]
    # data["sequence"]["stimuli"] and data["sequence"]["rewards"] corresponds to just the outcome states (half the length)
    # Create a reward sequence that matches state_sequence_ohe length:
    # - zeros for stimulus states (even indices)
    # - actual reward values for outcome states (odd indices)
    state_sequence_ohe = data["state_sequence_ohe"]
    rewards = data["sequence"]["rewards"]
    
    # Convert rewards to numpy array if needed and ensure it's 1D
    if isinstance(rewards, list):
        rewards = np.array(rewards)
    if rewards.ndim > 1:
        rewards = rewards.flatten()
    
    # Number of outcome states (odd indices in state_sequence_ohe)
    num_outcome_states = len(state_sequence_ohe) // 2
    
    # Ensure rewards length matches expected number of outcome states
    if len(rewards) < num_outcome_states:
        # Pad with zeros if rewards is shorter (shouldn't happen normally)
        rewards = np.pad(rewards, (0, num_outcome_states - len(rewards)), mode='constant')
    elif len(rewards) > num_outcome_states:
        # Truncate if rewards is longer
        rewards = rewards[:num_outcome_states]
    
    # Create interleaved reward sequence matching state_sequence_ohe length
    reward_sequence = np.zeros(len(state_sequence_ohe))
    # Fill odd indices (outcome states) with actual rewards
    reward_sequence[1::2] = rewards

    return state_sequence_ohe, reward_sequence, data["sequence"]["masks"]["reversal"], data["phase_boundaries"], data.get("state_map", None)


def load_reversal_abc_multi_data(data_path):
    """
    Load multi-reversal ABC task data from pickle file and prepare for environment.
    
    This is similar to load_reversal_abc_data but handles multi-reversal sequences
    where reversal_mask can have multiple phase values (0, 1, 2, 3, ...).
    
    Args:
        data_path: Path to pickle file containing task data
        
    Returns:
        state_sequence: Array of states (N x 5)
        reward_sequence: Array of rewards (N,)
        reversal_mask: Array indicating reversal phase (0, 1, 2, 3, ...)
        phase_boundaries: Dict with phase boundaries and reversal points
        state_map: Dict mapping state names to indices
    """
    with open(data_path, "rb") as fid:
        data = pickle.load(fid)

    # state_sequence_ohe is interleaved: [stimulus_0, outcome_0, stimulus_1, outcome_1, ...]
    # data["sequence"]["stimuli"] and data["sequence"]["rewards"] corresponds to just the outcome states (half the length)
    state_sequence_ohe = data["state_sequence_ohe"]
    rewards = data["sequence"]["rewards"]
    
    # Convert rewards to numpy array if needed and ensure it's 1D
    if isinstance(rewards, list):
        rewards = np.array(rewards)
    if rewards.ndim > 1:
        rewards = rewards.flatten()
    
    # Number of outcome states (odd indices in state_sequence_ohe)
    num_outcome_states = len(state_sequence_ohe) // 2
    
    # Ensure rewards length matches expected number of outcome states
    if len(rewards) < num_outcome_states:
        # Pad with zeros if rewards is shorter (shouldn't happen normally)
        rewards = np.pad(rewards, (0, num_outcome_states - len(rewards)), mode='constant')
    elif len(rewards) > num_outcome_states:
        # Truncate if rewards is longer
        rewards = rewards[:num_outcome_states]
    
    # Create interleaved reward sequence matching state_sequence_ohe length
    reward_sequence = np.zeros(len(state_sequence_ohe))
    # Fill odd indices (outcome states) with actual rewards
    reward_sequence[1::2] = rewards

    # Get phase boundaries if available
    phase_boundaries = data.get("phase_boundaries", None)

    return state_sequence_ohe, reward_sequence, data["sequence"]["masks"]["reversal"], phase_boundaries, data.get("state_map", None)

class ReversalABCDynamicEnv(gym.Env):
    """
    Reversal ABC task environment that generates trials dynamically.
    
    Does not require pre-generated sequences. Instead, generates stimuli and
    determines rewards on-the-fly based on reward contingencies:
    - A: Always rewarded (1.0) when agent licks (pre-reversal), never (post-reversal)
    - B: Never rewarded (pre-reversal), rewarded 50% of the time (post-reversal)
    - C: Rewarded 50% of the time (doesn't reverse)
    
    Supports reversal learning with pre-reversal and post-reversal phases.
    
    States (one-hot encoded):
    - A: Stimulus A [1, 0, 0, 0, 0]
    - B: Stimulus B [0, 1, 0, 0, 0]
    - C: Stimulus C [0, 0, 1, 0, 0]
    - unrewarded: No reward state [0, 0, 0, 1, 0]
    - rewarded: Reward state [0, 0, 0, 0, 1]
    
    Actions:
    - 0: lick
    - 1: no_lick
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, num_trials=100, reversal_trial=50, seed=None, render_mode=None,
                 state_map=None):
        """
        Initialize the dynamic reversal ABC environment.
        
        Args:
            num_trials: Total number of trials in the episode
            reversal_trial: Trial number at which reversal occurs (0-indexed)
            seed: Random seed for reproducibility
            render_mode: Rendering mode
            state_map: Dict mapping state names to numpy arrays. If None, uses default encodings.
        """
        super().__init__()
        
        # State space: 5D one-hot encoding (A, B, C, unrewarded, rewarded)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        # Action space: lick (0) or no_lick (1)
        self.action_space = spaces.Discrete(2)
        
        # Default state encodings (one-hot)
        default_state_encodings = {
            'A': np.array([1, 0, 0, 0, 0], dtype=np.float32),
            'B': np.array([0, 1, 0, 0, 0], dtype=np.float32),
            'C': np.array([0, 0, 1, 0, 0], dtype=np.float32),
            'unrewarded': np.array([0, 0, 0, 1, 0], dtype=np.float32),
            'rewarded': np.array([0, 0, 0, 0, 1], dtype=np.float32),
        }
        
        # Use state_map if provided, otherwise use defaults
        if state_map is not None:
            # Convert state_map values to numpy arrays if needed
            self.state_encodings = {}
            for key, value in state_map.items():
                if isinstance(value, (list, tuple)):
                    self.state_encodings[key] = np.array(value, dtype=np.float32)
                elif isinstance(value, np.ndarray):
                    self.state_encodings[key] = value.astype(np.float32)
                else:
                    self.state_encodings[key] = np.array(value, dtype=np.float32)
            # Merge with defaults for any missing keys
            for key, value in default_state_encodings.items():
                if key not in self.state_encodings:
                    self.state_encodings[key] = value
        else:
            self.state_encodings = default_state_encodings
        
        # Identify stimulus and outcome state names from state_encodings
        self.stimulus_states = ['A', 'B', 'C']  # For ABC env
        self.outcome_states = ['rewarded', 'unrewarded']
        
        # Environment parameters
        self.num_trials = num_trials
        self.reversal_trial = reversal_trial
        self.render_mode = render_mode
        
        # Internal state
        self.current_trial = 0
        self.current_state_type = None  # 'stimulus' or 'outcome'
        self.current_stimulus = None  # 'A', 'B', or 'C'
        self.current_state = None
        
        # Pre-determine B and C reward outcomes for consistency
        # B: 50% reward probability (only in post-reversal)
        # C: 50% reward probability (always)
        self.b_reward_outcomes = {}  # Maps trial -> reward available (True/False)
        self.c_reward_outcomes = {}  # Maps trial -> reward available (True/False)
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        self._seed_b_rewards()
        self._seed_c_rewards()
        
    def _seed_b_rewards(self):
        """Pre-determine B reward outcomes (50% probability)."""
        for trial in range(self.num_trials):
            self.b_reward_outcomes[trial] = self.np_random.rand() < 0.5
    
    def _seed_c_rewards(self):
        """Pre-determine C reward outcomes (50% probability)."""
        for trial in range(self.num_trials):
            self.c_reward_outcomes[trial] = self.np_random.rand() < 0.5
    
    def _get_reward_available(self, stimulus):
        """
        Check if reward is available for a given stimulus at current trial.
        
        Args:
            stimulus: 'A', 'B', or 'C'
            
        Returns:
            bool: True if reward is available, False otherwise
        """
        # Determine reversal phase
        is_pre_reversal = self.current_trial < self.reversal_trial
        
        if stimulus == 'A':
            # A: Always rewarded in pre-reversal, never in post-reversal
            return is_pre_reversal
        elif stimulus == 'B':
            # B: Never rewarded in pre-reversal, 50% in post-reversal
            if is_pre_reversal:
                return False
            else:
                return self.b_reward_outcomes.get(self.current_trial, False)
        elif stimulus == 'C':
            # C: Always 50% rewarded (doesn't reverse)
            return self.c_reward_outcomes.get(self.current_trial, False)
        else:
            return False
    
    def _generate_stimulus(self):
        """Generate next stimulus randomly (equal probability for A, B, C)."""
        return self.np_random.choice(['A', 'B', 'C'], p=[1/3, 1/3, 1/3])
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start of episode."""
        super().reset(seed=seed)
        
        self.current_trial = 0
        self.current_state_type = 'stimulus'
        self.current_stimulus = self._generate_stimulus()
        self.current_state = self.state_encodings[self.current_stimulus]
        
        state_idx = int(np.argmax(self.current_state))
        state_names = ['A', 'B', 'C', 'unrewarded', 'rewarded']
        is_pre_reversal = self.current_trial < self.reversal_trial
        
        info = {
            'timestep': 0,
            'trial': self.current_trial,
            'state_idx': state_idx,
            'state_name': state_names[state_idx] if state_idx < len(state_names) else 'unknown',
            'stimulus': self.current_stimulus,
            'reversal_phase': 0 if is_pre_reversal else 1
        }
        
        return self.current_state.copy(), info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (lick) or 1 (no_lick)
            
        Returns:
            observation: Next state (5D one-hot)
            reward: Reward received
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.current_state_type == 'stimulus':
            # At stimulus state: agent takes action, outcome depends on action
            reward_available = self._get_reward_available(self.current_stimulus)
            
            if action == 0:  # lick
                # Give reward if available
                reward = 1.0 if reward_available else 0.0
                # Move to outcome state based on whether reward was received
                if reward_available:
                    self.current_state = self.state_encodings['rewarded']
                else:
                    self.current_state = self.state_encodings['unrewarded']
            else:  # no_lick
                # No reward, always move to unrewarded state
                reward = 0.0
                self.current_state = self.state_encodings['unrewarded']
            
            # Transition to outcome state
            self.current_state_type = 'outcome'
            
        elif self.current_state_type == 'outcome':
            # At outcome state: agent takes action to proceed to next stimulus
            # Action is required to transition (any action works)
            reward = 0.0
            
            # Determine reversal phase BEFORE incrementing trial
            # (based on the trial that just completed)
            is_pre_reversal = self.current_trial < self.reversal_trial
            reversal_phase = 0 if is_pre_reversal else 1
            
            # Move to next trial
            self.current_trial += 1
            
            # Check if episode is done
            if self.current_trial >= self.num_trials:
                terminated = True
                self.current_state = self.state_encodings['unrewarded']
            else:
                # Generate next stimulus
                self.current_stimulus = self._generate_stimulus()
                self.current_state = self.state_encodings[self.current_stimulus]
                self.current_state_type = 'stimulus'
        else:
            # Determine reversal phase for stimulus states
            is_pre_reversal = self.current_trial < self.reversal_trial
            reversal_phase = 0 if is_pre_reversal else 1
        
        state_idx = int(np.argmax(self.current_state))
        state_names = ['A', 'B', 'C', 'unrewarded', 'rewarded']
        
        info = {
            'timestep': self.current_trial * 2 + (1 if self.current_state_type == 'outcome' else 0),
            'trial': self.current_trial,
            'action': 'lick' if action == 0 else 'no_lick',
            'reward': reward,
            'state_idx': state_idx,
            'state_name': state_names[state_idx] if state_idx < len(state_names) else 'unknown',
            'stimulus': self.current_stimulus if self.current_state_type == 'stimulus' else None,
            'reversal_phase': reversal_phase
        }
        
        return self.current_state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            state_names = ['A', 'B', 'C', 'unrewarded', 'rewarded']
            state_idx = np.argmax(self.current_state)
            state_name = state_names[state_idx] if state_idx < len(state_names) else 'unknown'
            print(f"Trial: {self.current_trial}, "
                  f"State type: {self.current_state_type}, "
                  f"State: {state_name}, "
                  f"Stimulus: {self.current_stimulus}")

"""
Agent classes for reinforcement learning.

This module contains:
- A2CAgent: Actor-Critic agent implementing the A2C algorithm
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from cog_nn.models import RNNActorCritic, SelfSupervisedRNNActorCritic

class A2CAgent:
    """
    Actor-Critic agent implementing the Advantage Actor-Critic (A2C) algorithm.
    
    The agent uses separate Actor and Critic networks to:
    - Actor: Learn a policy (action probabilities)
    - Critic: Learn state values
    
    Updates are performed using advantage estimates: A = R + γV(s') - V(s)
    """
    
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, 
                 gamma=0.9, device='cpu'):
        """
        Initialize A2C agent.
        
        Args:
            actor: Actor network (policy)
            critic: Critic network (value function)
            actor_optimizer: Optimizer for actor network
            critic_optimizer: Optimizer for critic network
            gamma: Discount factor for future rewards
            device: Device to run on ('cpu' or 'cuda')
        """
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.device = device
        
        # Move networks to device
        self.actor.to(device)
        self.critic.to(device)
        
        # Training metrics
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'rewards': [],
            'advantages': [],
            'values': []
        }
    
    def select_action(self, state, deterministic=False, policy_clip=None):
        """
        Select an action given the current state.
        
        Args:
            state: Current state (can be tensor or numpy array)
            deterministic: If True, select greedy action. If False, sample from policy.
            
        Returns:
            action: Selected action (integer)
            action_prob: Probability of selected action
            value: State value estimate
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        state = state.to(self.device)
        
        # Get action probabilities and value
        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)

            # Prevent zero probability
            if policy_clip is not None:
                action_probs = F.softmax(action_probs, dim=-1)
                action_probs = torch.clamp(action_probs, min=policy_clip)
                
                # Normalise again
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        if deterministic:
            # Greedy action
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
        else:
            # Sample from policy
            dist = Categorical(action_probs)
            action = dist.sample().item()
            action_prob = action_probs[action].item()
       

        return action, action_prob, value.item()
    
    def update(self, states, actions, rewards, next_states, dones):
        """
        Update Actor and Critic networks using A2C algorithm.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            dones: Batch of done flags (batch_size,)
        """
        # Convert to tensors
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).long()
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states).float()
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).float()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current values and next values
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        # Compute target values: R + γ * V(s') (if not done)
        # Detach next_values for stable bootstrapping (standard in TD learning)
        # This prevents gradients from flowing through the bootstrap estimate
        target_values = rewards + self.gamma * next_values.detach() * (1 - dones)
        
        # Compute advantages: A = target_value - V(s)
        # Keep target_values attached for critic loss, but detach when used for actor
        advantages = target_values - values
        
        # Get action probabilities
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Actor loss: -log_prob * advantage (policy gradient)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss: (target_value - V(s))^2 (value function error)
        # Use non-detached target_values so gradients flow through to critic
        critic_loss = (target_values - values).pow(2).mean()
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Store metrics
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        
        # Convert to numpy and handle scalar case (when batch_size=1)
        rewards_np = rewards.cpu().numpy()
        advantages_np = advantages.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        
        # Ensure we have arrays (not scalars) for extend()
        if rewards_np.ndim == 0:
            rewards_np = rewards_np.reshape(1)
        if advantages_np.ndim == 0:
            advantages_np = advantages_np.reshape(1)
        if values_np.ndim == 0:
            values_np = values_np.reshape(1)
        
        self.training_history['rewards'].extend(rewards_np.tolist())
        self.training_history['advantages'].extend(advantages_np.tolist())
        self.training_history['values'].extend(values_np.tolist())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': values.mean().item()
        }
    
    def get_action_probabilities(self, state):
        """
        Get action probabilities for a given state (without sampling).
        
        Args:
            state: Current state
            
        Returns:
            action_probs: Action probabilities (numpy array)
            value: State value estimate
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        state = state.to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)
        
        return action_probs.cpu().numpy(), value.item()
    
    def reset_history(self):
        """Reset training history."""
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'rewards': [],
            'advantages': [],
            'values': []
        }

class TabularACAgent:
    """
    Tabular Actor-Critic agent for discrete state and action spaces.
    
    Uses lookup tables for policy (actor) and value function (critic).
    """
    def __init__(self, state_size, action_size, actor_lr=0.1, critic_lr=0.1, gamma=0.9):
        """
        Initialize Tabular A2C agent.
        
        Args:
            state_size: Number of discrete states
            action_size: Number of discrete actions
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor for future rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        
        # Initialize policy and value tables
        self.policy_table = np.ones((state_size, action_size)) / action_size  # Uniform policy
        self.value_table = np.zeros(state_size)  # Zero initial values
        
    def select_action(self, state, deterministic=False):
        """
        Select an action given the current state.
        
        Args:
            state: Current discrete state (integer)
            deterministic: If True, select greedy action. If False, sample from policy.
            
        Returns:
            action: Selected action (integer)
            action_prob: Probability of selected action
            value: State value estimate
        """
        action_probs = self.policy_table[state]
        
        if deterministic:
            action = np.argmax(action_probs)
            action_prob = action_probs[action]
        else:
            action = np.random.choice(self.action_size, p=action_probs)
            action_prob = action_probs[action]
        
        value = self.value_table[state]
        
        return action, action_prob, value
    
    def update(self, state, action, reward, next_state, done):
        """
        Update policy and value tables using A2C algorithm.
        
        Args:
            state: Current state (integer)
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next state (integer)
            done: Done flag (boolean)
        """
        # Get current value and next value
        value = self.value_table[state]
        next_value = self.value_table[next_state] if not done else 0.0
        
        # Compute target value and advantage
        target_value = reward + self.gamma * next_value
        advantage = target_value - value

        # Update policy table (actor)
        self.policy_table[state, action] += self.actor_lr * advantage

        # Update value table (critic)
        self.value_table[state] += self.critic_lr * (target_value - value)

        # Normalize policy updates
        self.policy_table[state] /= np.sum(self.policy_table[state])

        return {
            'advantage': advantage, 
            'value': value,
            'target_value': target_value
        }       

class TabularVLearner:
    """
    Tabular Value Learner for discrete state spaces.
    
    Uses a lookup table for state values.
    """
    def __init__(self, state_size, lr=0.1, gamma=0.9):
        """
        Initialize Tabular Value Learner.
        
        Args:
            state_size: Number of discrete states
            lr: Learning rate
            gamma: Discount factor for future rewards
        """
        self.state_size = state_size
        self.lr = lr
        self.gamma = gamma
        
        # Initialize value table
        self.value_table = np.zeros(state_size)
        
    def update(self, state, reward, next_state, done):
        """
        Update value table using TD(0) update rule.
        
        Args:
            state: Current state (integer)
            reward: Reward received (float)
            next_state: Next state (integer)
            done: Done flag (boolean)
        """
        # Get current value and next value
        value = self.value_table[state]
        next_value = self.value_table[next_state] if not done else 0.0
        
        # Compute target value
        target_value = reward + self.gamma * next_value
        
        # Update value
        self.value_table[state] += self.lr * (target_value - value)
        
        return {
            'value': value,
            'target_value': target_value
        }
    
class QLearner:
    """
    Tabular Q-Learning agent for discrete state and action spaces.
    
    Uses a lookup table for Q-values.
    """
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.9):
        """
        Initialize Tabular Q-Learning agent.
        
        Args:
            state_size: Number of discrete states
            action_size: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor for future rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
    def select_action(self, state, epsilon=0.1):
        """
        Select an action using ε-greedy policy.
        
        Args:
            state: Current discrete state (integer)
            epsilon: Probability of choosing a random action
            
        Returns:
            action: Selected action (integer)
            action_value: Q-value of selected action
        """
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(self.q_table[state])
        
        action_value = self.q_table[state, action]
        
        return action, action_value
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state (integer)
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next state (integer)
            done: Done flag (boolean)
        """
        # Get current Q-value and max next Q-value
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state]) if not done else 0.0
        
        # Compute target Q-value
        target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state, action] += self.lr * (target_q - current_q)
        
        return {
            'current_q': current_q,
            'target_q': target_q
        }

class SARSA:
    """
    TODO: write up SARSA agent
    """
    pass


class DQN:
    """
    TODO: implement Deep-Q Network
    """
    pass

class MetaA2CAgent:
    """Meta-Reinforcement Learning agent using an RNN-based Actor-Critic model, as defined by the RNNActorCritic model in models.py."""
    def __init__(self, state_size, action_size, hidden_size=128, hook_fn=None):
        self.model = RNNActorCritic(state_size, action_size, hidden_size, hook_fn)
        self.hidden_state = None  # Initialize hidden state to None
    
    def reset_hidden_state(self):
        """Reset the RNN hidden state."""
        self.hidden_state = None
    
    def select_action(self, state, prev_action, prev_reward, deterministic=False, policy_clip=None, return_rnn_out=False):
        """
        Select an action given the current state, previous action, and previous reward.
        
        Args:
            state: Current state (tensor)
            prev_action: Previous action (one-hot tensor)
            prev_reward: Previous reward (tensor)
            deterministic: If True, select greedy action. If False, sample from policy.
            policy_clip: Minimum probability for each action (prevents zero probability). If None, no clipping.
            return_rnn_out: If True, also return the RNN output (hidden state representation).
        """
        state = state.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        prev_action = prev_action.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        prev_reward = prev_reward.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        
        with torch.no_grad():
            if return_rnn_out:
                action_probs, value, self.hidden_state, rnn_out = self.model(state, prev_action, prev_reward, self.hidden_state, return_rnn_out=True)
            else:
                action_probs, value, self.hidden_state = self.model(state, prev_action, prev_reward, self.hidden_state)
        
        action_probs = action_probs.squeeze(0)  # Remove batch dim
        
        # Prevent zero probability with policy clipping
        if policy_clip is not None:
            action_probs = torch.clamp(action_probs, min=policy_clip)
            # Normalise again
            action_probs = action_probs / action_probs.sum()
        
        if deterministic:
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            action_prob = action_probs[action].item()
        
        if return_rnn_out:
            rnn_out = rnn_out.squeeze(0)  # Remove batch dim
            return action, action_prob, value.item(), rnn_out
        return action, action_prob, value.item()
    
    def update(self, states, prev_actions, prev_rewards, actions, rewards, next_states, next_prev_actions, next_prev_rewards, dones):
        """
        Update the RNN Actor-Critic model using the provided batch of experience.
        
        Args:
            states: Batch of current states (batch_size, state_dim)
            prev_actions: Batch of previous actions (batch_size, action_dim)
            prev_rewards: Batch of previous rewards (batch_size,)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            next_prev_actions: Batch of next previous actions (batch_size, action_dim)
            next_prev_rewards: Batch of next previous rewards (batch_size,)
            dones: Batch of done flags (batch_size,)
        """
        # Convert to tensors and move to device
        device = next(self.model.parameters()).device
        
        states = states.to(device)
        prev_actions = prev_actions.to(device)
        prev_rewards = prev_rewards.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        next_prev_actions = next_prev_actions.to(device)
        next_prev_rewards = next_prev_rewards.to(device)
        dones = dones.to(device)
        
        # Get current action probabilities and values
        action_probs, values, _ = self.model(states, prev_actions, prev_rewards)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Get next values
        with torch.no_grad():
            _, next_values, _ = self.model(next_states, next_prev_actions, next_prev_rewards)
        
        # Compute target values and advantages
        target_values = rewards + self.model.gamma * next_values * (1 - dones)
        advantages = target_values - values
        
        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = (target_values - values).pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + critic_loss
        
        # Backpropagation
        self.model.optimizer.zero_grad()
        total_loss.backward()
        self.model.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': values.mean().item()
        }

class SelfSupervisedRNNA2CAgent:
    # TODO: Check training logic of this - perhaps SSL loss should just train RNN and AC losses should just train heads?
    """Meta-RL agent with self-supervised RNN Actor-Critic model."""
    def __init__(self, state_size, action_size, hidden_size=128, hook_fn=None):
        self.model = SelfSupervisedRNNActorCritic(state_size, action_size, hidden_size, hook_fn)
        self.hidden_state = None  # Initialize hidden state to None
    
    def reset_hidden_state(self):
        """Reset the RNN hidden state."""
        self.hidden_state = None
    
    def select_action(self, state, prev_action, prev_reward, deterministic=False):
        """
        Select an action given the current state, previous action, and previous reward.
        
        Args:
            state: Current state (tensor)
            prev_action: Previous action (one-hot tensor)
            prev_reward: Previous reward (tensor)
            deterministic: If True, select greedy action. If False, sample from policy.
        """
        state = state.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        prev_action = prev_action.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        prev_reward = prev_reward.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        
        with torch.no_grad():
            action_probs, value, self.hidden_state = self.model(state, prev_action, prev_reward, self.hidden_state)
        
        action_probs = action_probs.squeeze(0)  # Remove batch dim
        
        if deterministic:
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            action_prob = action_probs[action].item()
        
        return action, action_prob, value.item()
    
    def update(self, states, prev_actions, actions, rewards, next_states, next_prev_actions, dones):
        """
        Update the RNN Actor-Critic model using the provided batch of experience.
        Args:
            states: Batch of current states (batch_size, state_dim)
            prev_actions: Batch of previous actions (batch_size, action_dim)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            next_prev_actions: Batch of next previous actions (batch_size, action_dim)
            dones: Batch of done flags (batch_size,)
        """
        # Convert to tensors and move to device
        device = next(self.model.parameters()).device
        
        states = states.to(device)
        prev_actions = prev_actions.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        next_prev_actions = next_prev_actions.to(device)
        dones = dones.to(device)
        
        # Get current action probabilities and values
        action_probs, values, _ = self.model(states, prev_actions, None)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Get next values
        with torch.no_grad():
            _, next_values, ss_output, _ = self.model(next_states, next_prev_actions, None)
        
        # Compute target values and advantages
        target_values = rewards + self.model.gamma * next_values * (1 - dones)
        advantages = target_values - values
        
        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = (target_values - values).pow(2).mean()
        
        # Self-supervised loss (predict next state)
        ss_loss = F.mse_loss(ss_output, next_states)

        # Total loss
        total_loss = actor_loss + critic_loss + ss_loss

        # Backpropagation
        self.model.optimizer.zero_grad()
        total_loss.backward()
        self.model.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': values.mean().item()
        }
    
class SelfSupervisedRNNA2CAgentSeparateTraining:
    """
    Meta-RL agent with self-supervised RNN Actor-Critic model in which AC and SSL losses are used to train core and heads separately
    (stop grad on connetion between RNN and heads).
    """
    def __init__(self, state_size, action_size, hidden_size=128, hook_fn=None):
        self.model = SelfSupervisedRNNActorCritic(state_size, action_size, hidden_size, hook_fn)
        self.hidden_state = None  # Initialize hidden state to None
            
    def reset_hidden_state(self):
        """Reset the RNN hidden state."""
        self.hidden_state = None
    
    def select_action(self, state, prev_action, prev_reward, deterministic=False):
        """
        Select an action given the current state, previous action, and previous reward.
        
        Args:
            state: Current state (tensor)
            prev_action: Previous action (one-hot tensor)
            prev_reward: Previous reward (tensor)
            deterministic: If True, select greedy action. If False, sample from policy.
        """
        state = state.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        prev_action = prev_action.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        prev_reward = prev_reward.unsqueeze(0).to(next(self.model.parameters()).device)  # Add batch dim
        
        with torch.no_grad():
            action_probs, value, self.hidden_state = self.model(state, prev_action, prev_reward, self.hidden_state)
        
        action_probs = action_probs.squeeze(0)  # Remove batch dim
        
        if deterministic:
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            action_prob = action_probs[action].item()
        
        return action, action_prob, value.item()

    def update(self, states, prev_actions, actions, rewards, next_states, next_prev_actions, dones):
        """
        Update the RNN Actor-Critic model using the provided batch of experience.
        Args:
            states: Batch of current states (batch_size, state_dim)
            prev_actions: Batch of previous actions (batch_size, action_dim)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            next_prev_actions: Batch of next previous actions (batch_size, action_dim)
            dones: Batch of done flags (batch_size,)
        """
        # Convert to tensors and move to device
        device = next(self.model.parameters()).device
        
        states = states.to(device)
        prev_actions = prev_actions.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        next_prev_actions = next_prev_actions.to(device)
        dones = dones.to(device)
        
        # Get current action probabilities and values
        action_probs, values, _ = self.model(states, prev_actions, None)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Get next values
        with torch.no_grad():
            _, next_values, ss_output, _ = self.model(next_states, next_prev_actions, None)
        
        # Compute target values and advantages
        target_values = rewards + self.model.gamma * next_values * (1 - dones)
        advantages = target_values - values
        
        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = (target_values - values).pow(2).mean()
        
        # Self-supervised loss (predict next state)
        ss_loss = F.mse_loss(ss_output, next_states)

        # Backpropagation for Actor-Critic (stop grad on RNN to heads connection)
        self.model.optimizer.zero_grad()
        (actor_loss + critic_loss).backward(retain_graph=True)
        self.model.optimizer.step()

        # Backpropagation for Self-Supervised loss
        self.model.optimizer.zero_grad()
        ss_loss.backward()
        self.model.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': values.mean().item()
        }
    
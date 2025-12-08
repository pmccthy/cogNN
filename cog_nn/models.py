"""
Model definitions for reinforcement learning and self-supervised learning.
"""

from torch import nn
import torch.nn.functional as F
from torch.nn import RNN, Linear


class Actor(nn.Module):
    """
    Actor network which receives state as input and outputs a policy (distribution over actions).
    """
    def __init__(self, state_size, action_size, hidden_size=128, hook_fn=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)
        if hook_fn is not None:
            self.fc1.register_forward_hook(hook_fn)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        action_probs = self.softmax(x)
        return action_probs


class Critic(nn.Module):
    """
    Critic network which receives state as input and outputs a state value estimate.
    """
    def __init__(self, state_size, hidden_size=128, hook_fn=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        if hook_fn is not None:
            self.fc1.register_forward_hook(hook_fn)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value


class RNNReadout(nn.Module):
    """
    RNN with linear readout for sequence prediction (self-supervised learning).
    TODO: rethink this - may make mroe sense to just have linear readout from last hidden state
    """
    def __init__(self, input_size, hidden_size, output_size, hook_fn=None):
        super(RNNReadout, self).__init__()
        self.rnn = RNN(input_size=input_size,
                       hidden_size=hidden_size,
                       batch_first=True)  # expects dims (batch, seq, feature)
        self.fc = Linear(hidden_size, output_size)
        if hook_fn is not None:
            self.rnn.register_forward_hook(hook_fn)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # ignore hidden state
        out = self.fc(rnn_out)    # apply linear readout to each time step
        return out, rnn_out


class ActorCriticRNN(nn.Module):
    """
    Actor-Critic network with shared RNN backbone (for metalearning).
    
    The RNN processes state sequences, and separate heads output:
    - Actor head: action probabilities
    - Critic head: state value estimates
    """
    def __init__(self, state_size, action_size, rnn_hidden_size=128, 
                 actor_hidden_size=64, critic_hidden_size=64, hook_fn=None):
        super(ActorCriticRNN, self).__init__()
        
        # Shared RNN backbone
        self.rnn = RNN(input_size=state_size,
                      hidden_size=rnn_hidden_size,
                      batch_first=True)
        
        # Actor head
        self.actor_fc1 = nn.Linear(rnn_hidden_size, actor_hidden_size)
        self.actor_fc2 = nn.Linear(actor_hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Critic head
        self.critic_fc1 = nn.Linear(rnn_hidden_size, critic_hidden_size)
        self.critic_fc2 = nn.Linear(critic_hidden_size, 1)
        
        if hook_fn is not None:
            self.rnn.register_forward_hook(hook_fn)
    
    def forward(self, state_sequence):
        """
        Forward pass through RNN and both heads.
        
        Args:
            state_sequence: (batch_size, seq_len, state_size) or (seq_len, state_size)
            
        Returns:
            action_probs: (batch_size, seq_len, action_size) or (seq_len, action_size)
            values: (batch_size, seq_len, 1) or (seq_len, 1)
            rnn_out: RNN hidden states
        """
        # Ensure batch dimension
        if state_sequence.dim() == 2:
            state_sequence = state_sequence.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # RNN forward
        rnn_out, _ = self.rnn(state_sequence)  # (batch, seq_len, rnn_hidden_size)
        
        # Actor head
        actor_x = F.relu(self.actor_fc1(rnn_out))
        actor_logits = self.actor_fc2(actor_x)
        action_probs = self.softmax(actor_logits)
        
        # Critic head
        critic_x = F.relu(self.critic_fc1(rnn_out))
        values = self.critic_fc2(critic_x)
        
        if squeeze_batch:
            action_probs = action_probs.squeeze(0)
            values = values.squeeze(0)
            rnn_out = rnn_out.squeeze(0)
        
        return action_probs, values, rnn_out
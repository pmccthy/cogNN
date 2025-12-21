"""
Model definitions for reinforcement learning and self-supervised learning.
"""

from torch import nn, cat
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


class RNNActorCritic(nn.Module):
    """
    Combined Actor-Critic model with shared RNN backbone, with RNN receiving previous action and reward as input, as well as current state.
    This is designed for meta-RL, based on the architecture in Wang et al. (2016).
    """
    def __init__(self, state_size, action_size, hidden_size=128, hook_fn=None):
        super(RNNActorCritic, self).__init__()
        self.rnn = RNN(input_size=state_size + action_size + 1,  # +1 for reward
                       hidden_size=hidden_size,
                       batch_first=True)
        self.actor_fc = nn.Linear(hidden_size, action_size)
        self.critic_fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        if hook_fn is not None:
            self.rnn.register_forward_hook(hook_fn)

    def forward(self, state, prev_action, prev_reward, hidden_state=None, return_rnn_out=False):
        # Concatenate state, previous action (one-hot), and previous reward
        x = cat([state, prev_action, prev_reward.unsqueeze(-1)], dim=-1).unsqueeze(1)  # add seq dim
        rnn_out, hidden_state = self.rnn(x, hidden_state)  # rnn_out shape: (batch, seq=1, hidden)
        rnn_out = rnn_out.squeeze(1)  # remove seq dim

        # Actor head
        action_logits = self.actor_fc(rnn_out)
        action_probs = self.softmax(action_logits)

        # Critic head
        value = self.critic_fc(rnn_out)

        if return_rnn_out:
            return action_probs, value, hidden_state, rnn_out
        return action_probs, value, hidden_state
    
class SelfSupervisedRNNActorCritic(nn.Module):
    """
    Combined Actor-Critic model with shared RNN backbone and self-supervised readout, with RNN receiving current state and previous action as input, and predicting next state.
    This is based on the architecture in in Blanco-Pozo et al. (2024).
    """
    def __init__(self, state_size, action_size, hidden_size=128, hook_fn=None):
        super(SelfSupervisedRNNActorCritic, self).__init__()
        self.rnn = RNN(input_size=state_size + action_size,  # +1 for reward
                       hidden_size=hidden_size,
                       batch_first=True)
        self.actor_fc = nn.Linear(hidden_size, action_size)
        self.critic_fc = nn.Linear(hidden_size, 1)
        self.ss_fc = nn.Linear(hidden_size, state_size)  # self-supervised readout to predict next state
        self.softmax = nn.Softmax(dim=-1)
        if hook_fn is not None:
            self.rnn.register_forward_hook(hook_fn)

    def forward(self, state, prev_action, hidden_state=None):
        # Concatenate state and previous action (one-hot)
        x = cat([state, prev_action], dim=-1).unsqueeze(1)  # add seq dim
        rnn_out, hidden_state = self.rnn(x, hidden_state)  # rnn_out shape: (batch, seq=1, hidden)
        rnn_out = rnn_out.squeeze(1)  # remove seq dim

        # Actor head
        action_logits = self.actor_fc(rnn_out)
        action_probs = self.softmax(action_logits)

        # Critic head
        value = self.critic_fc(rnn_out)

        # Self-supervised readout
        ss_output = self.ss_fc(rnn_out)

        return action_probs, value, ss_output, hidden_state
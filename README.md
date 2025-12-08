# cogNN

Neural network models for solving cognitive tasks through reinforcement learning and self-supervised learning.

Repository: https://github.com/pmccthy/cogNN

This repository contains code for training reinforcement learning agents on reversal learning tasks, including implementations of Actor-Critic (A2C) models and hybrid self-supervised learning (SSL) + RL models.

## Repository Structure

```
new_repo/
├── tasks/
│   ├── task_generation.ipynb          # Notebook for generating task sequences
│   └── reversal_envs.py               # Gymnasium environments (AB, ABC)
├── models.py                           # Model definitions:
│                                      # - Actor, Critic (A2C)
│                                      # - RNNReadout (SSL)
│                                      # - ActorCriticRNN (metalearning)
│                                      # - ConvAutoencoder (image reconstruction)
│                                      # - MLPAutoencoder (vector reconstruction)
│                                      # - HybridSSLRNNA2C (SSL + RL)
├── agents.py                           # Agent classes (A2CAgent)
├── train_a2c_reversal_abc.ipynb       # Train A2C on reversal ABC task
├── train_hybrid_ssl_rnn_a2c.ipynb     # Train hybrid SSL-RNN-A2C model
└── README.md                           # This file
```

## Tasks

### Reversal AB Task
- Two stimuli: A and B
- Reward contingencies can reverse (A rewarded → B rewarded)
- Single-timestep version

### Reversal ABC Task (Sandra's Task)
- Three stimuli: A, B, and C
- A and B: Deterministic reward (contingencies can reverse)
- C: Random reward (50% probability, doesn't reverse)
- Single-timestep version

## Models

### Actor-Critic Models
- **Actor**: Policy network for action selection
- **Critic**: Value network for state value estimation
- **ActorCriticRNN**: Shared RNN backbone with Actor/Critic heads (for metalearning)

### Self-Supervised Learning Models
- **RNNReadout**: RNN with linear readout for next-state prediction
- **ConvAutoencoder**: Convolutional autoencoder for image reconstruction
- **MLPAutoencoder**: MLP autoencoder for vector/feature reconstruction

### Hybrid Models
- **HybridSSLRNNA2C**: Combines SSL (next-state prediction) and RL (A2C)
  - RNN processes sequences and learns temporal structure
  - RNN hidden states feed into Actor and Critic heads
  - Can be trained with combined SSL + RL losses

## Agents

### A2CAgent
- Implements Advantage Actor-Critic (A2C) algorithm
- Wraps Actor and Critic networks
- Handles action selection, value estimation, and policy updates

## Usage

### Training A2C on Reversal ABC Task

```python
# See train_a2c_reversal_abc.ipynb
# Tracks and visualizes:
# - Lick probability for each stimulus (A, B, C)
# - Value estimates for each stimulus
# - Performance before and after reversal
```

### Training Hybrid SSL-RNN-A2C Model

```python
# See train_hybrid_ssl_rnn_a2c.ipynb
# Combines:
# - SSL loss: next-state prediction
# - RL loss: A2C policy and value updates
```

### Generating Task Sequences

```python
# See tasks/task_generation.ipynb
# Generates reversal_ab and reversal_abc task sequences
# Saves to pickle files
```

## Environment API

All environments follow the Gymnasium API:

```python
from tasks.reversal_envs import ReversalABCEnv

env = ReversalABCEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- scikit-learn (for task generation)

## Notes

- All tasks use single-timestep versions (not multitimestep)
- C stimulus in ABC task has random 50% reward probability
- Environments can load from pickle files or generate default sequences
- Models support hook functions for activation tracking





# cogNN

PyTorch RNN and feedforward actor–critic models for reversal-learning and meta-reinforcement learning (meta-RL) on structured cognitive tasks. Implements Gymnasium environments, A2C/meta-A2C agents, and a suite of decoding and visualisation tools for analysing learned representations.

---

## Overview

This project investigates how recurrent neural networks (RNNs) learn and represent task structure during probabilistic reversal learning. Models range from simple feedforward actor–critic networks trained with A2C to meta-RL RNNs that adapt within a single forward pass (after Wang et al., 2016). Variants include context-input networks with partial readout, and self-supervised auxiliary prediction heads (after Blanco-Pozo et al., 2024).

The companion analysis pipeline (`cog_nn/analysis_utils.py`) supports plasticity-switch experiments on ACE/ABEF task schedules, including PCA of hidden states and time-resolved logistic decoding of stimulus value and contingency.

---

## Repository Structure

```
cogNN/
├── cog_nn/                     # Core library
│   ├── models.py               # PyTorch model definitions
│   ├── agents.py               # RL agent classes (A2C, meta-A2C, tabular)
│   ├── tasks/
│   │   ├── reversal_envs.py    # Gymnasium reversal environments (AB, ABC, ABCDEF)
│   │   └── generate_*.py       # Task sequence generators (pickled schedules)
│   ├── analysis_utils.py       # ACE/ABEF plasticity-switch analysis pipeline
│   ├── utils.py                # Activation/gradient hooks, helpers
│   └── plot_style.py           # Global Matplotlib style
├── experiment_nbs/             # Jupyter notebooks (training, JEPA, analysis)
├── experiment_scripts/         # Batch training runs, grid searches, plot regeneration
├── task_data/                  # Pickled task sequences (reproducible schedules)
├── results/                    # Experiment outputs (configs, metrics, figures)
├── environment.yml             # Conda environment
└── setup.py                    # Package install
```

---

## Models

| Class | Description |
|---|---|
| `Actor` / `Critic` | Two-layer MLP policy and value heads |
| `RNNActorCritic` | Shared RNN with previous action and reward as inputs (Wang et al., 2016 meta-RL architecture) |
| `RNNActorCriticPartialReadout` | RNN where only a subset of hidden units project to the actor/critic — separates recurrent dynamics from readout |
| `RNNActorCriticPartialReadoutContextInput` | Partial-readout RNN with explicit context (contingency) input instead of action/reward history |
| `SelfSupervisedRNNActorCritic` | RNN with an auxiliary next-state prediction head (Blanco-Pozo et al., 2024) |

---

## Agents

| Class | Description |
|---|---|
| `A2CAgent` | Standard advantage actor–critic (TD target, MSE critic loss) |
| `A2CAgentAsymmetricLR` | A2C with asymmetric scaling for positive vs negative advantages |
| `MetaA2CAgent` | Wraps `RNNActorCritic`; maintains hidden state across episode steps; optional auxiliary stimulus/reward prediction heads |
| `MetaA2CContextAgent` | Wraps `RNNActorCriticPartialReadoutContextInput`; accepts context tensor at each step |
| `SelfSupervisedRNNA2CAgent` | A2C + MSE next-state prediction loss in a single optimisation step |
| `SelfSupervisedRNNA2CAgentSeparateTraining` | Same, but runs two separate optimisation steps (actor–critic and SSL) |
| `TabularACAgent`, `TabularTDLearner`, `QLearner`, etc. | Discrete-state tabular baselines |

---

## Environments

All environments are [Gymnasium](https://gymnasium.farama.org/)-compatible with one-hot state observations and discrete actions (lick / no-lick).

| Class | Description |
|---|---|
| `ReversalABEnv` / `ReversalABCEnv` | Single-timestep reversal with two or three stimuli |
| `ReversalABCMultiTimestepEnv` | Multi-timestep trials with stimulus, ITI, and reward windows |
| `ReversalABCDEFMultiTimestepEnv` | Six-stimulus (A–F) multi-timestep environment; supports partial and full reversals; used in plasticity-switch and meta-RL experiments |

Task sequences are pre-generated as `.pkl` files under `task_data/` for reproducibility.

---

## Installation

**Requirements:** macOS or Linux, [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cogNN.git
cd cogNN

# 2. Create the Conda environment
conda env create -f environment.yml
conda activate cog_nn

# 3. Install the package in editable mode
pip install -e .
```

Key dependencies: Python 3.11, PyTorch 2.8, Gymnasium 1.2, scikit-learn 1.7, Matplotlib 3.10, neurogym 2.2.

---

## Usage

### Training a meta-RL agent on a reversal task

```python
from cog_nn.tasks.reversal_envs import ReversalABCDEFMultiTimestepEnv
from cog_nn.models import RNNActorCriticPartialReadoutContextInput
from cog_nn.agents import MetaA2CContextAgent

env = ReversalABCDEFMultiTimestepEnv(task_data_path="task_data/abcdef_train.pkl")
model = RNNActorCriticPartialReadoutContextInput(
    input_size=env.observation_space.n,
    hidden_size=256,
    n_actions=env.action_space.n,
    context_size=env.n_contexts,
)
agent = MetaA2CContextAgent(model, lr=1e-3)

obs, info = env.reset()
for _ in range(1000):
    action, log_prob, value = agent.select_action(obs, context=info["context"])
    obs, reward, done, _, info = env.step(action)
    agent.store(log_prob, value, reward)
    if done:
        agent.update()
        obs, info = env.reset()
```

### Running a batch experiment

```bash
python experiment_scripts/run_ace_plasticity_multirun_v2.py
```

### Grid search

```bash
python experiment_scripts/meta_ac_reversal_abc_gridsearch.py
python experiment_scripts/analyse_gridsearch_results.py
```

See `experiment_scripts/RUN_GRIDSEARCH.md` for full documentation.

---

## Analysis

`cog_nn/analysis_utils.py` provides a shared pipeline for the ACE/ABEF **plasticity-switch** experiments:

- **PCA** of RNN hidden states across trial phases
- **Time-resolved logistic decoding** of stimulus value and task contingency
- Cross-condition decoding (train on one context, test on another)
- Aggregation and paper-style figure generation

Results are written to timestamped subdirectories under `results/`.

---

## Notebooks

Exploratory and analysis notebooks live under `experiment_nbs/`, including:

- A2C and meta-A2C training on reversal tasks
- JEPA / self-supervised learning trials
- Value-subspace and decoder visualisation
- Dated plasticity-switch analysis notebooks

---

## References

- Wang, J. X. et al. (2016). Learning to reinforcement learn. *arXiv:1611.05763*
- Blanco-Pozo, M. et al. (2024). Dopamine-independent state-space representations in prefrontal cortex. *Nature Neuroscience*

# PPO From Scratch

A minimal implementation of **Proximal Policy Optimization (PPO)** from scratch using PyTorch and Gymnasium.

This project was built to understand and implement the core PPO algorithm rather than relying on an existing RL library.

## Overview

The implementation contains:

* Actor-Critic architecture
* Rollout collection
* Action sampling using categorical distributions
* Generalized Advantage Estimation (GAE)
* Return calculation
* PPO probability ratios
* Clipped PPO objective
* Actor loss
* Critic loss
* Separate Actor and Critic optimizers
* Training loop

The environment used for the proof of concept is **CartPole-v1**.

## Project Structure

```text
ppo-from-scratch/
├── main.py       # Training loop and orchestration
├── agent.py      # Actor and Critic networks
├── rollout.py    # Experience collection
├── gae.py        # GAE and return calculation
├── ppo.py        # PPO objective and losses
├── README.md
├── requirements.txt
└── .gitignore
```

## Architecture

### Actor

The Actor receives the CartPole observation and outputs action logits.

```text
Observation (4)
      ↓
Linear(4 → 64)
      ↓
ReLU
      ↓
Linear(64 → 64)
      ↓
ReLU
      ↓
Linear(64 → 2)
      ↓
Action logits
      ↓
Categorical distribution
      ↓
Action
```

### Critic

The Critic estimates the value of the current state.

```text
Observation (4)
      ↓
Linear(4 → 64)
      ↓
ReLU
      ↓
Linear(64 → 64)
      ↓
ReLU
      ↓
Linear(64 → 1)
      ↓
State value V(s)
```

## PPO Pipeline

The training process follows:

```text
Environment
    ↓
Collect rollout
    ↓
Store observations, actions, rewards,
dones, old log probabilities and values
    ↓
Calculate GAE
    ↓
Calculate returns
    ↓
Evaluate actions under current policy
    ↓
Calculate probability ratio
    ↓
Apply PPO clipping
    ↓
Calculate Actor loss
    ↓
Calculate Critic loss
    ↓
Backpropagation
    ↓
Update Actor + Critic
    ↓
Repeat
```

## Key PPO Components

### Probability Ratio

The probability ratio compares the current policy against the policy that generated the rollout:

```text
ratio = exp(new_log_prob - old_log_prob)
```

### GAE

Generalized Advantage Estimation is used to estimate how much better or worse an action performed compared to the Critic's expectation.

```text
A_t = δ_t + γλ A_(t+1)
```

where:

```text
δ_t = r_t + γV(s_(t+1)) - V(s_t)
```

Episode boundaries are handled so that value estimates are not bootstrapped across terminated episodes.

### PPO Clipping

The probability ratio is clipped around the old policy:

```text
L1 = ratio × advantage

L2 = clip(ratio, 1 - ε, 1 + ε) × advantage

L_clip = min(L1, L2)
```

The Actor minimizes the negative PPO objective.

The Critic minimizes the squared error between predicted values and calculated returns.

## Environment

This project uses:

**CartPole-v1**

Observation:

```text
[cart position,
 cart velocity,
 pole angle,
 pole angular velocity]
```

Actions:

```text
0 → Move left
1 → Move right
```

The reward is `+1` for every timestep the pole remains balanced.

## Results

The implementation successfully learned CartPole.

Example training progression:

```text
Iteration 0:   reward = 28
Iteration 100: reward = 22
Iteration 300: reward = 40
Iteration 500: reward = 84
Iteration 550: reward = 141
Iteration 650: reward = 217
Iteration 700: reward = 238
Iteration 800: reward = 341
Iteration 900: reward = 256
```

The training is intentionally minimal and somewhat noisy because this project focuses on understanding the core PPO implementation rather than maximizing CartPole performance.

## Requirements

* Python 3.x
* PyTorch
* Gymnasium
* Gymnasium Classic Control

Install dependencies with:

```bash
pip install torch gymnasium[classic-control]
```

## Running

Run the training loop with:

```bash
python main.py
```

The program will train the Actor-Critic agent on CartPole and print rollout rewards during training.

## What I Learned

This project was primarily built as a learning implementation.

Topics covered:

* Policy networks
* Value networks
* Log probabilities
* Categorical action distributions
* Rollouts
* TD errors
* GAE
* Returns
* Policy ratios
* PPO clipping
* Actor-Critic optimization
* Reinforcement learning training loops

## Limitations

This is a learning implementation, not a fully optimized PPO implementation.

Currently it does not include several common PPO improvements such as:

* Mini-batch updates
* Multiple optimization epochs per rollout
* Advantage normalization
* Entropy bonus
* Gradient clipping
* Large parallelized rollout collection
* Extensive evaluation tooling
* Hyperparameter optimization

These can be added later if needed.

---

**Built from scratch with PyTorch.**

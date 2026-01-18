# Pensieve PPO

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A user-friendly PyTorch implementation of **Pensieve** [1], a neural adaptive video streaming system. This implementation uses **Proximal Policy Optimization (PPO)** instead of the original A3C algorithm, achieving improved training stability and comparable performance.

## Features

- **Modern PyTorch Implementation**: Clean, modular codebase using PyTorch 2.0+
- **Gymnasium Environment**: Standard RL environment interface for easy integration
- **Parallel Training**: Multi-worker distributed training support
- **TensorBoard Integration**: Real-time training visualization
- **Extensible Architecture**: Easy to add new agents and environments

## Installation

### From Source

```bash
git clone https://github.com/yindaheng98/Pensieve-PPO.git
cd Pensieve-PPO
pip install -e . # or
pip install --target . --upgrade . --no-deps
```

### Dependencies

```bash
pip install torch numpy gymnasium tensorboard
```

## Quick Start

### Using the Gymnasium Environment

```python
from pensieve_ppo.gym import ABREnv
from pensieve_ppo.defaults import create_env_with_default

# Create environment with default Pensieve parameters
env = create_env_with_default(train=True)

# Standard Gymnasium API
obs, info = env.reset()
while True:
    action = env.action_space.sample()  # or use your agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Training a PPO Agent

```bash
# Train with default settings
python -m pensieve_ppo.train

# Train with custom parameters
python -m pensieve_ppo.train \
    --parallel-workers 16 \
    --train-epochs 500000 \
    --model-save-interval 300 \
    --output-dir ./ppo
```

### Testing a Trained Model

```bash
# Test with a trained model
python -m pensieve_ppo.test --model-path ./ppo/model_ep300.pt

# Test with custom trace folder
python -m pensieve_ppo.test \
    --model-path ./ppo/model_ep300.pt \
    --test-trace-folder ./src/test/
```

## Package Structure

```
pensieve_ppo/
├── core/                  # Core simulation components
│   ├── trace/             # Network trace handling (bandwidth, latency)
│   ├── video/             # Video processing (chunk sizes, bitrates, playback)
│   └── simulator/         # Combines trace & video into ABR simulator
├── gym/                   # Gymnasium environment wrapper
│   └── env.py             # ABREnv - standard RL environment interface
├── agent/                 # RL agent implementations
│   ├── abc.py             # Abstract base classes
│   ├── registry.py        # Agent factory and registration
│   ├── trainer.py         # Distributed training framework
│   └── rl/
│       ├── observer.py    # State observation and reward calculation
│       └── ppo/           # PPO agent implementation
├── defaults.py            # Default parameters and factory functions
├── train.py               # Training script
└── test.py                # Testing script
```

## API Reference

### Creating Environment and Agent

```python
from pensieve_ppo.defaults import (
    create_env_with_default,
    create_env_agent_with_default,
    create_env_agent_factory_with_default,
)

# Create just the environment
env = create_env_with_default(
    levels_quality=[300., 750., 1200., 1850., 2850., 4300.],  # Kbps
    trace_folder='./src/train/',
    train=True,
)

# Create compatible env and agent pair
env, agent = create_env_agent_with_default(
    agent_name='ppo',
    model_path='./ppo/model.pt',  # Optional: load pretrained weights
    device='cuda',
)

# Create factories for distributed training
env_factory, agent_factory = create_env_agent_factory_with_default(
    agent_name='ppo',
    train=True,
)
```

### Using the Agent

```python
from pensieve_ppo.agent import create_agent

# Create agent directly
agent = create_agent(
    name='ppo',
    state_dim=(6, 8),  # (S_INFO, S_LEN)
    action_dim=6,       # Number of bitrate levels
    device='cuda',
    learning_rate=1e-4,
    gamma=0.99,
)

# Predict action
state = env.reset()[0]
action, action_prob = agent.select_action(state)

# Train on batch
metrics = agent.train(s_batch, a_batch, p_batch, v_batch, epoch)
```

### Environment Details

**Observation Space** (`Box(6, 8)`):
| Index | Description |
|-------|-------------|
| 0 | Last quality normalized by max quality |
| 1 | Buffer size normalized by buffer_norm_factor |
| 2 | Throughput (chunk_size / delay) in Mbps |
| 3 | Delay normalized by buffer_norm_factor |
| 4 | Next chunk sizes at each bitrate level (MB) |
| 5 | Remaining chunks normalized by total |

**Action Space** (`Discrete(6)`): Select bitrate level (0-5)

**Reward**: `quality - 4.3 * rebuffer - 1.0 * |quality_change|`

## Command Line Options

### Training (`pensieve_ppo.train`)

```
--train-trace-folder    Training trace folder (default: ./src/train/)
--output-dir            Output directory (default: ./ppo)
--parallel-workers      Number of parallel workers (default: 16)
--steps-per-epoch       Steps per epoch per worker (default: 1000)
--train-epochs          Total training epochs (default: 500000)
--model-save-interval   Model checkpoint interval (default: 300)
--pretrained-model-path Resume from pretrained model
```

### Testing (`pensieve_ppo.test`)

```
--test-trace-folder     Test trace folder (default: ./src/test/)
--model-path            Path to trained model weights
--test-log-file-prefix  Prefix for test log files
```

### Common Options

```
--agent-name            RL algorithm (default: ppo)
--device                PyTorch device (cuda/cpu)
--levels-quality        Bitrate levels in Kbps
--state-history-len     State history length (default: 8)
--random-seed           Random seed (default: 42)
-o, --agent-options     Extra agent kwargs (e.g., learning_rate=1e-4)
-e, --env-options       Extra env kwargs (e.g., rebuf_penalty=4.3)
```

## TensorBoard Monitoring

Monitor training in real-time:

```bash
tensorboard --logdir=./ppo
```

---

# Original README

### Updates

**Jan. 18, 2025:** We removed the rate-based method and added NetLLM [4].

**May. 4, 2024:** We removed the Elastic, revised  BOLA, and add new baseline Comyco [3] and Genet [2].

**Jan. 26, 2024:** We are excited to announce significant updates to Pensieve-PPO! We have replaced TensorFlow with PyTorch, and we have achieved a similar training speed while training models that rival in performance.

*For the TensorFlow version, please check [Pensieve-PPO TF Branch](https://github.com/godka/Pensieve-PPO/tree/master).*

**Dec. 28, 2021:** In a previous update, we enhanced Pensieve-PPO with several state-of-the-art technologies, including Dual-Clip PPO and adaptive entropy decay.

## About Pensieve-PPO

Pensieve-PPO is a user-friendly PyTorch implementation of Pensieve [1], a neural adaptive video streaming system. Unlike A3C, we utilize the Proximal Policy Optimization (PPO) algorithm for training.

This stable version of Pensieve-PPO includes both the training and test datasets.

You can run the repository by executing the following command:

```
python train.py
```

The results will be evaluated on the test set (from HSDPA) every 300 epochs.

## Tensorboard Integration

To monitor the training process in real time, you can leverage Tensorboard. Simply run the following command:

```
tensorboard --logdir=./
```

## Pretrained Model

We have also added a pretrained model, which can be found at [this link](https://github.com/godka/Pensieve-PPO/tree/torch/src/pretrain). This model demonstrates a substantial improvement of 7.03% (from 0.924 to 0.989) in average Quality of Experience (QoE) compared to the original Pensieve model [1]. For a more detailed performance analysis, refer to the figures below:

<p align="center">
    <img src="src/baselines-br.png" width="50%"><img src="src/baselines-bs.png" width="50%">
</p>
<p align="center">
    <img src="src/baselines-qoe.png" width="100%">
</p>
If you have any questions or require further assistance, please don't hesitate to reach out.

## Additional Reinforcement Learning Algorithms

For more implementations of reinforcement learning algorithms, please visit the following branches:

- DQN: [Pensieve-PPO DQN Branch](https://github.com/godka/Pensieve-PPO/tree/dqn)
- SAC: [Pensieve-PPO SAC Branch](https://github.com/godka/Pensieve-PPO/tree/SAC) or [Pensieve-SAC Repository](https://github.com/godka/Pensieve-SAC)

[1] Mao H, Netravali R, Alizadeh M. Neural adaptive video streaming with Pensieve[C]//Proceedings of the Conference of the ACM Special Interest Group on Data Communication. ACM, 2017: 197-210.

[2] Xia, Zhengxu, et al. "Genet: automatic curriculum generation for learning adaptation in networking." Proceedings of the ACM SIGCOMM 2022 Conference. 2022.

[3] Huang, Tianchi, et al. "Comyco: Quality-aware adaptive video streaming via imitation learning." Proceedings of the 27th ACM international conference on multimedia. 2019.

[4] Wu, Duo, et al. "Netllm: Adapting large language models for networking." Proceedings of the ACM SIGCOMM 2024 Conference. 2024.

* We use the following command to test the *entire traces* in the dataset.

```
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --trace-num -1 --model-dir  data/ft_plms/try_llama2_7b
```

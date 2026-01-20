"""DQN (Deep Q-Network) algorithm implementation.

This module provides the DQN agent and its neural network model,
following the architecture from the original Pensieve-PPO DQN implementation.

The DQN agent uses:
- Double DQN for reduced overestimation bias
- Experience replay buffer for stable learning
- Soft target network updates

Reference:
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/test_dqn.py
"""

from .agent import DQNAgent
from .model import QNetwork
from ...registry import register_agent, register_env
from ..env import create_rl_env

# Register DQN agent
register_agent("dqn", DQNAgent)

# Register DQN environment (same as PPO/A3C, uses RLABRStateObserver)
register_env("dqn", create_rl_env)

__all__ = [
    'DQNAgent',
    'QNetwork',
]

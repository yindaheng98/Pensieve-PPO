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
from ...registry import register
from ..observer import RLABRStateObserver

# Register DQN agent
register("dqn", DQNAgent, RLABRStateObserver)

__all__ = [
    'DQNAgent',
    'QNetwork',
]

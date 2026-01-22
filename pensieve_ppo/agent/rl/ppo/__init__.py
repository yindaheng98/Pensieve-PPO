"""PPO (Proximal Policy Optimization) algorithm implementation.

This module provides the PPO agent and its neural network models.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from .agent import PPOAgent
from .model import Actor, Critic
from ...registry import register
from ..observer import RLABRStateObserver

# Register PPO agent
register("ppo", PPOAgent, RLABRStateObserver)

__all__ = [
    'PPOAgent',
    'Actor',
    'Critic',
]

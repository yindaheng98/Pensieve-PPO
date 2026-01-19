"""PPO (Proximal Policy Optimization) algorithm implementation.

This module provides the PPO agent and its neural network models.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from .agent import PPOAgent
from .model import Actor, Critic
from ...registry import register_agent, register_env
from ..env import create_rl_env

# Register PPO agent
register_agent("ppo", PPOAgent)

# Register PPO environment
register_env("ppo", create_rl_env)

__all__ = [
    'PPOAgent',
    'Actor',
    'Critic',
]

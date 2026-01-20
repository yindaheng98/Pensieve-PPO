"""A3C (Asynchronous Advantage Actor-Critic) algorithm implementation.

This module provides the A3C agent and its neural network models,
following the architecture from the original Pensieve implementation.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py
"""

from .agent import A3CAgent, compute_entropy, discount
from .model import Actor, Critic
from ...registry import register_agent, register_env
from ..env import create_rl_env

# Register A3C agent
register_agent("a3c", A3CAgent)

# Register A3C environment (same as PPO, uses RLABRStateObserver)
register_env("a3c", create_rl_env)

__all__ = [
    'A3CAgent',
    'Actor',
    'Critic',
    'compute_entropy',
    'discount',
]

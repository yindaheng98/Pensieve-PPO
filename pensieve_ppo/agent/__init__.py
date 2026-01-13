"""Agent module for reinforcement learning algorithms.

This module provides the base classes and implementations for RL agents.
"""

from .abc import AbstractAgent
from .trainer import Trainer
from .registry import create_agent, register_agent, get_available_agents

# Import agent implementations to trigger registration
from . import ppo  # noqa: F401

__all__ = [
    'AbstractAgent',
    'Trainer',
    'create_agent',
    'register_agent',
    'get_available_agents',
    'ppo',
]

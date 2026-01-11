"""Agent module for reinforcement learning algorithms.

This module provides the base classes and implementations for RL agents.
"""

from .abc import AbstractAgent
from .trainer import Trainer

__all__ = [
    'AbstractAgent',
    'Trainer',
]

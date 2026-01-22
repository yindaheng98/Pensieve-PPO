"""Agent module for Pensieve PPO.

This module provides the base classes and implementations for RL agents.
"""

from .abc import AbstractAgent
from .trainable import AbstractTrainableAgent, Step, TrainingBatch
from .trainer import Trainer, EpochEndCallback, SaveModelCallback
from .imitate import ImitationTrainer
from .registry import create_agent, register, get_available_agents, get_available_trainable_agents, create_env

# Import agent implementations to trigger registration
from . import rl  # noqa: F401
from . import bba  # noqa: F401
from . import mpc  # noqa: F401

__all__ = [
    'AbstractAgent',
    'Step',
    'TrainingBatch',
    'AbstractTrainableAgent',
    'Trainer',
    'ImitationTrainer',
    'EpochEndCallback',
    'SaveModelCallback',
    'create_agent',
    'register',
    'get_available_agents',
    'get_available_trainable_agents',
    'create_env',
    'rl',
    'bba',
    'mpc',
]

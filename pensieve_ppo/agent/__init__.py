"""Agent module for Pensieve PPO.

This module provides the base classes and implementations for RL agents.
"""

from .abc import AbstractAgent, ActionDecision
from .trainable import AbstractTrainableAgent, Step, TrainingBatch, TrainBatchInfo
from .trainer import Trainer, EpochEndCallback, SaveModelCallback
from .imitate import ImitationTrainer
from .registry import create_agent, register, get_available_agents, get_available_trainable_agents, create_env, create_imitation_env

__all__ = [
    'AbstractAgent',
    'ActionDecision',
    'Step',
    'TrainingBatch',
    'TrainBatchInfo',
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
    'create_imitation_env',
]

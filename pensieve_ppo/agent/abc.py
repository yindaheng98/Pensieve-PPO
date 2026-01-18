"""Abstract base classes for agents.

This module provides the abstract base class hierarchy for all agents:
- AbstractAgent: Base class with select_action

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


@dataclass
class Step:
    """A single environment step.

    Attributes:
        observation: State observation.
        action: Action (one-hot encoded).
        action_prob: Action probability distribution.
        reward: Reward received.
    """
    observation: np.ndarray
    action: List[int]
    action_prob: List[float]
    reward: float


@dataclass
class TrainingBatch:
    """A batch of training data produced from a trajectory.

    Contains observations, actions, action probabilities, and computed value
    targets, ready to be used for training the agent.

    Attributes:
        s_batch: List of observations (states).
        a_batch: List of actions (one-hot encoded).
        p_batch: List of action probabilities.
        v_batch: List of computed value targets (returns).
    """
    s_batch: List[np.ndarray]
    a_batch: List[List[int]]
    p_batch: List[List[float]]
    v_batch: List[float]


class AbstractAgent(ABC):
    """Abstract base class for agents with action selection capability.

    This class defines the minimal interface for agents that can select
    actions from states.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Tuple[int, List[float]]:
        """Select an action for a given state.

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        pass

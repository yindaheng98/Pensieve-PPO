"""Abstract base classes for reinforcement learning agents.

This module provides the abstract base class hierarchy for all RL agents:
- AbstractAgent: Base class with predict and select_action

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch


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
    """Abstract base class for agents with prediction capability.

    This class defines the minimal interface for agents that can predict
    actions from states.

    Attributes:
        s_dim: State dimension as [num_features, sequence_length].
        a_dim: Action dimension (number of discrete actions).
        device: PyTorch device for computations.
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        device: Optional[torch.device] = None,
    ):
        """Initialize the abstract agent.

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Action dimension (number of discrete actions).
            device: PyTorch device for computations. If None, uses CPU.
        """
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.device = device if device is not None else torch.device('cpu')

    @abstractmethod
    def predict(self, state: np.ndarray) -> List[float]:
        """Predict action probabilities for a given state.

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).
                   The batch dimension will be added internally.

        Returns:
            Action probability distribution as a 1D list with length a_dim.
        """
        pass

    def select_action(self, state: np.ndarray) -> Tuple[int, List[float]]:
        """Select an action deterministically (greedy policy).

        This method selects the action with highest probability, without any
        exploration noise. Use this for testing/evaluation.

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        action_prob = self.predict(state)  # np.reshape(state, (1, S_INFO, S_LEN)) inside predict
        action = np.argmax(action_prob)
        return int(action), action_prob

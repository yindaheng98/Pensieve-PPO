"""Abstract base class for reinforcement learning agents.

This module provides the AbstractAgent base class that defines the interface
for all RL agents in this project. Specific algorithms (e.g., PPO, A2C)
should inherit from this class and implement the abstract methods.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class AbstractAgent(ABC):
    """Abstract base class for RL agents.

    This class defines the common interface that all RL agents must implement.
    It provides some common functionality and defines abstract methods for
    algorithm-specific operations.

    Attributes:
        s_dim: State dimension as [num_features, sequence_length].
        a_dim: Action dimension (number of discrete actions).
        device: PyTorch device for computations.
    """

    def __init__(
        self,
        state_dim: List[int],
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
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action probabilities for a given state.

        Args:
            state: Input state with shape (1, s_dim[0], s_dim[1]).

        Returns:
            Action probability distribution as numpy array with shape (a_dim,).
        """
        pass

    @abstractmethod
    def train(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        p_batch: np.ndarray,
        v_batch: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """Train the agent on a batch of experiences.

        Args:
            s_batch: Batch of states with shape (batch_size, s_dim[0], s_dim[1]).
            a_batch: Batch of actions (one-hot) with shape (batch_size, a_dim).
            p_batch: Batch of action probabilities with shape (batch_size, a_dim).
            v_batch: Batch of computed returns with shape (batch_size, 1).
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics (e.g., loss values).
        """
        pass

    @abstractmethod
    def compute_v(
        self,
        s_batch: List[np.ndarray],
        a_batch: List[np.ndarray],
        r_batch: List[float],
        terminal: bool,
    ) -> List[float]:
        """Compute value targets (returns) for a trajectory.

        Args:
            s_batch: List of states in the trajectory.
            a_batch: List of actions (one-hot) in the trajectory.
            r_batch: List of rewards in the trajectory.
            terminal: Whether the trajectory ended in a terminal state.

        Returns:
            List of computed returns for each timestep.
        """
        pass

    @abstractmethod
    def get_network_params(self) -> Any:
        """Get the current network parameters.

        Returns:
            Network parameters in a format suitable for set_network_params.
        """
        pass

    @abstractmethod
    def set_network_params(self, params: Any) -> None:
        """Set the network parameters.

        Args:
            params: Network parameters from get_network_params.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Args:
            path: Path to load the model from.
        """
        pass

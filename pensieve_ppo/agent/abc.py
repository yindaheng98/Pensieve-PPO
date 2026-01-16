"""Abstract base classes for ABR agents.

This module provides the abstract base classes for all agents:
- AbstractAgent: Minimal interface for any ABR agent (predict only)
- AbstractRLAgent: Extended interface for reinforcement learning agents

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, TYPE_CHECKING

import numpy as np
import torch

from ..gym.env import Observation


# Normalization constants
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L16
M_IN_K = 1000.0

# State dimensions (used by RL agents for state representation)
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L8
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past

# Normalization constants (used by RL agents for state computation)
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
BUFFER_NORM_FACTOR = 10.0

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class AbstractAgent(ABC):
    """Abstract base class for any ABR agent.

    This is the minimal interface that all ABR agents must implement.
    It only requires a predict method that takes an observation and returns an action.
    This allows for simple rule-based agents, heuristic agents, or RL agents to all
    share the same interface for evaluation.
    """

    @abstractmethod
    def predict(self, observation: Observation) -> int:
        """Predict an action given an observation.

        Args:
            observation: Raw observation from the environment.

        Returns:
            Action index (bitrate level to select).
        """
        pass


class AbstractRLAgent(AbstractAgent):
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

    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """Select an action using Gumbel-softmax sampling.

        This implements the action selection strategy used in the original
        Pensieve-PPO implementation.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L145-L150

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        action_prob = self.predict(
            np.reshape(state, (1, self.s_dim[0], self.s_dim[1])))

        # gumbel noise
        noise = np.random.gumbel(size=len(action_prob))
        action = np.argmax(np.log(action_prob) + noise)

        return int(action), action_prob

    def create_action_vector(self, action: int) -> np.ndarray:
        """Create a one-hot action vector.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L154-L155

        Args:
            action: Action index.

        Returns:
            One-hot encoded action vector with shape (a_dim,).
        """
        action_vec = np.zeros(self.a_dim)
        action_vec[action] = 1
        return action_vec

    def tensorboard_logging(self, writer: 'SummaryWriter', epoch: int) -> None:
        """Log metrics to TensorBoard.

        This method can be overridden by subclasses to log agent-specific metrics.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch.
        """
        pass

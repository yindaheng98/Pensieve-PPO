"""Abstract base class for reinforcement learning agents.

This module provides the AbstractAgent base class that defines the interface
for all RL agents in this project. Specific algorithms (e.g., PPO, A2C)
should inherit from this class and implement the abstract methods.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


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
    action: np.ndarray
    action_prob: np.ndarray
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
    a_batch: List[np.ndarray]
    p_batch: List[np.ndarray]
    v_batch: List[float]


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
            state: Input state with shape (s_dim[0], s_dim[1]).
                   The batch dimension will be added internally.

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
        action_prob = self.predict(state) # np.reshape(state, (1, S_INFO, S_LEN)) inside predict

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

    def produce_training_batch(
        self,
        trajectory: List[Step],
        done: bool,
    ) -> TrainingBatch:
        """Produce a training batch from a trajectory.

        Extracts observations, actions, rewards, and action probabilities from
        the trajectory steps, computes value targets, and returns a training
        batch ready for the training step.

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            Training batch with computed value targets.
        """
        # Extract data from steps
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L143
        s_batch = [step.observation for step in trajectory]
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L156-158
        a_batch = [step.action for step in trajectory]
        r_batch = [step.reward for step in trajectory]
        p_batch = [step.action_prob for step in trajectory]

        # Compute value targets
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L161
        v_batch = self.compute_v(s_batch, a_batch, r_batch, done)

        return TrainingBatch(
            s_batch=s_batch,
            a_batch=a_batch,
            p_batch=p_batch,
            v_batch=v_batch,
        )

    def train_batch(
        self,
        training_batches: List[TrainingBatch],
        epoch: int,
    ) -> Dict[str, float]:
        """Train on multiple training batches.

        Concatenates data from all training batches and performs a training step.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L102-L114

        Args:
            training_batches: List of training batches from workers.
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics.
        """
        s, a, p, v = [], [], [], []
        for batch in training_batches:
            s += batch.s_batch
            a += batch.a_batch
            p += batch.p_batch
            v += batch.v_batch

        s_batch = np.stack(s, axis=0)
        a_batch = np.vstack(a)
        p_batch = np.vstack(p)
        v_batch = np.vstack(v)

        return self.train(s_batch, a_batch, p_batch, v_batch, epoch)

    def tensorboard_logging(self, writer: 'SummaryWriter', epoch: int) -> None:
        """Log metrics to TensorBoard.

        This method can be overridden by subclasses to log agent-specific metrics.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch.
        """
        pass

"""Abstract base classes for trainable agents.

This module provides the abstract base class hierarchy for all trainable agents:
- Step: A single environment step data class
- TrainingBatch: A batch of training data
- AbstractTrainableAgent: Adds training infrastructure methods based on AbstractAgent

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from .abc import AbstractAgent
from ..gym import State


@dataclass
class Step:
    """A single environment step.

    Attributes:
        state: State observation.
        action: Action (one-hot encoded).
        action_prob: Action probability distribution.
        reward: Reward received.
    """
    state: State
    action: List[int]
    action_prob: List[float]
    reward: float
    step: int
    done: bool


class TrainingBatch:
    """Abstract base class for training batches.

    This is an abstract container for training data produced from a trajectory.
    Subclasses should define the specific data fields needed for their training
    algorithm.

    For RL algorithms, see RLTrainingBatch in abcrl.py which contains
    observations, actions, action probabilities, and computed value targets.
    """
    pass


class AbstractTrainableAgent(AbstractAgent):
    """Abstract base class for trainable agents.

    This class extends AbstractAgent with training infrastructure methods
    including model persistence, network parameter access, and training batch
    handling.

    Subclasses must implement the abstract methods for model saving/loading
    and network parameter access. The produce_training_batch and train_batch
    methods are abstract here but implemented in AbstractRLAgent.
    """

    @abstractmethod
    def select_action_for_training(self, state: State) -> Tuple[int, List[float]]:
        """Select an action using Gumbel-softmax sampling for exploration.

        This implements the action selection strategy used in the original
        Pensieve-PPO implementation, with Gumbel noise for exploration during
        training.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L145-L150

        Args:
            state: Input state.

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        pass

    @abstractmethod
    def produce_training_batch(
        self,
        trajectory: List[Step],
        done: bool,
    ) -> TrainingBatch:
        """Produce a training batch from a trajectory.

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            Training batch with computed value targets.
        """
        pass

    @abstractmethod
    def train_batch(
        self,
        training_batches: List[TrainingBatch],
        epoch: int,
    ) -> Dict[str, float]:
        """Train on multiple training batches.

        Args:
            training_batches: List of training batches from workers.
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics.
        """
        pass

    @abstractmethod
    def get_params(self) -> Any:
        """Get the current network parameters.

        Returns:
            Network parameters in a format suitable for set_params.
        """
        pass

    @abstractmethod
    def set_params(self, params: Any) -> None:
        """Set the network parameters.

        Args:
            params: Network parameters from get_params.
        """
        pass

    def save(self, path: str) -> None:
        """Save the model to a file.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L142-L144

        Args:
            path: Path to save the model.
        """
        torch.save(self.get_params(), path)

    def load(self, path: str) -> None:
        """Load the model from a file.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L137-L140

        Args:
            path: Path to load the model from.
        """
        self.set_params(torch.load(path))

    def tensorboard_logging(self, writer: 'SummaryWriter', epoch: int) -> None:
        """Log metrics to TensorBoard.

        This method can be overridden by subclasses to log agent-specific metrics.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch.
        """
        pass

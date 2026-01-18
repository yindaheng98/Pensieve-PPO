"""Abstract base classes for reinforcement learning agents.

This module provides the abstract base class hierarchy for all RL agents:
- AbstractAgent: Base class with predict and select_action
- AbstractTrainableAgent: Adds training infrastructure methods
- AbstractRLAgent: Adds RL-specific training methods (train, compute_v)

Specific algorithms (e.g., PPO, A2C) should inherit from AbstractRLAgent
and implement the abstract methods.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import abstractmethod
from typing import Any, Dict, Tuple, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from .abc import Step, TrainingBatch, AbstractAgent


class AbstractTrainableAgent(AbstractAgent):
    """Abstract base class for trainable agents.

    This class extends AbstractAgent with training infrastructure methods
    including model persistence, network parameter access, and training batch
    handling.

    Subclasses must implement the abstract methods for model saving/loading
    and network parameter access. The produce_training_batch and train_batch
    methods are abstract here but implemented in AbstractRLAgent.
    """

    def select_action_for_training(self, state: np.ndarray) -> Tuple[int, List[float]]:
        """Select an action using Gumbel-softmax sampling for exploration.

        This implements the action selection strategy used in the original
        Pensieve-PPO implementation, with Gumbel noise for exploration during
        training.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L145-L150

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        action_prob = self.predict(state)  # np.reshape(state, (1, S_INFO, S_LEN)) inside predict

        # gumbel noise for exploration
        noise = np.random.gumbel(size=len(action_prob))
        action = np.argmax(np.log(action_prob) + noise)

        return int(action), action_prob

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

    def tensorboard_logging(self, writer: 'SummaryWriter', epoch: int) -> None:
        """Log metrics to TensorBoard.

        This method can be overridden by subclasses to log agent-specific metrics.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch.
        """
        pass


class AbstractRLAgent(AbstractTrainableAgent):
    """Abstract base class for RL agents.

    This class extends AbstractTrainableAgent with reinforcement learning
    specific methods (train, compute_v) and provides implementations of
    produce_training_batch and train_batch using these RL methods.

    Specific RL algorithms (e.g., PPO, A2C) should inherit from this class
    and implement the abstract methods.
    """

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
        a_batch: List[List[int]],
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

"""Abstract base classes for reinforcement learning agents.

This module provides the abstract base class hierarchy for all RL agents:
- AbstractAgent: Base class with select_action
- AbstractTrainableAgent: Adds training infrastructure methods
- AbstractRLAgent: Adds RL-specific training methods (train, compute_v)
- RLTrainingBatch: Training batch with RL-specific data fields

Specific algorithms (e.g., PPO, A2C) should inherit from AbstractRLAgent
and implement the abstract methods.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


from ...core.video import VideoChunkRequest
from ...quality_ladder import DEFAULT_QUALITY, QualityLadderActionDecision, QualityLadderRequest
from ...agent.trainable import Step, TrainingBatch, TrainBatchInfo, AbstractTrainableAgent
from .observer import RLState


@dataclass(frozen=True)
class RLActionDecision(QualityLadderActionDecision):
    """Quality-ladder decision with index and probability metadata."""

    action_index: int
    action_prob: List[float]

    @classmethod
    def from_index(
        cls,
        action_index: int,
        action_prob: List[float],
    ) -> "RLActionDecision":
        """Build a decision from a quality level index."""
        return cls(
            action=QualityLadderRequest(action_index),
            action_index=action_index,
            action_prob=action_prob,
        )


@dataclass
class RLTrainingBatch(TrainingBatch):
    """A batch of training data for reinforcement learning.

    Contains observations, actions, action probabilities, and computed value
    targets, ready to be used for training RL agents.

    Attributes:
        s_batch: List of observations (states).
        a_batch: List of actions (one-hot encoded).
        p_batch: List of action probabilities.
        v_batch: List of computed value targets (returns).
    """
    s_batch: List[RLState]
    a_batch: List[List[int]]
    p_batch: List[List[float]]
    v_batch: List[float]


class AbstractRLAgent(AbstractTrainableAgent):
    """Abstract base class for RL agents.

    This class extends AbstractTrainableAgent with reinforcement learning
    specific methods (train, compute_v) and provides implementations of
    produce_training_batch and train_batch using these RL methods.

    Specific RL algorithms (e.g., PPO, A2C) should inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, initial_level: int = DEFAULT_QUALITY):
        """Initialize common quality-ladder RL agent defaults."""
        super().__init__()
        self.initial_level = initial_level

    def reset(
        self,
        initial_chunk_request: Optional[VideoChunkRequest] = None,
    ) -> VideoChunkRequest:
        """Reset stateless RL agents and return the initial request."""
        return super().reset(initial_chunk_request or QualityLadderRequest(self.initial_level))

    @abstractmethod
    def train(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        p_batch: np.ndarray,
        v_batch: np.ndarray,
        epoch: int,
    ) -> TrainBatchInfo:
        """Train the agent on a batch of experiences.

        Args:
            s_batch: Batch of states.
            a_batch: Batch of actions (one-hot).
            p_batch: Batch of action probabilities.
            v_batch: Batch of computed returns.
            epoch: Current training epoch.

        Returns:
            TrainBatchInfo containing training metrics. Subclasses should
            construct this with loss and any extra metrics in the extra dict.
        """
        pass

    @abstractmethod
    def compute_v(
        self,
        s_batch: List[RLState],
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
    ) -> RLTrainingBatch:
        """Produce a training batch from a trajectory.

        Extracts observations and raw decisions from the trajectory steps,
        converts quality-ladder decisions into RL action fields, computes value
        targets, and returns a training batch ready for the training step.

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            Training batch with computed value targets.
        """
        # Extract data from steps
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L143
        # Note: step.state is State (Any), but for RL agents it should be RLState (np.ndarray)
        s_batch: List[RLState] = [step.state for step in trajectory]  # type: ignore[assignment]
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L156-L157
        a_batch: List[List[int]] = []
        p_batch: List[List[float]] = []
        for step in trajectory:
            action: RLActionDecision = step.action
            assert isinstance(action, RLActionDecision), "step.action must be RLActionDecision"
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L154-L155
            action_vec = [0] * len(action.action_prob)
            action_vec[action.action_index] = 1
            a_batch.append(action_vec)
            p_batch.append(action.action_prob)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L158
        r_batch = [step.reward for step in trajectory]

        # Compute value targets
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L161
        v_batch = self.compute_v(s_batch, a_batch, r_batch, done)

        return RLTrainingBatch(
            s_batch=s_batch,
            a_batch=a_batch,
            p_batch=p_batch,
            v_batch=v_batch,
        )

    def train_batch(
        self,
        training_batches: List[RLTrainingBatch],
        epoch: int,
    ) -> TrainBatchInfo:
        """Train on multiple training batches.

        Concatenates data from all training batches and performs a training step.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L102-L114

        Args:
            training_batches: List of training batches from workers.
            epoch: Current training epoch.

        Returns:
            TrainBatchInfo containing training metrics returned by train().
        """
        s: List[RLState] = []
        a, p, v = [], [], []
        for batch in training_batches:
            s += batch.s_batch
            a += batch.a_batch
            p += batch.p_batch
            v += batch.v_batch

        # Extract state_matrix arrays from RLState objects for stacking.
        # For imitation learning with ImitationObserver, the student_state (RLState)
        # is used for training, which provides the state_matrix attribute.
        s_batch = np.stack([state.state_matrix for state in s], axis=0)
        a_batch = np.vstack(a)
        p_batch = np.vstack(p)
        v_batch = np.vstack(v)

        return self.train(s_batch, a_batch, p_batch, v_batch, epoch)

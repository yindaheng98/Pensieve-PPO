"""Experience pool writer agent for collecting trajectories.

This module provides an agent that writes training data to an experience pool
instead of training a neural network.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
"""

import dataclasses
from typing import Any, Dict, List

from ..agent import AbstractTrainableAgent
from ..agent.trainable import Step
from .abc import DictTrainingBatch
from .pool import ExperiencePool


class ExpPoolWriterAgent(AbstractTrainableAgent):
    """Agent that collects training data into an experience pool instead of training.

    This agent implements AbstractTrainableAgent interface but instead of training
    a neural network, it collects all training batches into an internal experience
    pool and saves them to disk.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
    """

    def __init__(self, exp_pool_path: str):
        """Initialize the experience pool writer agent.

        Args:
            exp_pool_path: Path to save the experience pool file.
        """
        self._exp_pool_path = exp_pool_path
        self._exp_pool = ExperiencePool()

    def get_params(self):
        raise NotImplementedError("ExpPoolWriterAgent.get_params should not be called")

    def set_params(self, params):
        raise NotImplementedError("ExpPoolWriterAgent.set_params should not be called")

    def select_action(self, state):
        raise NotImplementedError("ExpPoolWriterAgent.select_action should not be called")

    def select_action_for_training(self, state):
        raise NotImplementedError("ExpPoolWriterAgent.select_action_for_training should not be called")

    def produce_training_batch(
        self,
        trajectory: List[Step],
        done: bool,
    ) -> DictTrainingBatch:
        """Produce a training batch from a trajectory by extracting state fields.

        This method extracts all fields from each state in the trajectory and
        concatenates them into lists, creating a flexible DictTrainingBatch.

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            DictTrainingBatch with fields extracted from states.
        """
        if not trajectory:
            return DictTrainingBatch(data={})

        data: Dict[str, List[Any]] = {}

        for step in trajectory:
            state = step.state
            # Extract all fields from state dataclass
            for f in dataclasses.fields(state):
                field_name = f.name
                field_value = getattr(state, field_name)
                if field_name not in data:
                    data[field_name] = []
                data[field_name].append(field_value)

        return DictTrainingBatch(data=data)

    def save(self, path: str = None) -> None:
        """Save the experience pool to disk.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L367-L368

        Args:
            path: Ignored. Always saves to the configured exp_pool_path.
        """
        self._exp_pool.save(self._exp_pool_path)

    def train_batch(
        self,
        training_batches: List[DictTrainingBatch],
        epoch: int,
    ) -> Dict[str, float]:
        """Write training batch data to the experience pool instead of training.

        This method adds each training batch to the internal experience pool.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L365-L366

        Args:
            training_batches: List of DictTrainingBatch from workers.
            epoch: Current training epoch.

        Returns:
            Dictionary containing experience collection metrics.
        """
        total_samples = 0
        for batch in training_batches:
            samples_added = self._exp_pool.add_batch(batch)
            total_samples += samples_added

        return {
            'exp_pool_size': len(self._exp_pool),
            'new_samples': total_samples,
            'epoch': epoch,
        }

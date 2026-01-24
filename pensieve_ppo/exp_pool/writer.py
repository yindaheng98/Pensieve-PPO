"""Experience pool writer agent for collecting trajectories.

This module provides an agent that writes training data to an experience pool
instead of training a neural network.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
"""

from typing import Dict, List

from ..agent import AbstractTrainableAgent, Step
from .pool import ExperiencePool, Trajectory

# StepBatch is simply a list of Steps (same as Trajectory)
StepBatch = Trajectory


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
    ) -> StepBatch:
        """Produce a training batch from a trajectory.

        This method simply returns the trajectory as a TrajectoryBatch.

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            TrajectoryBatch (List[Step]) containing the trajectory.
        """
        return trajectory

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
        training_batches: List[StepBatch],
        epoch: int,
    ) -> Dict[str, float]:
        """Write training batch data to the experience pool instead of training.

        This method adds each trajectory to the internal experience pool.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L365-L366

        Args:
            training_batches: List of TrajectoryBatch (List[Step]) from workers.
            epoch: Current training epoch.

        Returns:
            Dictionary containing experience collection metrics.
        """
        total_steps = 0
        for trajectory in training_batches:
            total_steps += self._exp_pool.add_trajectory(trajectory)

        return {
            'exp_pool_trajectories': len(self._exp_pool),
            'exp_pool_total_steps': self._exp_pool.total_steps,
            'new_steps': total_steps,
            'epoch': epoch,
        }

"""PyTorch Dataset for Experience Pool.

This module provides a PyTorch Dataset class for loading trajectories from
an ExperiencePool and converting them to TrainingBatch objects.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py
"""

from torch.utils.data import Dataset

from ..agent import AbstractTrainableAgent, TrainingBatch
from .pool import ExperiencePool


class ExpPoolDataset(Dataset):
    """PyTorch Dataset for loading trajectories from an ExperiencePool.

    This dataset loads trajectories from an ExperiencePool and converts them
    to TrainingBatch objects using the agent's `produce_training_batch` method.

    The `done` flag for each trajectory is determined by the last Step's `done`
    field in the trajectory.

    Example:
        >>> pool = ExperiencePool.load("exp_pool.pkl")
        >>> dataset = ExpPoolDataset(pool, agent)
        >>> dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:
        ...     # batch is a list of TrainingBatch
        ...     loss = compute_loss(batch)

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py
    """

    def __init__(
        self,
        exp_pool: ExperiencePool,
        agent: AbstractTrainableAgent,
    ) -> None:
        """Initialize the ExpPoolDataset.

        Args:
            exp_pool: ExperiencePool containing the trajectory data.
            agent: TrainableAgent used to convert trajectories to TrainingBatch.
                The agent's `produce_training_batch` method will be called.
        """
        self.exp_pool = exp_pool
        self.agent = agent

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self.exp_pool)

    def __getitem__(self, index: int) -> TrainingBatch:
        """Get a TrainingBatch for the trajectory at the given index.

        Args:
            index: Index of the trajectory.

        Returns:
            TrainingBatch produced from the trajectory using the agent's
            `produce_training_batch` method.
        """
        trajectory = self.exp_pool.data[index]
        # Determine done from the last step's done field
        done = trajectory[-1].done if trajectory else False
        return self.agent.produce_training_batch(trajectory, done)

    @property
    def num_trajectories(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self.exp_pool)

    @property
    def total_steps(self) -> int:
        """Return the total number of steps across all trajectories."""
        return self.exp_pool.total_steps

"""PyTorch Dataset for Experience Pool.

This module provides a PyTorch Dataset class for loading trajectories from
an ExperiencePool and converting them to TrainingBatch objects.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py
"""

from typing import List, Tuple

from torch.utils.data import Dataset

from ..agent import Step
from .pool import ExperiencePool


# Type alias for trajectory with done flag
TrajectoryWithDone = Tuple[List[Step], bool]


class ExpPoolDataset(Dataset):
    """PyTorch Dataset for loading trajectories from an ExperiencePool.

    This dataset loads trajectories from an ExperiencePool and returns them
    as (trajectory, done) tuples. The conversion to TrainingBatch should be
    done in the training loop to avoid pickle issues with models that have
    hooks (e.g., from enable_input_require_grads).

    The `done` flag for each trajectory is determined by the last Step's `done`
    field in the trajectory.

    Example:
        >>> pool = ExperiencePool.load("exp_pool.pkl")
        >>> dataset = ExpPoolDataset(pool)
        >>> dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> for trajectories in dataloader:
        ...     # trajectories is a list of (trajectory, done) tuples
        ...     batches = [agent.produce_training_batch(t, d) for t, d in trajectories]

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py
    """

    def __init__(
        self,
        exp_pool: ExperiencePool,
    ) -> None:
        """Initialize the ExpPoolDataset.

        Args:
            exp_pool: ExperiencePool containing the trajectory data.
        """
        self.exp_pool = exp_pool

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self.exp_pool)

    def __getitem__(self, index: int) -> TrajectoryWithDone:
        """Get a trajectory and done flag at the given index.

        Args:
            index: Index of the trajectory.

        Returns:
            Tuple of (trajectory, done) where trajectory is List[Step]
            and done is a boolean indicating if the trajectory ended.
        """
        trajectory = self.exp_pool.data[index]
        # Determine done from the last step's done field
        done = trajectory[-1].done if trajectory else False
        return (trajectory, done)

    @property
    def num_trajectories(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self.exp_pool)

    @property
    def total_steps(self) -> int:
        """Return the total number of steps across all trajectories."""
        return self.exp_pool.total_steps

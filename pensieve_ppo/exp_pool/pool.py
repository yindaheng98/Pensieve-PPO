"""Experience pool for collecting and storing trajectories.

This module provides the ExperiencePool class for managing experience data
collected during agent rollouts. It supports loading from file, saving to file,
and adding batches of training data.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
"""

import os
import pickle
from typing import List

from ..agent import Step

# Type alias for a single trajectory
Trajectory = List[Step]


class ExperiencePool:
    """Experience pool for collecting and storing trajectories.

    This class stores experience data as a list of trajectories,
    where each trajectory is a List[Step].

    Attributes:
        data: List of trajectories (List[List[Step]]).

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
    """

    def __init__(self, data: List[Trajectory] = []):
        """Initialize the experience pool.

        Args:
            data: Optional initial list of trajectories. If None, an empty list is used.
        """
        self._data: List[Trajectory] = data

    @property
    def data(self) -> List[Trajectory]:
        """Get the internal data list (list of trajectories)."""
        return self._data

    def __len__(self) -> int:
        """Return the number of trajectories in the pool.

        Returns:
            Number of trajectories in the pool.
        """
        return len(self._data)

    @property
    def total_steps(self) -> int:
        """Return the total number of steps across all trajectories.

        Returns:
            Total number of Steps in all trajectories.
        """
        return sum(len(traj) for traj in self._data)

    def add_trajectory(self, trajectory: Trajectory) -> int:
        """Add a single trajectory to the experience pool.

        Args:
            trajectory: A trajectory (List[Step]) to add.

        Returns:
            Number of steps added from this trajectory.
        """
        steps_added = len(trajectory)
        self._data.append(trajectory)
        return steps_added

    def save(self, path: str) -> None:
        """Save the experience pool to a file.

        The data is saved as a pickle file containing the list of trajectories.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L367-L368

        Args:
            path: Path to save the experience pool file.
        """
        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self._data, f)

    @classmethod
    def load(cls, path: str) -> "ExperiencePool":
        """Load an experience pool from a file.

        Args:
            path: Path to the experience pool file.

        Returns:
            ExperiencePool instance loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Experience pool file not found: {path}")

        with open(path, 'rb') as f:
            data: List[Trajectory] = pickle.load(f)

        return cls(data=data)

    def __repr__(self) -> str:
        """Return a string representation of the experience pool."""
        return f"ExperiencePool(trajectories={len(self)}, total_steps={self.total_steps})"

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

from ..agent.trainable import Step


class ExperiencePool:
    """Experience pool for collecting and storing trajectories.

    This class stores experience data as a list of Steps.

    Attributes:
        data: List of Step objects.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
    """

    def __init__(self, data: List[Step] = []):
        """Initialize the experience pool.

        Args:
            data: Optional initial list of Steps. If None, an empty list is used.
        """
        self._data: List[Step] = data

    @property
    def data(self) -> List[Step]:
        """Get the internal data list."""
        return self._data

    def __len__(self) -> int:
        """Return the number of experiences in the pool.

        Returns:
            Number of Steps in the pool.
        """
        return len(self._data)

    def add_batch(self, batch: List[Step]) -> int:
        """Add a training batch (TrajectoryBatch) to the experience pool.

        This method extends the internal data list with Steps from the batch.

        Args:
            batch: A TrajectoryBatch (List[Step]) containing the data to add.

        Returns:
            Number of samples added from this batch.
        """
        samples_added = len(batch)
        self._data.extend(batch)
        return samples_added

    def save(self, path: str) -> None:
        """Save the experience pool to a file.

        The data is saved as a pickle file containing a dictionary with a 'data' key.

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
            data: List[Step] = pickle.load(f)

        return cls(data=data)

    def __repr__(self) -> str:
        """Return a string representation of the experience pool."""
        return f"ExperiencePool(size={len(self)})"

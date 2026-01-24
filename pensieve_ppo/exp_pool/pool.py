"""Experience pool for collecting and storing trajectories.

This module provides the ExperiencePool class for managing experience data
collected during agent rollouts. It supports loading from file, saving to file,
and adding batches of training data.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
"""

import os
import pickle
from typing import Any, Dict, List, Optional

from .abc import DictTrainingBatch


class ExperiencePool:
    """Experience pool for collecting and storing trajectories.

    This class stores experience data as a dictionary of lists, where each key
    represents a field name (e.g., 'state', 'action', 'reward') and each value
    is a list of corresponding values across all experiences.

    Attributes:
        data: Dictionary mapping field names to lists of values.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
    """

    def __init__(self, data: Optional[Dict[str, List[Any]]] = None):
        """Initialize the experience pool.

        Args:
            data: Optional initial data dictionary. If None, an empty dict is used.
        """
        self._data: Dict[str, List[Any]] = data if data is not None else {}

    @property
    def data(self) -> Dict[str, List[Any]]:
        """Get the internal data dictionary."""
        return self._data

    def __len__(self) -> int:
        """Return the number of experiences in the pool.

        Returns:
            Number of experiences, determined by the length of the first field's list.
            Returns 0 if the pool is empty.
        """
        if not self._data:
            return 0
        first_key = next(iter(self._data))
        return len(self._data[first_key])

    def add_batch(self, batch: "DictTrainingBatch") -> int:
        """Add a training batch to the experience pool.

        This method extends the internal data lists with values from the batch.
        The batch's data dictionary should have the same structure as the pool's
        data (same keys mapping to lists of values).

        Args:
            batch: A DictTrainingBatch containing the data to add.

        Returns:
            Number of samples added from this batch.

        Raises:
            TypeError: If batch is not a DictTrainingBatch or doesn't have a data attribute.
        """
        if not hasattr(batch, 'data'):
            raise TypeError(f"Expected DictTrainingBatch with 'data' attribute, got {type(batch)}")

        samples_added = 0
        for field_name, field_values in batch.data.items():
            if field_name not in self._data:
                self._data[field_name] = []
            self._data[field_name].extend(field_values)
            # Track samples from first field
            if samples_added == 0:
                samples_added = len(field_values)

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
            pickle.dump({'data': self._data}, f)

    @classmethod
    def load(cls, path: str) -> "ExperiencePool":
        """Load an experience pool from a file.

        Args:
            path: Path to the experience pool file.

        Returns:
            ExperiencePool instance loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Experience pool file not found: {path}")

        with open(path, 'rb') as f:
            loaded = pickle.load(f)

        # Handle both old format (dict with 'data' key) and raw data dict
        if isinstance(loaded, dict):
            if 'data' in loaded:
                data = loaded['data']
            else:
                # Assume the dict itself is the data
                data = loaded
        else:
            raise ValueError(f"Invalid experience pool format: expected dict, got {type(loaded)}")

        return cls(data=data)

    def get_fields(self) -> List[str]:
        """Get the names of all fields in the experience pool.

        Returns:
            List of field names.
        """
        return list(self._data.keys())

    def __repr__(self) -> str:
        """Return a string representation of the experience pool."""
        fields = self.get_fields()
        return f"ExperiencePool(size={len(self)}, fields={fields})"

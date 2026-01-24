"""Abstract base classes and data types for experience pool.

This module provides the base data types used by the experience pool,
including the DictTrainingBatch class for storing training data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..agent.trainable import TrainingBatch


@dataclass
class DictTrainingBatch(TrainingBatch):
    """A flexible training batch that stores data as a dictionary of lists.

    This batch type dynamically adapts its fields based on the state's fields,
    making it suitable for experience pool generation where the exact state
    structure may vary.

    Attributes:
        data: Dictionary mapping field names to lists of values.
    """
    data: Dict[str, List[Any]] = field(default_factory=dict)

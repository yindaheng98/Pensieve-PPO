"""Abstract base classes for agents.

This module provides the abstract base class hierarchy for all agents:
- AbstractAgent: Base class with select_action

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


class AbstractAgent(ABC):
    """Abstract base class for agents with action selection capability.

    This class defines the minimal interface for agents that can select
    actions from states.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Tuple[int, List[float]]:
        """Select an action for a given state.

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        pass

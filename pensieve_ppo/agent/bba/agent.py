"""BBA (Buffer-Based Adaptive) Agent implementation.

This module implements the BBA algorithm for adaptive bitrate streaming.
BBA selects bitrates based purely on the current buffer level, using a
simple piecewise linear mapping.

Reference:
    https://github.com/GenetProject/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

import logging
from typing import List, Tuple

import numpy as np

from ..abc import AbstractAgent


# BBA Parameters
# https://github.com/GenetProject/Genet/blob/main/src/simulator/abr_simulator/bba.py#L15-L16
RESERVOIR = 5  # seconds - minimum buffer threshold
CUSHION = 10   # seconds - buffer range for linear interpolation


class BBAAgent(AbstractAgent):
    """Buffer-Based Adaptive (BBA) Agent.

    This agent implements a simple buffer-based ABR algorithm that selects
    bitrates based on the current buffer occupancy:
    - If buffer < RESERVOIR: select lowest bitrate (0)
    - If buffer >= RESERVOIR + CUSHION: select highest bitrate
    - Otherwise: linearly interpolate between min and max bitrate

    The algorithm is deterministic and does not require training.

    Attributes:
        action_dim: Number of available bitrate levels.
        reservoir: Minimum buffer threshold (seconds).
        cushion: Buffer range for linear interpolation (seconds).
    """

    def __init__(
        self,
        action_dim: int,
        reservoir: float = RESERVOIR,
        cushion: float = CUSHION,
        **kwargs,
    ):
        """Initialize the BBA agent.

        Args:
            action_dim: Number of discrete actions (bitrate levels).
            reservoir: Minimum buffer threshold in seconds (default: 5).
            cushion: Buffer range for linear interpolation in seconds (default: 10).
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self.action_dim = action_dim
        self.reservoir = reservoir
        self.cushion = cushion
        if kwargs:
            logging.warning(f"kwargs are ignored in BBAAgent: {kwargs}")

    def get_bitrate_from_buffer(self, buffer_size: float) -> int:
        """Compute bitrate level from buffer size using BBA algorithm.

        Reference:
            https://github.com/GenetProject/Genet/blob/main/src/simulator/abr_simulator/bba.py#L30-L37

        Args:
            buffer_size: Current buffer occupancy in seconds.

        Returns:
            Selected bitrate level (0 to action_dim-1).
        """
        if buffer_size < self.reservoir:
            bit_rate = 0
        elif buffer_size >= self.reservoir + self.cushion:
            bit_rate = self.action_dim - 1
        else:
            # Linear interpolation between reservoir and reservoir + cushion
            bit_rate = (self.action_dim - 1) * (buffer_size - self.reservoir) / float(self.cushion)
        return int(bit_rate)

    def select_action(self, state: np.ndarray) -> Tuple[int, List[float]]:
        """Select an action for a given state.

        BBA uses the buffer size from the state to determine the action.
        The returned probability distribution is a one-hot encoding of
        the selected action.

        The state format follows BBAStateObserver:
        - state[0] = buffer_size in seconds (not normalized)

        Args:
            state: Input state with shape (1,) from BBAStateObserver,
                   containing buffer_size in seconds.

        Returns:
            Action probability distribution as a 1D list (one-hot for BBA).
        """
        # Extract buffer size from state (already in seconds, not normalized)
        # State format: state[0] = buffer_size in seconds
        buffer_size = state[0]

        # Get bitrate from buffer using BBA algorithm
        bit_rate = self.get_bitrate_from_buffer(buffer_size)

        # Return one-hot probability distribution
        # BBA is deterministic, so we return 1.0 for selected action, 0.0 for others
        action_prob = [0.0] * self.action_dim
        action_prob[bit_rate] = 1.0

        return bit_rate, action_prob

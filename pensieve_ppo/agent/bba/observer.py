"""BBA State Observer.

This module provides a simplified state observer for BBA algorithm.
BBA only needs buffer_size to make decisions, so the state is simplified
to a single value instead of the full RL state representation.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""


import numpy as np
from gymnasium import spaces

from ..rl.observer import RLABRStateObserver
from ...core.simulator import StepResult
from ...gym.env import ABREnv


class BBAStateObserver(RLABRStateObserver):
    """State observer for BBA algorithm.

    BBA only needs buffer_size to make bitrate decisions, so this observer
    outputs a simplified state containing only the buffer_size (in seconds).

    This simplifies the BBA agent implementation by providing the buffer_size
    directly without normalization.
    """

    @property
    def observation_space(self) -> spaces.Box:
        """Gymnasium observation space for the BBA state.

        Returns a 1D space containing only buffer_size.
        """
        return spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )

    def build_and_set_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> np.ndarray:
        """Build initial state representation on reset.

        For BBA, the initial state is just buffer_size = 0.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index (unused for BBA state).

        Returns:
            Initial state array with shape (1,) containing buffer_size.
        """
        # Initial buffer is empty (0 seconds)
        state = np.array([0.0], dtype=np.float32)
        # Set internal state
        self.state = state
        return state

    def compute_and_update_state(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
    ) -> np.ndarray:
        """Compute new state representation from simulator result.

        For BBA, the state is simply the current buffer_size in seconds.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected (unused for BBA state).
            result: Result from simulator.step().

        Returns:
            State array with shape (1,) containing buffer_size in seconds.
        """
        # BBA only needs buffer_size (in seconds, not normalized)
        buffer_size = result.buffer_size
        state = np.array([buffer_size], dtype=np.float32)
        # Set internal state
        self.state = state
        return state

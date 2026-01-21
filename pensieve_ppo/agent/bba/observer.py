"""BBA State Observer.

This module provides a state observer for BBA algorithm that is compatible
with RL training for imitation learning.

BBA only needs buffer_size to make decisions, but the state also contains
the full RL state_matrix for training RL agents via imitation learning.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from ..rl.observer import RLABRStateObserver, RLState
from ...core.simulator import StepResult
from ...gym.env import ABREnv


@dataclass
class BBAState(RLState):
    """State class for BBA algorithm.

    This dataclass extends RLState to provide the buffer_size field that
    BBA needs for decision making, while also maintaining the full state_matrix
    for RL training compatibility.

    By inheriting from RLState, BBAState is compatible with RL training,
    enabling imitation learning where an RL agent learns from BBA decisions.

    Attributes:
        state_matrix: The numpy array representing the observation state (inherited from RLState).
        buffer_size: Current buffer size in seconds (for BBA decision making).
    """
    buffer_size: float


class BBAStateObserver(RLABRStateObserver):
    """State observer for BBA algorithm.

    This observer extends RLABRStateObserver to provide BBAState objects that:
    1. Contain buffer_size for BBA's decision making (simple and direct)
    2. Contain state_matrix for RL training compatibility (imitation learning)

    This allows trajectories collected by BBA agents to be used for training
    RL agents via imitation learning / behavioral cloning.
    """

    def build_and_set_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> BBAState:
        """Build initial BBAState on reset.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial BBAState with zero state_matrix and buffer_size=0.
        """
        # Get the RLState from parent, extract state_matrix for BBAState
        # Note: state_matrix is already copied in parent's build_and_set_initial_state
        rl_state = super().build_and_set_initial_state(env, initial_bit_rate)
        return BBAState(
            state_matrix=rl_state.state_matrix,
            buffer_size=0.0,  # Initial buffer is empty
        )

    def compute_and_update_state(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
    ) -> BBAState:
        """Compute new BBAState from simulator result.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            New BBAState with updated state_matrix and buffer_size.
        """
        # Get the RLState from parent, extract state_matrix for BBAState
        # Note: state_matrix is already copied in parent's compute_and_update_state
        rl_state = super().compute_and_update_state(env, bit_rate, result)
        return BBAState(
            state_matrix=rl_state.state_matrix,
            buffer_size=result.buffer_size,  # BBA uses buffer_size directly (in seconds)
        )

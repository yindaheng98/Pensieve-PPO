"""BBA State Observer.

This module provides a state observer for the BBA (Buffer-Based Adaptive)
algorithm. The observer inherits from RLABRStateObserver to reuse reward
calculation logic, but BBAState is independent (does not inherit from RLState).

For imitation learning (e.g., training RL from BBA demonstrations), use
ImitationObserver from pensieve_ppo.gym.imitate to combine BBAStateObserver
with an RLABRStateObserver.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from dataclasses import dataclass

from ..rl.observer import RLABRStateObserver
from ...core.simulator import StepResult
from ...gym.env import ABREnv, State


@dataclass
class BBAState(State):
    """State class for BBA algorithm.

    This dataclass provides the buffer_size field that BBA needs for decision
    making. It inherits directly from State, not from RLState.

    For imitation learning, use ImitationObserver to combine this with an
    RLState-producing observer.

    Attributes:
        buffer_size: Current buffer size in seconds (for BBA decision making).
    """
    buffer_size: float


class BBAStateObserver(RLABRStateObserver):
    """State observer for BBA algorithm.

    This observer inherits from RLABRStateObserver to reuse the reward
    calculation logic (compute_reward method). However, it returns BBAState
    objects (which do not inherit from RLState) containing only the information
    needed for BBA's decision making.

    This design enables:
    1. Reward calculation reuse from RLABRStateObserver
    2. Clean BBAState that doesn't depend on RLState
    3. Flexible composition via ImitationObserver for imitation learning

    Example for standalone BBA:
        >>> observer = BBAStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> env = ABREnv(simulator=simulator, observer=observer)

    Example for imitation learning:
        >>> from pensieve_ppo.gym.imitate import ImitationObserver
        >>> rl_observer = RLABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> bba_observer = BBAStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> imitation_observer = ImitationObserver(rl_observer, bba_observer)
        >>> env = ABREnv(simulator=simulator, observer=imitation_observer)
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
            Initial BBAState with buffer_size=0.
        """
        return BBAState(buffer_size=0.0)

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
            New BBAState with updated buffer_size.
        """
        return BBAState(buffer_size=result.buffer_size)

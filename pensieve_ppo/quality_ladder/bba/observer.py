"""BBA State Observer.

This module provides a state observer for the BBA (Buffer-Based Adaptive)
algorithm. The observer inherits from RLABRStateObserver to reuse quality-ladder
QoE integration, but BBAState is independent (does not inherit from RLState).

For imitation learning (e.g., training RL from BBA demonstrations), use
ImitationObserver from pensieve_ppo.gym.imitate to combine BBAStateObserver
with an RLABRStateObserver.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from dataclasses import dataclass

from ..abc import QualityLadderRequest
from ..rl.observer import RLABRStateObserver
from ...core.simulator import StepResult
from ...gym.env import ABREnv, State
from ...gym.qoe import QoEState


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

    This observer inherits from RLABRStateObserver to reuse quality-ladder QoE
    integration. However, it returns BBAState objects (which do not inherit from
    RLState) containing only the information needed for BBA's decision making.

    This design enables:
    1. QoE calculation reuse through RLABRStateObserver
    2. Clean BBAState that doesn't depend on RLState
    3. Flexible composition via ImitationObserver for imitation learning

    Example for standalone BBA:
        >>> observer = BBAStateObserver()
        >>> env = ABREnv(simulator=simulator, observer=observer)

    Example for imitation learning:
        >>> from pensieve_ppo.gym.imitate import ImitationObserver
        >>> rl_observer = RLABRStateObserver()
        >>> bba_observer = BBAStateObserver()
        >>> imitation_observer = ImitationObserver(rl_observer, bba_observer)
        >>> env = ABREnv(simulator=simulator, observer=imitation_observer)
    """

    def compute_and_update_state(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
        qoe_state: QoEState,
    ) -> BBAState:
        """Compute new BBAState from simulator result.

        Args:
            env: The ABREnv instance to observe.
            chunk_request: Current video chunk request.
            result: Result from simulator.step().
            qoe_state: Generic QoE observation from QoEObserver.observe().

        Returns:
            New BBAState with updated buffer_size.
        """
        return BBAState(buffer_size=result.buffer_size)

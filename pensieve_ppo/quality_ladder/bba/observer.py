"""BBA State Observer.

This module provides a state observer for the BBA (Buffer-Based Adaptive)
algorithm. The observer reuses QoEObserver for reward calculation, while
BBAState contains only the buffer information needed by BBA.

For imitation learning (e.g., training RL from BBA demonstrations), use
ImitationObserver from pensieve_ppo.gym.imitate to combine BBAStateObserver
with an RLABRStateObserver.

Reference:
    https://github.com/hongzimao/Genet/blob/main/src/simulator/abr_simulator/bba.py
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ..abc import QualityLadderRequest
from ..observer import QualityLadderQoEObserver
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


class BBAStateObserver(QualityLadderQoEObserver):
    """State observer for BBA algorithm.

    This observer inherits from QoEObserver to reuse reward calculation.
    It returns BBAState objects containing only the information needed for
    BBA's decision making.

    This design enables:
    1. QoE calculation reuse without depending on RL state handling
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

    def observe(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
    ) -> Tuple[BBAState, float, Dict[str, Any]]:
        """Process simulator result and build BBAState."""
        _, reward, info = super().observe(env, chunk_request, result)
        return BBAState(buffer_size=result.buffer_size), reward, info

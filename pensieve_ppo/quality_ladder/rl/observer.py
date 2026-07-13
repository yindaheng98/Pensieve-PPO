"""RL-specific ABR State Observer.

This module implements the state observer and reward calculation for
reinforcement learning based ABR (Adaptive Bitrate) agents.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...core.simulator import StepResult
from ...gym.env import ABREnv
from ...gym.qoe import QoEObserver, QoEState
from ..abc import QualityLadderRequest
from ..player import QualityLadderResolvedChunk
from .utils import (
    get_chunk_qualities,
    get_next_chunk_sizes,
)


@dataclass
class RLState(QoEState):
    """State class for RL agents.

    This dataclass wraps the numpy state array used by RL agents for training.
    By inheriting from QoEState, it ensures compatibility with other agent types
    (e.g., MPC, BBA) for imitation learning scenarios.

    Attributes:
        QoEState fields: Generic QoE inputs and reward.
        state_matrix: The numpy array representing the observation state.
            Shape is (S_INFO, state_history_len), e.g., (6, 8) by default.
    """
    state_matrix: np.ndarray


# State dimensions
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L8
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past

# Quality-ladder and throughput normalization constants
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
M_IN_K = 1000.0


class RLABRStateObserver(QoEObserver):
    """Observer for ABR environment state and reward calculation.

    This class handles state representation and reward computation,
    decoupled from the environment dynamics.

    Note:
        The internal state (self.state) and the state returned to external
        callers can be different. The internal state is updated by
        compute_and_update_state(), while the observe() method returns a copy
        that may be modified or transformed before being returned.

    Attributes:
        rebuf_penalty: Penalty coefficient for rebuffering.
        smooth_penalty: Penalty coefficient for quality changes.
        state_history_len: Number of past observations in state.
        buffer_norm_factor: Normalization factor for buffer size.
        state: Internal state array (may differ from returned state).
    """

    def __init__(
        self,
        state_history_len: int = S_LEN,
        **configs: Any,
    ):
        """Initialize the ABR state observer.

        Args:
            state_history_len: Number of past observations to keep in state (default: 8)
            **configs: QoEObserver configuration such as rebuf_penalty,
                smooth_penalty, and buffer_norm_factor.
        """

        super().__init__(**configs)
        self.state_history_len = state_history_len

        # State tracking (initialized in reset)
        # Note: self._state_matrix is the internal numpy array for manipulation,
        # while methods return RLState objects wrapping this array.
        self.state_matrix: Optional[np.ndarray] = None

    def reset(
        self,
        env: ABREnv,
    ) -> None:
        """Reset observer state.

        Args:
            env: The ABREnv instance to observe.
        """
        super().reset(env)
        self.state_matrix = np.zeros((S_INFO, self.state_history_len), dtype=np.float32)

    def compute_quality(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
    ) -> float:
        """Compute the quality-ladder quality value in QoE reward units."""
        resolved_chunk = result.resolved_chunk
        if not isinstance(resolved_chunk, QualityLadderResolvedChunk):
            raise TypeError(
                "RL observers require QualityLadderResolvedChunk, "
                f"got {type(resolved_chunk).__name__}"
            )
        return resolved_chunk.quality / M_IN_K

    def compute_and_update_state(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
        qoe_state: QoEState,
    ) -> RLState:
        """Compute new state representation from simulator result.

        This method updates the internal state matrix and returns an RLState object.
        Note that the returned state may be modified or transformed before
        being returned to external callers in the observe() method.

        Args:
            env: The ABREnv instance to observe.
            chunk_request: Current video chunk request.
            result: Result from simulator.step().
            qoe_state: Generic QoE observation from QoEObserver.observe().

        Returns:
            The computed state as RLState dataclass.
        """
        chunk_til_video_end_cap = env.simulator.video_player.total_chunks
        # Unpack result (matches original variable names)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L75-L78
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L48-L51
        delay, \
            video_chunk_size, \
            video_chunk_remain = (
                result.delay,
                result.video_chunk_size,
                result.video_chunk_remain,
            )
        next_video_chunk_sizes = get_next_chunk_sizes(env, result)

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L90-L104
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L52-L66
        state = np.roll(self.state_matrix, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = qoe_state.quality * M_IN_K / \
            float(np.max(get_chunk_qualities(env, result)))  # last quality
        state[1, -1] = qoe_state.buffer_size_norm  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / self.buffer_norm_factor  # 10 sec
        state[4, :len(next_video_chunk_sizes)] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  chunk_til_video_end_cap) / float(chunk_til_video_end_cap)

        # Update internal state matrix
        self.state_matrix = state

        return RLState(
            **asdict(qoe_state),
            state_matrix=state.copy(),
        )

    def observe(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
    ) -> Tuple[RLState, float, Dict[str, Any]]:
        """Process simulator result: compute reward and update state.

        Note:
            The returned state is a copy of the internal state and may differ
            from self.state. This allows the internal state to be preserved
            while returning a potentially modified version to external callers.

        Args:
            env: The ABREnv instance to observe.
            chunk_request: Current video chunk request.
            result: Result from simulator.step().

        Returns:
            Tuple of (state_copy, reward, info_dict), where state_copy may
            differ from the internal self.state.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L83-L87
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L80-L84
        qoe_state, reward, info = super().observe(env, chunk_request, result)

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L90-L104
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L52-L66
        state = self.compute_and_update_state(env, chunk_request, result, qoe_state)

        return state, reward, info

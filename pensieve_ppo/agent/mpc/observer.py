"""MPC State Observer for MPC algorithm.

This module provides a state observer for MPC (Model Predictive Control) algorithm
that provides basic state information without future bandwidth information.

For imitation learning (e.g., training RL from MPC demonstrations), use
ImitationObserver from pensieve_ppo.gym.imitate to combine MPCABRStateObserver
with an RLABRStateObserver.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py
"""

from dataclasses import dataclass, field
from typing import List


from ..rl import RLABRStateObserver
from ...core.simulator import StepResult
from ...gym import ABREnv, State

from ...core.trace import TraceSimulator
from ...core.video import VideoPlayer


# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L8-L10
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0

# Normalization constant for throughput calculation
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
M_IN_K = 1000.0


@dataclass
class MPCState(State):
    """State class for MPC algorithm.

    This dataclass provides the information needed by the MPC algorithm for
    decision making. It inherits directly from State, not from RLState.

    For imitation learning, use ImitationObserver to combine this with an
    RLState-producing observer.

    Attributes:
        trace_simulator: Reference to the trace simulator.
        video_player: Reference to the video player for chunk information.
        bit_rate: Current bitrate level.
        levels_quality: Quality metric list for each bitrate level.
        rebuf_penalty: Penalty coefficient for rebuffering.
        smooth_penalty: Penalty coefficient for bitrate changes.
        past_bandwidths: Fixed-length list of past bandwidth values in MB/s for
            bandwidth prediction. Length is state_history_len (default S_LEN=8).
            Values roll left, with new values appended at the end.
    """
    trace_simulator: TraceSimulator
    video_player: VideoPlayer
    bit_rate: int
    levels_quality: list[float]
    rebuf_penalty: float
    smooth_penalty: float
    past_bandwidths: List[float] = field(default_factory=list)

    def get_chunk_size(self, quality: int, chunk_idx: int) -> int:
        """Get the size of a video chunk at given quality and index.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L44-L49

        Args:
            quality: Bitrate quality level (0 to bitrate_levels-1).
            chunk_idx: Video chunk index.

        Returns:
            Chunk size in bytes.
        """
        if chunk_idx < 0 or chunk_idx >= self.video_player.total_chunks:
            return 0
        return self.video_player.get_chunk_size(quality, chunk_idx)

    @property
    def video_chunk_counter(self) -> int:
        """Get current video chunk index from video player.

        This corresponds to last_index in the original MPC code:
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
                   = TOTAL_CHUNKS - (TOTAL_CHUNKS - video_chunk_counter)
                   = video_chunk_counter

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L177
        """
        return self.video_player.video_chunk_counter

    @property
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        return self.video_player.total_chunks

    @property
    def buffer_size(self) -> float:
        """Get current buffer size from trace simulator.

        Returns the current buffer size in seconds by querying the trace simulator
        directly. The buffer size is converted from milliseconds to seconds.

        Returns:
            Buffer size in seconds.
        """
        buffer_size_ms = self.trace_simulator.get_buffer_size()
        return buffer_size_ms / MILLISECONDS_IN_SECOND


class MPCABRStateObserver(RLABRStateObserver):
    """State observer for MPC algorithm.

    This observer inherits from RLABRStateObserver to reuse the reward
    calculation logic (compute_reward method). However, it returns MPCState
    objects (which do not inherit from RLState) containing only the information
    needed for MPC's decision making.

    This design enables:
    1. Reward calculation reuse from RLABRStateObserver
    2. Clean MPCState that doesn't depend on RLState
    3. Flexible composition via ImitationObserver for imitation learning

    Example for standalone MPC:
        >>> observer = MPCABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> env = ABREnv(simulator=simulator, observer=observer)

    Example for imitation learning:
        >>> from pensieve_ppo.gym.imitate import ImitationObserver
        >>> rl_observer = RLABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> mpc_observer = MPCABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> imitation_observer = ImitationObserver(rl_observer, mpc_observer)
        >>> env = ABREnv(simulator=simulator, observer=imitation_observer)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the MPC state observer."""
        super().__init__(*args, **kwargs)
        # Track bandwidth history for MPC bandwidth prediction
        # Fixed length, initialized in build_and_set_initial_state
        self.past_bandwidths: List[float] = [0.0] * self.state_history_len

    def build_and_set_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> MPCState:
        """Build initial MPCState on reset.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial MPCState with zero-initialized bandwidth history.
        """
        # Reset bandwidth history on new episode (fixed length, all zeros)
        self.past_bandwidths = [0.0] * self.state_history_len

        return MPCState(
            trace_simulator=env.simulator.trace_simulator.unwrapped,
            video_player=env.simulator.video_player,
            bit_rate=initial_bit_rate,
            levels_quality=self.levels_quality,
            rebuf_penalty=self.rebuf_penalty,
            smooth_penalty=self.smooth_penalty,
            past_bandwidths=list(self.past_bandwidths),
        )

    def compute_and_update_state(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
    ) -> MPCState:
        """Compute new MPCState from simulator result.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            New MPCState with updated bandwidth history.
        """
        # Compute bandwidth: video_chunk_size / delay / M_IN_K (in MB/s)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L56-L57
        bandwidth = float(result.video_chunk_size) / float(result.delay) / M_IN_K

        # Roll and update bandwidth history (like np.roll with -1)
        self.past_bandwidths = self.past_bandwidths[1:] + [bandwidth]

        return MPCState(
            trace_simulator=env.simulator.trace_simulator.unwrapped,
            video_player=env.simulator.video_player,
            bit_rate=bit_rate,
            levels_quality=self.levels_quality,
            rebuf_penalty=self.rebuf_penalty,
            smooth_penalty=self.smooth_penalty,
            past_bandwidths=list(self.past_bandwidths),
        )

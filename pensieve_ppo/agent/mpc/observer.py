"""MPC State Observer for MPC algorithm.

This module provides a state observer for MPC (Model Predictive Control) algorithm
that provides basic state information without future bandwidth information.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py
"""

from dataclasses import dataclass


from ..rl import RLABRStateObserver, RLState
from ...core.simulator import StepResult
from ...gym import ABREnv

from ...core.trace import TraceSimulator
from ...core.video import VideoPlayer


# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L8-L10
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0


@dataclass
class MPCState(RLState):
    """State class for MPC algorithm.

    This dataclass extends RLState to provide additional methods for accessing
    video and simulator information needed by the MPC algorithm.

    By inheriting from RLState, MPCState is compatible with RL training,
    enabling imitation learning where an RL agent learns from MPC decisions.

    Attributes:
        state_matrix: The numpy array representing the observation state (inherited from RLState).
        trace_simulator: Reference to the trace simulator.
        video_player: Reference to the video player for chunk information.
        bit_rate: Current bitrate level.
        levels_quality: Quality metric list for each bitrate level.
        rebuf_penalty: Penalty coefficient for rebuffering.
        smooth_penalty: Penalty coefficient for bitrate changes.
    """
    trace_simulator: TraceSimulator
    video_player: VideoPlayer
    bit_rate: int
    levels_quality: list[float]
    rebuf_penalty: float
    smooth_penalty: float

    def copy(self) -> 'MPCState':
        """Create a copy of this MPCState.

        Creates a deep copy of the state_matrix array while sharing references to
        trace_simulator and video_player.

        Returns:
            A new MPCState with copied state_matrix array.
        """
        return MPCState(
            state_matrix=self.state_matrix.copy(),
            trace_simulator=self.trace_simulator,
            video_player=self.video_player,
            bit_rate=self.bit_rate,
            levels_quality=self.levels_quality,
            rebuf_penalty=self.rebuf_penalty,
            smooth_penalty=self.smooth_penalty,
        )

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

    This observer extends RLABRStateObserver to provide MPCState objects
    that include video and simulator information for MPC decision making.
    """

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
            Initial MPCState with zero state_matrix array.
        """
        state = MPCState(
            state_matrix=super().build_and_set_initial_state(env, initial_bit_rate).state_matrix,
            trace_simulator=env.simulator.trace_simulator.unwrapped,
            video_player=env.simulator.video_player,
            bit_rate=initial_bit_rate,
            levels_quality=self.levels_quality,
            rebuf_penalty=self.rebuf_penalty,
            smooth_penalty=self.smooth_penalty,
        )
        return state

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
            New MPCState with updated observation.
        """
        state = MPCState(
            state_matrix=super().compute_and_update_state(env, bit_rate, result).state_matrix,
            trace_simulator=env.simulator.trace_simulator.unwrapped,
            video_player=env.simulator.video_player,
            bit_rate=bit_rate,
            levels_quality=self.levels_quality,
            rebuf_penalty=self.rebuf_penalty,
            smooth_penalty=self.smooth_penalty,
        )
        return state

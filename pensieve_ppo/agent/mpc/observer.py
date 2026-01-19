"""MPC State Observer with Future Bandwidth Prediction.

This module provides a state observer for MPC (Model Predictive Control) algorithm
that includes future bandwidth information for decision making.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py
"""

from dataclasses import dataclass

import numpy as np

from ..rl import RLABRStateObserver
from ...core.simulator import StepResult
from ...gym import ABREnv

from ...core.trace import TraceSimulator
from ...core.video import VideoPlayer


# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L8-L10
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0


@dataclass
class PredictionState:
    """State class for MPC algorithm with future prediction capabilities.

    This is a mostly read-only dataclass that wraps the numpy state array
    and provides methods for computing future download times using virtual
    pointers. The virtual pointers are copied from the Observer when the
    state is created, and only modified within this instance.

    Attributes:
        state: The numpy array representing the observation state.
        trace_simulator: Reference to the trace simulator for future prediction.
        video_player: Reference to the video player for chunk information.
        virtual_mahimahi_ptr: Virtual pointer for future prediction (internal).
        virtual_last_mahimahi_time: Virtual time for future prediction (internal).
    """
    state: np.ndarray
    trace_simulator: TraceSimulator
    video_player: VideoPlayer
    virtual_mahimahi_ptr: int
    virtual_last_mahimahi_time: float

    def copy(self) -> 'PredictionState':
        """Create a copy of this PredictionState.

        Creates a deep copy of the state array while sharing references to
        trace_simulator and video_player. Virtual pointers are copied so each
        instance maintains its own prediction state.

        Returns:
            A new PredictionState with copied state array and virtual pointers.
        """
        return PredictionState(
            state=self.state.copy(),
            trace_simulator=self.trace_simulator,
            video_player=self.video_player,
            virtual_mahimahi_ptr=self.virtual_mahimahi_ptr,
            virtual_last_mahimahi_time=self.virtual_last_mahimahi_time,
        )

    def reset_download_time(self) -> None:
        """Reset virtual pointers to current actual pointers for future prediction.

        This method synchronizes the virtual pointers with the actual
        trace simulator pointers, preparing for a new future prediction sequence.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L56-L58
        """
        self.virtual_mahimahi_ptr = self.trace_simulator.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.trace_simulator.last_mahimahi_time

    def get_download_time(self, video_chunk_size: int) -> float:
        """Compute download time for a chunk using virtual pointers.

        This method simulates the download time without affecting the actual
        simulator state, allowing the MPC algorithm to peek into the future.
        Only modifies internal virtual pointers.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L60-L92

        Args:
            video_chunk_size: Size of the video chunk in bytes.

        Returns:
            Download time in seconds.
        """
        cooked_time = self.trace_simulator.cooked_time
        cooked_bw = self.trace_simulator.cooked_bw
        packet_payload_portion = self.trace_simulator.packet_payload_portion

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L62-L92
        delay = 0.0  # in seconds
        video_chunk_counter_sent = 0  # in bytes

        while True:
            throughput = cooked_bw[self.virtual_mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = cooked_time[self.virtual_mahimahi_ptr] \
                - self.virtual_last_mahimahi_time

            packet_payload = throughput * duration * packet_payload_portion

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / packet_payload_portion
                delay += fractional_time
                self.virtual_last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.virtual_last_mahimahi_time = cooked_time[self.virtual_mahimahi_ptr]
            self.virtual_mahimahi_ptr += 1

            if self.virtual_mahimahi_ptr >= len(cooked_bw):
                # loop back to the beginning
                # note: trace file starts with time 0
                self.virtual_mahimahi_ptr = 1
                self.virtual_last_mahimahi_time = 0

        return delay

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
        """Get current video chunk index from video player."""
        return self.video_player.video_chunk_counter - 1

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
    """State observer for MPC algorithm with future bandwidth prediction.

    This observer extends RLABRStateObserver to provide PredictionState objects
    that include methods for computing future download times, enabling the
    MPC algorithm to plan ahead using actual future bandwidth information.

    The observer maintains virtual mahimahi pointers that are synchronized with
    the actual trace simulator pointers. These are copied to PredictionState
    instances when they are created.
    """

    def build_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> PredictionState:
        """Build initial PredictionState on reset.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial PredictionState with zero state array and synchronized virtual pointers.
        """
        state = PredictionState(
            state=super().build_initial_state(env, initial_bit_rate),
            trace_simulator=env.simulator.trace_simulator,
            video_player=env.simulator.video_player,
            virtual_mahimahi_ptr=None,
            virtual_last_mahimahi_time=None,
        )
        state.reset_download_time()
        return state

    def compute_state(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
    ) -> PredictionState:
        """Compute new PredictionState from simulator result.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            New PredictionState with updated observation and synchronized virtual pointers.
        """
        state = PredictionState(
            state=super().compute_state(env, bit_rate, result),
            trace_simulator=env.simulator,
            video_player=env.simulator.video_player,
            virtual_mahimahi_ptr=None,
            virtual_last_mahimahi_time=None,
        )
        state.reset_download_time()
        return state

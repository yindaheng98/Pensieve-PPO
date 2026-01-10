"""Base network simulator implementation."""

import math

from .abc import TraceData, AbstractTraceSimulator


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L3
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNK_LEN = 4000.0  # millisec, every time add this amount to buffer
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec


class TraceSimulator(AbstractTraceSimulator):
    """Base network simulator without noise or random trace selection.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L21

    This implements the core simulation logic from fixed_env.py.
    """

    def __init__(
        self,
        trace_data: TraceData,
    ):
        """Initialize the network simulator.

        Args:
            trace_data: Loaded network trace data
        """
        assert len(trace_data.all_cooked_time) == len(trace_data.all_cooked_bw)

        self.all_cooked_time = trace_data.all_cooked_time
        self.all_cooked_bw = trace_data.all_cooked_bw

        # Initialize with first trace
        self.reset()

    # ==================== Methods for reset ====================

    def reset(self) -> None:
        """Reset the simulator state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L31-L39

        Args:
            trace_idx: Trace index to use. If None, keeps current trace.
        """
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def on_video_finished(self) -> None:
        """Handle end of video by moving to next trace.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L137-L150
        """
        self.buffer_size = 0

        self.trace_idx += 1
        if self.trace_idx >= len(self.all_cooked_time):
            self.trace_idx = 0

        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the video
        # note: trace file starts with time 0
        self.mahimahi_ptr = self.mahimahi_start_ptr
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    # ==================== Methods for runtime ====================

    def download_chunk(self, video_chunk_size: int) -> float:
        """Simulate downloading a video chunk over the network.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L55-L87

        Args:
            video_chunk_size: Size of chunk to download in bytes

        Returns:
            Download delay in milliseconds
        """
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        return delay

    def update_buffer(self, delay: float) -> float:
        """Update playback buffer after chunk download.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L89-L96

        Args:
            delay: Download delay in milliseconds

        Returns:
            Rebuffer (stall) time in milliseconds
        """
        # rebuffer time
        rebuf = max(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = max(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        return rebuf

    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer when it exceeds the maximum threshold.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L99-L123

        Returns:
            Sleep time in milliseconds
        """
        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = math.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                    - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        return sleep_time

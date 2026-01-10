"""Base network simulator implementation.

Time Unit Convention:
=====================
- Trace data (cooked_time, last_mahimahi_time): SECONDS
- Buffer and delay values (buffer_size, delay, rebuf, sleep_time): MILLISECONDS
- All parameters (video_chunk_len, buffer_thresh, drain_buffer_sleep_time, link_rtt): MILLISECONDS

The trace files store timestamps in seconds, but all internal buffer/delay
calculations use milliseconds for consistency with the original Pensieve implementation.
"""

import math

from .data import TraceData
from .abc import AbstractTraceSimulator


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
        video_chunk_len: float = VIDEO_CHUNK_LEN,
        buffer_thresh: float = BUFFER_THRESH,
        drain_buffer_sleep_time: float = DRAIN_BUFFER_SLEEP_TIME,
        packet_payload_portion: float = PACKET_PAYLOAD_PORTION,
        link_rtt: float = LINK_RTT,
    ):
        """Initialize the network simulator.

        Args:
            trace_data: Loaded network trace data
            video_chunk_len: Video chunk length in milliseconds (default: 4000.0)
            buffer_thresh: Maximum buffer limit in milliseconds (default: 60000.0)
            drain_buffer_sleep_time: Sleep time when draining buffer in milliseconds (default: 500.0)
            packet_payload_portion: Portion of packet that is payload (default: 0.95)
            link_rtt: Link round-trip time in milliseconds (default: 80)
        """
        assert len(trace_data.all_cooked_time) == len(trace_data.all_cooked_bw)

        self.all_cooked_time = trace_data.all_cooked_time
        self.all_cooked_bw = trace_data.all_cooked_bw

        # Store simulation parameters
        self.video_chunk_len = video_chunk_len
        self.buffer_thresh = buffer_thresh
        self.drain_buffer_sleep_time = drain_buffer_sleep_time
        self.packet_payload_portion = packet_payload_portion
        self.link_rtt = link_rtt

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

            packet_payload = throughput * duration * self.packet_payload_portion

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / self.packet_payload_portion
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
        delay += self.link_rtt

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
        self.buffer_size += self.video_chunk_len

        return rebuf

    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer when it exceeds the maximum threshold.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L99-L123

        Returns:
            Sleep time in milliseconds
        """
        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.buffer_thresh:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.buffer_thresh
            sleep_time = math.ceil(drain_buffer_time / self.drain_buffer_sleep_time) * \
                self.drain_buffer_sleep_time
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

    def get_buffer_size(self) -> float:
        """Get the current buffer size in milliseconds.

        Returns:
            Current buffer size in milliseconds
        """
        return self.buffer_size

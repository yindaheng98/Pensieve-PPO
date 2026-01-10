"""Simulator that combines VideoPlayer and TraceSimulator."""

from dataclasses import dataclass
from typing import List

from ..core.video.player import VideoPlayer
from ..core.trace.simulator import TraceSimulator


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L3
MILLISECONDS_IN_SECOND = 1000.0


@dataclass
class StepResult:
    """Result of a single simulation step.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L159
    """
    delay: float                        # Download delay in milliseconds
    sleep_time: float                   # Sleep time in milliseconds (when buffer is full)
    buffer_size: float                  # Current buffer size in seconds
    rebuffer: float                     # Rebuffering time in seconds
    video_chunk_size: int               # Size of downloaded chunk in bytes
    next_video_chunk_sizes: List[int]   # Sizes of next chunk at each bitrate
    end_of_video: bool                  # Whether video has ended
    video_chunk_remain: int             # Number of remaining chunks


class Simulator:
    """Simulator that combines VideoPlayer and TraceSimulator.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L18

    The `step` method orchestrates these components in order:
    1. `video_player.get_chunk_size` - Get chunk size for requested quality
    2. `trace_simulator.download_chunk` - Simulate network transmission
    3. `trace_simulator.update_buffer` - Update playback buffer after download
    4. `trace_simulator.drain_buffer_overflow` - Handle buffer overflow by sleeping
    5. `video_player.advance` - Move to next chunk
    6. `trace_simulator.on_video_finished` - Handle video end (if needed)
    7. `video_player.get_next_chunk_sizes` - Get sizes for next chunk
    """

    def __init__(
        self,
        video_player: VideoPlayer,
        trace_simulator: TraceSimulator,
    ):
        """Initialize the simulator.

        Args:
            video_player: VideoPlayer instance for managing video playback
            trace_simulator: TraceSimulator instance for network simulation
        """
        self.video_player = video_player
        self.trace_simulator = trace_simulator

    @property
    def buffer_size(self) -> float:
        """Current buffer size in milliseconds."""
        return self.trace_simulator.buffer_size

    @property
    def trace_idx(self) -> int:
        """Current trace index being used."""
        return self.trace_simulator.trace_idx

    def reset(self) -> None:
        """Reset the simulator state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L31

        Args:
            trace_idx: Trace index to use.
        """
        self.video_player.reset()
        self.trace_simulator.reset()

    def step(self, quality: int) -> StepResult:
        """Simulate downloading a video chunk at given quality level.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L48

        Args:
            quality: Bitrate level to download (0 to num_bitrates-1)

        Returns:
            StepResult containing simulation results
        """
        assert quality >= 0
        assert quality < self.video_player.num_bitrates

        video_chunk_size = self.video_player.get_chunk_size(quality)

        # 2. Simulate network download
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L55-L87
        delay = self.trace_simulator.download_chunk(video_chunk_size)

        # 3. Update playback buffer (compute rebuffer and update buffer)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L89-L96
        rebuf = self.trace_simulator.update_buffer(delay)

        # 4. Handle buffer overflow (sleep if buffer too full)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L99-L123
        sleep_time = self.trace_simulator.drain_buffer_overflow()

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L125-L129
        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.trace_simulator.buffer_size

        # 5. Advance to next chunk
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L131
        end_of_video, video_chunk_remain = self.video_player.advance()

        # 6. Handle video end if needed
        if end_of_video:
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L138
            self.video_player.reset()
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L137-L150
            self.trace_simulator.on_video_finished()

        # 7. Get next chunk sizes
        next_video_chunk_sizes = self.video_player.get_next_chunk_sizes()

        return StepResult(
            delay=delay,
            sleep_time=sleep_time,
            buffer_size=return_buffer_size / MILLISECONDS_IN_SECOND,
            rebuffer=rebuf / MILLISECONDS_IN_SECOND,
            video_chunk_size=video_chunk_size,
            next_video_chunk_sizes=next_video_chunk_sizes,
            end_of_video=end_of_video,
            video_chunk_remain=video_chunk_remain,
        )

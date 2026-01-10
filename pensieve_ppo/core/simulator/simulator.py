"""Simulator that combines VideoPlayer and TraceSimulator."""

from dataclasses import dataclass
from typing import List

from ..video.player import VideoPlayer
from ..trace.abc import AbstractTraceSimulator


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
        1. video_player.get_chunk_size - Get chunk size for requested quality
        2. trace_simulator.step - Simulate network download, update buffer, handle overflow
        3. video_player.advance - Move to next chunk
        4. trace_simulator.on_video_finished - Handle video end (if needed)
        5. video_player.get_next_chunk_sizes - Get sizes for next chunk
    """

    def __init__(
        self,
        video_player: VideoPlayer,
        trace_simulator: AbstractTraceSimulator,
    ):
        """Initialize the simulator.

        Args:
            video_player: VideoPlayer instance for managing video playback
            trace_simulator: TraceSimulator instance for network simulation
        """
        self.video_player = video_player
        self.trace_simulator = trace_simulator

    def reset(self) -> None:
        """Reset the simulator state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L31
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

        # 1. Get chunk size for requested quality
        video_chunk_size = self.video_player.get_chunk_size(quality)

        # 2. Simulate network download, update buffer, handle overflow
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L55-L129
        result = self.trace_simulator.step(video_chunk_size)
        delay = result.delay
        rebuf = result.rebuf
        sleep_time = result.sleep_time
        return_buffer_size = result.buffer_size

        # 3. Advance to next chunk
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L131
        end_of_video, video_chunk_remain = self.video_player.advance()

        # 4. Handle video end if needed
        if end_of_video:
            self.video_player.reset()
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L137-L150
            self.trace_simulator.on_video_finished()

        # 5. Get next chunk sizes
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

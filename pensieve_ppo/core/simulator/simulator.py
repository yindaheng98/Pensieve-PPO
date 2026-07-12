"""Simulator that combines VideoPlayer and TraceSimulator.

Time Unit Convention:
=====================
- Internal calculations (from TraceSimulator): MILLISECONDS
- StepResult output: delay/sleep_time in MILLISECONDS, buffer_size/rebuffer in SECONDS

This module converts internal millisecond values to seconds for buffer_size and rebuffer
in the StepResult, while keeping delay and sleep_time in milliseconds for compatibility.
"""

from dataclasses import dataclass

from ..video import VideoChunkRequest, VideoPlayer
from ..trace.abc import AbstractTraceSimulator


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L3
MILLISECONDS_IN_SECOND = 1000.0


@dataclass
class StepResult:
    """Result of a single simulation step.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L159

    Time Unit Notes:
    - delay: [millisec] download time including RTT
    - sleep_time: [millisec] time spent waiting when buffer is full
    - buffer_size: [sec] current playback buffer (converted from internal millisec)
    - rebuffer: [sec] stall/rebuffering time (converted from internal millisec)
    """
    delay: float                        # [millisec] Download delay
    sleep_time: float                   # [millisec] Sleep time (when buffer is full)
    buffer_size: float                  # [sec] Current buffer size
    rebuffer: float                     # [sec] Rebuffering/stall time
    video_chunk_size: int               # [bytes] Size of downloaded chunk
    video_chunk_quality: float          # Resolved quality value of downloaded chunk
    end_of_video: bool                  # Whether video has ended
    video_chunk_remain: int             # Number of remaining chunks


class Simulator:
    """Simulator that combines VideoPlayer and TraceSimulator.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L18

    The `step` method orchestrates these components in order:
        1. video_player.advance - Resolve chunk metadata and move to next chunk
        2. trace_simulator.step - Simulate network download, update buffer, handle overflow
        3. trace_simulator.on_video_finished - Handle video end (if needed)
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

    def step(self, chunk_request: VideoChunkRequest) -> StepResult:
        """Simulate downloading a video chunk for the given chunk request.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L48

        Args:
            chunk_request: Request used to resolve the chunk quality and size.

        Returns:
            StepResult containing simulation results
            - delay, sleep_time: in milliseconds
            - buffer_size, rebuffer: converted to seconds
        """
        buffer_size = self.trace_simulator.get_buffer_size()  # milliseconds

        # 1. Advance to next chunk
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L131
        player_info = self.video_player.advance(
            chunk_request,
            buffer_size,
        )

        # 2. Simulate network download, update buffer, handle overflow
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L55-L129
        result = self.trace_simulator.step(player_info.resolved_chunk.size, player_info.resolved_chunk.length)
        delay = result.delay
        rebuf = result.rebuf
        sleep_time = result.sleep_time
        return_buffer_size = result.buffer_size

        # 3. Handle video end if needed
        if player_info.end_of_video:
            self.video_player.reset()
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L137-L150
            self.trace_simulator.on_video_finished()

        return StepResult(
            delay=delay,                                            # [millisec] - keep as is
            sleep_time=sleep_time,                                  # [millisec] - keep as is
            buffer_size=return_buffer_size / MILLISECONDS_IN_SECOND,  # [millisec] -> [sec]
            rebuffer=rebuf / MILLISECONDS_IN_SECOND,                  # [millisec] -> [sec]
            video_chunk_size=player_info.resolved_chunk.size,                   # [bytes]
            video_chunk_quality=player_info.resolved_chunk.quality,
            end_of_video=player_info.end_of_video,
            video_chunk_remain=player_info.remaining_chunks,
        )

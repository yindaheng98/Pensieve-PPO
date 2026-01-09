"""Abstract network simulator and related data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L3
MILLISECONDS_IN_SECOND = 1000.0


@dataclass
class StepResult:
    """Result of a single simulation step.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L159
    """
    delay: float                        # Download delay in seconds
    sleep_time: float                   # Sleep time in seconds (when buffer is full)
    buffer_size: float                  # Current buffer size in seconds
    rebuffer: float                     # Rebuffering time in seconds
    video_chunk_size: int               # Size of downloaded chunk in bytes
    next_video_chunk_sizes: List[int]   # Sizes of next chunk at each bitrate
    end_of_video: bool                  # Whether video has ended
    video_chunk_remain: int             # Number of remaining chunks


class NetworkSimulator(ABC):
    """Abstract base class for network simulators.

    The `step` method orchestrates these abstract methods in order:
    1. `get_chunk_size` - Get chunk size for requested quality
    2. `download_chunk` - Simulate network transmission
    3. `update_buffer` - Update playback buffer after download
    4. `drain_buffer_overflow` - Handle buffer overflow by sleeping
    5. `advance_video` - Move to next chunk, handle video end
    6. `get_next_chunk_sizes` - Get sizes for next chunk

    Subclasses must implement all abstract methods and properties.
    """

    # ==================== Abstract Methods ====================

    @abstractmethod
    def reset(self, trace_idx: Optional[int] = None) -> None:
        """Reset the simulator state.

        Args:
            trace_idx: Optional trace index to use. If None, use default behavior.
        """
        pass

    @abstractmethod
    def get_chunk_size(self, quality: int) -> int:
        """Get the size of current chunk at given quality level.

        Args:
            quality: Bitrate level (0 to num_bitrates-1)

        Returns:
            Chunk size in bytes
        """
        pass

    @abstractmethod
    def get_next_chunk_sizes(self) -> List[int]:
        """Get sizes of next chunk at all quality levels.

        Returns:
            List of chunk sizes in bytes for each bitrate level
        """
        pass

    @abstractmethod
    def download_chunk(self, chunk_size: int) -> float:
        """Simulate downloading a video chunk over the network.

        This method simulates network transmission delay based on
        bandwidth traces and chunk size.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L59

        Args:
            chunk_size: Size of chunk to download in bytes

        Returns:
            Download delay in milliseconds
        """
        pass

    @abstractmethod
    def update_buffer(self, delay_ms: float) -> float:
        """Update playback buffer after chunk download.

        Calculates rebuffer (stall) time and updates buffer state:
        - Rebuffer = max(delay - buffer, 0)
        - Buffer = max(buffer - delay, 0) + chunk_duration

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L89

        Args:
            delay_ms: Download delay in milliseconds

        Returns:
            Rebuffer (stall) time in milliseconds
        """
        pass

    @abstractmethod
    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer when it exceeds the maximum threshold.

        When buffer exceeds the limit, the client sleeps to drain it.
        This also advances the trace playback position.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L99

        Returns:
            Sleep time in milliseconds
        """
        pass

    # ==================== Concrete Step Method ====================

    def step(self, quality: int) -> StepResult:
        """Simulate downloading a video chunk at given quality level.

        This method orchestrates the simulation by calling abstract methods
        in sequence. Override the individual abstract methods to customize
        behavior rather than overriding this method.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L48

        Args:
            quality: Bitrate level to download (0 to num_bitrates-1)

        Returns:
            StepResult containing simulation results
        """
        assert 0 <= quality < self.num_bitrates

        # 1. Get chunk size for requested quality
        chunk_size = self.get_chunk_size(quality)

        # 2. Simulate network download
        delay_ms = self.download_chunk(chunk_size)

        # 3. Update playback buffer (compute rebuffer and update buffer)
        rebuffer_ms = self.update_buffer(delay_ms)

        # 4. Handle buffer overflow (sleep if buffer too full)
        sleep_ms = self.drain_buffer_overflow()

        # Save buffer size for return (after all updates)
        buffer_ms = self.buffer_size_ms

        # 5. Advance to next chunk (handle video end)
        end_of_video, remaining_chunks = self.advance_video()

        # 6. Get next chunk sizes
        next_chunk_sizes = self.get_next_chunk_sizes()

        # Convert milliseconds to seconds for StepResult
        return StepResult(
            delay=delay_ms / MILLISECONDS_IN_SECOND,
            sleep_time=sleep_ms / MILLISECONDS_IN_SECOND,
            buffer_size=buffer_ms / MILLISECONDS_IN_SECOND,
            rebuffer=rebuffer_ms / MILLISECONDS_IN_SECOND,
            video_chunk_size=chunk_size,
            next_video_chunk_sizes=next_chunk_sizes,
            end_of_video=end_of_video,
            video_chunk_remain=remaining_chunks,
        )

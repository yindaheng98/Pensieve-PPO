"""Abstract simulator interface."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import TraceSimulator


@dataclass
class TraceProgress:
    """Current trace progress information for logging.

    Attributes:
        trace_index: Index of the current trace (0-based)
        all_trace_names: List of all trace file names
    """
    trace_index: int
    all_trace_names: list[str]


@dataclass
class TraceSimulateResult:
    """Result of a single simulation step.

    Attributes:
        delay: Download delay in milliseconds
        rebuf: Rebuffer (stall) time in milliseconds
        sleep_time: Time spent sleeping due to buffer overflow in milliseconds
        buffer_size: Current buffer size in milliseconds
    """
    delay: float
    rebuf: float
    sleep_time: float
    buffer_size: float


class AbstractTraceSimulator(ABC):
    """Abstract base class defining the TraceSimulator interface.

    Both TraceSimulator (concrete implementation) and TraceSimulatorWrapper
    inherit from this class to ensure a consistent interface.
    """

    # ==================== Methods for logging ====================
    @abstractmethod
    def get_trace_progress(self) -> TraceProgress:
        """Get current trace progress information for logging.

        Returns:
            TraceProgress containing trace index and all trace names
        """
        ...

    # ==================== Methods for reset ====================

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulator state."""
        ...

    @abstractmethod
    def on_video_finished(self) -> None:
        """Handle end of video."""
        ...

    # ==================== Methods for runtime ====================

    @abstractmethod
    def download_chunk(self, video_chunk_size: int) -> float:
        """Simulate downloading a video chunk.

        Args:
            video_chunk_size: Size of chunk to download in bytes

        Returns:
            Download delay in milliseconds
        """
        ...

    @abstractmethod
    def update_buffer(self, delay: float) -> float:
        """Update playback buffer after chunk download.

        Args:
            delay: Download delay in milliseconds

        Returns:
            Rebuffer (stall) time in milliseconds
        """
        ...

    @abstractmethod
    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer when it exceeds the maximum threshold.

        Returns:
            Sleep time in milliseconds
        """
        ...

    @abstractmethod
    def get_buffer_size(self) -> float:
        """Get the current buffer size in milliseconds.

        Returns:
            Current buffer size in milliseconds
        """
        ...

    # ==================== Unwrapped property ====================

    @property
    @abstractmethod
    def unwrapped(self) -> 'TraceSimulator':
        """Get the underlying TraceSimulator for accessing state variables.

        All state (buffer_size, trace_idx, etc.) lives on the unwrapped simulator.

        Returns:
            The underlying TraceSimulator instance
        """
        ...

    # ==================== Step method ====================

    def step(self, video_chunk_size: int) -> TraceSimulateResult:
        """Execute one simulation step: download chunk and update buffer.

        This method orchestrates the simulation by calling the abstract methods
        in the correct order. The simulation flow is:
            1. (External) Select video chunk quality/size based on policy
            2. Simulate network download -> delay
            3. Update playback buffer -> rebuf
            4. Handle buffer overflow -> sleep_time

        Args:
            video_chunk_size: Size of the video chunk to download in bytes

        Returns:
            TraceSimulateResult containing delay, rebuf, and sleep_time
        """
        # 1. Simulate network download
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L55-L87
        delay = self.download_chunk(video_chunk_size)

        # 2. Update playback buffer (compute rebuffer and add new chunk)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L89-L96
        rebuf = self.update_buffer(delay)

        # 3. Handle buffer overflow (sleep if buffer exceeds threshold)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L99-L123
        sleep_time = self.drain_buffer_overflow()

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L125-L129
        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.get_buffer_size()

        return TraceSimulateResult(
            delay=delay,
            rebuf=rebuf,
            sleep_time=sleep_time,
            buffer_size=return_buffer_size,
        )

"""TraceSimulator wrapper for modifying simulation behavior.

Design principles:
    - Subclasses use `super().method()` to call the wrapped simulator's method
    - Subclasses use `self.unwrapped` to access state variables on the base TraceSimulator
    - The wrapped simulator is private (`__base`) to enforce this pattern
"""

from .abc import AbstractTraceSimulator, TraceProgress
from .simulator import TraceSimulator


class TraceSimulatorWrapper(AbstractTraceSimulator):
    """Base wrapper class for modifying TraceSimulator behavior.

    Usage pattern for subclasses:
        - Override methods and call `super().method()` for delegation
        - Access state variables via `self.unwrapped.attribute`

    Example:
        class NoiseWrapper(TraceSimulatorWrapper):
            def download_chunk(self, video_chunk_size: int) -> float:
                delay = super().download_chunk(video_chunk_size)
                return delay * self.rng.uniform(0.9, 1.1)
    """

    def __init__(self, base: AbstractTraceSimulator):
        """Initialize the wrapper.

        Args:
            base: The simulator to wrap (TraceSimulator or another wrapper)
        """
        self.__base = base

    @property
    def unwrapped(self) -> TraceSimulator:
        """Get the underlying TraceSimulator for accessing state variables.

        All state (buffer_size, trace_idx, etc.) lives on the unwrapped simulator.
        """
        if isinstance(self.__base, TraceSimulatorWrapper):
            return self.__base.unwrapped
        # __base is TraceSimulator
        return self.__base  # type: ignore[return-value]

    # ==================== Delegated methods (use super() in subclasses) ====================

    def get_trace_progress(self) -> TraceProgress:
        """Get current trace progress information for logging."""
        return self.__base.get_trace_progress()

    def reset(self) -> None:
        """Reset the simulator state."""
        self.__base.reset()

    def on_video_finished(self) -> None:
        """Handle end of video."""
        self.__base.on_video_finished()

    def download_chunk(self, video_chunk_size: int) -> float:
        """Simulate downloading a video chunk. Returns delay in ms."""
        return self.__base.download_chunk(video_chunk_size)

    def update_buffer(self, delay: float) -> float:
        """Update playback buffer. Returns rebuffer time in ms."""
        return self.__base.update_buffer(delay)

    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer. Returns sleep time in ms."""
        return self.__base.drain_buffer_overflow()

    def get_buffer_size(self) -> float:
        """Get the current buffer size in milliseconds.

        Returns:
            Current buffer size in milliseconds
        """
        return self.__base.get_buffer_size()

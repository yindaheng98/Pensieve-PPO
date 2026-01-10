"""TraceSimulator wrapper for modifying simulation behavior."""

from typing import Protocol, runtime_checkable

from .simulator import TraceSimulator


@runtime_checkable
class SimulatorProtocol(Protocol):
    """Protocol defining the minimal simulator interface.

    This is the public API that external code should depend on.
    """

    def reset(self) -> None:
        """Reset the simulator state."""
        ...

    def download_chunk(self, video_chunk_size: int) -> float:
        """Simulate downloading a video chunk. Returns delay in ms."""
        ...

    def update_buffer(self, delay: float) -> float:
        """Update playback buffer. Returns rebuffer time in ms."""
        ...

    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer. Returns sleep time in ms."""
        ...

    def on_video_finished(self) -> None:
        """Handle end of video."""
        ...

    @property
    def trace_idx(self) -> int:
        """Current trace index (used for logging/identification)."""
        ...


class TraceSimulatorWrapper:
    """Base wrapper class using composition with explicit delegation.

    Subclasses can override methods to modify simulation behavior.
    Only exposes the minimal public API defined by SimulatorProtocol.
    """

    def __init__(self, base_simulator: SimulatorProtocol):
        """Initialize the wrapper.

        Args:
            base_simulator: The simulator to wrap (TraceSimulator or another wrapper)
        """
        self._base = base_simulator

    # ==================== Wrapper-specific ====================

    @property
    def base_simulator(self) -> TraceSimulator:
        """Get the underlying base TraceSimulator."""
        return self._base  # type: ignore[return-value]

    @property
    def unwrapped(self) -> TraceSimulator:
        """Get the underlying unwrapped TraceSimulator."""
        if isinstance(self._base, TraceSimulatorWrapper):
            return self._base.unwrapped
        return self._base  # type: ignore[return-value]

    # ==================== Public API (SimulatorProtocol) ====================

    def reset(self) -> None:
        """Reset the simulator state."""
        self.base_simulator.reset()

    def download_chunk(self, video_chunk_size: int) -> float:
        """Simulate downloading a video chunk. Returns delay in ms."""
        return self.base_simulator.download_chunk(video_chunk_size)

    def update_buffer(self, delay: float) -> float:
        """Update playback buffer. Returns rebuffer time in ms."""
        return self.base_simulator.update_buffer(delay)

    def drain_buffer_overflow(self) -> float:
        """Drain excess buffer. Returns sleep time in ms."""
        return self.base_simulator.drain_buffer_overflow()

    def on_video_finished(self) -> None:
        """Handle end of video."""
        self.base_simulator.on_video_finished()

    @property
    def trace_idx(self) -> int:
        """Current trace index (read-only, used for logging)."""
        return self.base_simulator.trace_idx

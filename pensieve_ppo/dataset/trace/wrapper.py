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
        self.base_simulator = base_simulator

    # ==================== Wrapper-specific ====================

    @property
    def unwrapped(self) -> TraceSimulator:
        """Get the underlying unwrapped TraceSimulator."""
        if isinstance(self.base_simulator, TraceSimulatorWrapper):
            return self.base_simulator.unwrapped
        return self.base_simulator  # type: ignore[return-value]

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

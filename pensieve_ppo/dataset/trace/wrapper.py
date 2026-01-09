"""TraceSimulator wrapper for modifying simulation behavior."""

from .simulator import TraceSimulator


class TraceSimulatorWrapper(TraceSimulator):
    """Base wrapper class that delegates to a wrapped TraceSimulator.

    Uses __getattr__ and __setattr__ for automatic delegation.
    Subclasses can override methods to modify simulation behavior.
    """

    def __init__(self, base_simulator: TraceSimulator):
        """Initialize the wrapper.

        Args:
            base_simulator: The TraceSimulator to wrap
        """
        self.base_simulator = base_simulator

    def __getattr__(self, name):
        """Delegate attribute access to wrapped simulator."""
        return getattr(self.base_simulator, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to wrapped simulator."""
        assert name != '_simulator', "Should not reassign _simulator"
        return setattr(self.base_simulator, name, value)

    # Methods that subclasses might override
    def reset(self, trace_idx: int = 0) -> None:
        self.base_simulator.reset(trace_idx)

    def download_chunk(self, video_chunk_size: int) -> float:
        return self.base_simulator.download_chunk(video_chunk_size)

    def update_buffer(self, delay: float) -> float:
        return self.base_simulator.update_buffer(delay)

    def drain_buffer_overflow(self) -> float:
        return self.base_simulator.drain_buffer_overflow()

    def on_video_finished(self) -> None:
        self.base_simulator.on_video_finished()

    @property
    def unwrapped(self) -> TraceSimulator:
        """Get the underlying unwrapped simulator."""
        if isinstance(self.base_simulator, TraceSimulatorWrapper):
            return self.base_simulator.unwrapped
        return self.base_simulator

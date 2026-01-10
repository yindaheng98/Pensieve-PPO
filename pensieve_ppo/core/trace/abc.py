"""Network trace data class."""

from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TraceData:
    """Container for loaded trace data."""
    all_cooked_time: List[List[float]]
    all_cooked_bw: List[List[float]]
    all_file_names: List[str]

    def __len__(self) -> int:
        return len(self.all_file_names)

    def __getitem__(self, idx: int) -> Tuple[List[float], List[float], str]:
        return (
            self.all_cooked_time[idx],
            self.all_cooked_bw[idx],
            self.all_file_names[idx]
        )


class AbstractTraceSimulator(ABC):
    """Abstract base class defining the TraceSimulator interface.

    Both TraceSimulator (concrete implementation) and TraceSimulatorWrapper
    inherit from this class to ensure a consistent interface.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulator state."""
        ...

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
    def on_video_finished(self) -> None:
        """Handle end of video."""
        ...

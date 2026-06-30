"""Video player for tracking playback position and chunk information."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VideoChunkRequest(ABC):
    """Base class for video chunk requests."""
    pass


class VideoPlayer(ABC):
    """Tracks video playback position and provides chunk information.

    This class manages the video chunk counter and provides methods
    to get chunk sizes and advance through the video.
    """

    def __init__(self):
        """Initialize the video player."""
        self.reset()

    def reset(self) -> None:
        """Reset the video playback position to the beginning.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L138
        """
        self.video_chunk_counter = 0

    @abstractmethod
    def get_chunk_quality(
        self,
        chunk_request: VideoChunkRequest,
        chunk_idx: int,
    ) -> float:
        """Get the actual quality level for a chunk request.

        Args:
            chunk_request: Request used to resolve the chunk quality.
            chunk_idx: Video chunk index.

        Returns:
            Quality value for the selected bitrate level.
        """
        ...

    @abstractmethod
    def get_chunk_size(
        self,
        chunk_request: VideoChunkRequest,
        chunk_idx: int,
    ) -> int:
        """Get the size of current chunk for a video chunk request.

        Args:
            chunk_request: Request used to resolve the chunk quality.
            chunk_idx: Video chunk index.

        Returns:
            Chunk size in bytes
        """
        ...

    def advance(self) -> Tuple[bool, int]:
        """Advance to the next video chunk.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L131

        Returns:
            Tuple of (end_of_video, remaining_chunks)
        """
        self.video_chunk_counter += 1
        video_chunk_remain = self.total_chunks - self.video_chunk_counter

        end_of_video = self.video_chunk_counter >= self.total_chunks

        return end_of_video, video_chunk_remain

    @property
    @abstractmethod
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        ...

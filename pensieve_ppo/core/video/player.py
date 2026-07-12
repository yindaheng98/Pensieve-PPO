"""Video player for tracking playback position and chunk information."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoChunkRequest(ABC):
    """Base class for video chunk requests."""
    pass


@dataclass(frozen=True)
class ResolvedChunk:
    """Chunk metadata resolved from a video chunk request."""
    size: int
    quality: float
    length: float


@dataclass(frozen=True)
class PlayerInfo:
    """Video player state returned after advancing playback."""
    resolved_chunk: ResolvedChunk
    end_of_video: bool
    remaining_chunks: int


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
        self.last_chunk_request: Optional[VideoChunkRequest] = None
        self.last_buffer_size: Optional[float] = None  # milliseconds
        self.video_chunk_counter = 0

    @abstractmethod
    def advance_chunk(
        self,
        chunk_request: VideoChunkRequest,
        chunk_idx: int,
        buffer_size: float,
    ) -> ResolvedChunk:
        """Advance chunk-specific state and return chunk metadata.

        Args:
            chunk_request: Request used to resolve the chunk quality.
            chunk_idx: Video chunk index.
            buffer_size: Current playback buffer size in milliseconds.

        Returns:
            Resolved chunk metadata.
        """
        ...

    def advance(
        self,
        chunk_request: VideoChunkRequest,
        buffer_size: float,
    ) -> PlayerInfo:
        """Advance to the next video chunk, recording the agent's decision.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/fixed_env.py#L131

        Args:
            chunk_request: The request that was used for the chunk that
                was just downloaded. Stored as :attr:`last_chunk_request`.
            buffer_size: Current playback buffer size in milliseconds.

        Returns:
            Resolved chunk metadata plus end-of-video progress information.
        """
        chunk_idx = self.video_chunk_counter
        resolved_chunk = self.advance_chunk(
            chunk_request,
            chunk_idx,
            buffer_size,
        )

        self.last_chunk_request = chunk_request
        self.last_buffer_size = buffer_size
        self.video_chunk_counter += 1
        video_chunk_remain = self.total_chunks - self.video_chunk_counter

        end_of_video = self.video_chunk_counter >= self.total_chunks

        return PlayerInfo(
            resolved_chunk=resolved_chunk,
            end_of_video=end_of_video,
            remaining_chunks=video_chunk_remain,
        )

    @property
    @abstractmethod
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        ...

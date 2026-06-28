"""Video player for tracking playback position and chunk information."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar, get_args, get_origin


@dataclass(frozen=True)
class VideoChunkRequest(ABC):
    """Base class for video chunk requests."""
    pass


VideoChunkRequestType = TypeVar("VideoChunkRequestType", bound=VideoChunkRequest)


class VideoChunkRequestTyped(ABC, Generic[VideoChunkRequestType]):
    """Interface for objects parameterized by a VideoChunkRequest type."""

    @property
    def request_cls(self) -> type[VideoChunkRequestType]:
        """Concrete request dataclass used by this object."""
        cls = type(self)
        for mro_cls in cls.__mro__:
            for base in getattr(mro_cls, "__orig_bases__", ()):
                origin = get_origin(base)
                if not isinstance(origin, type) or not issubclass(origin, VideoChunkRequestTyped):
                    continue
                args = get_args(base)
                if not args:
                    continue
                request_cls = args[0]
                if isinstance(request_cls, type) and issubclass(request_cls, VideoChunkRequest):
                    return request_cls

        raise TypeError(
            f"{cls.__name__} must declare a concrete VideoChunkRequest type via "
            "VideoChunkRequestTyped[RequestType]"
        )


class VideoPlayer(VideoChunkRequestTyped[VideoChunkRequestType]):
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
        chunk_request: VideoChunkRequestType,
        chunk_idx: Optional[int] = None,
    ) -> float:
        """Get the actual quality level for a chunk request.

        Args:
            chunk_request: Request used to resolve the chunk quality.
            chunk_idx: Video chunk index. Defaults to the current position.

        Returns:
            Quality value for the selected bitrate level.
        """
        ...

    @abstractmethod
    def get_chunk_size(
        self,
        chunk_request: VideoChunkRequestType,
        chunk_idx: Optional[int] = None,
    ) -> int:
        """Get the size of current chunk for a video chunk request.

        Args:
            chunk_request: Request used to resolve the chunk quality.
            chunk_idx: Video chunk index. Defaults to the current position.

        Returns:
            Chunk size in bytes
        """
        ...

    @abstractmethod
    def get_next_chunk_sizes(self) -> List[int]:
        """Get sizes of next chunk at all quality levels.

        Returns:
            List of chunk sizes in bytes for each bitrate level
        """
        ...

    @abstractmethod
    def get_next_chunk_qualities(self) -> List[float]:
        """Get qualities of next chunk at all quality levels."""
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
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels."""
        ...

    @property
    @abstractmethod
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        ...

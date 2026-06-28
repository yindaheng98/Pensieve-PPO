"""Envivio video player implementation."""

from typing import List, Optional

from ..player import VideoChunkRequest, VideoPlayer
from .data import EnvivioVideoData


class EnvivioVideoPlayer(VideoPlayer):
    """Video player backed by Envivio video chunk size files."""

    def __init__(self, *args, **kwargs):
        """Initialize the Envivio video player.

        Args:
            *args: Positional arguments passed to EnvivioVideoData.
            **kwargs: Keyword arguments passed to EnvivioVideoData.
        """
        self._video_data = EnvivioVideoData(*args, **kwargs)
        super().__init__()

    def get_chunk_quality(
        self,
        chunk_request: VideoChunkRequest,
        chunk_idx: Optional[int] = None,
    ) -> float:
        """Get the actual quality level for a chunk request."""
        return self._video_data.get_chunk_quality(chunk_request.level, chunk_idx or self.video_chunk_counter)

    def get_chunk_size(
        self,
        chunk_request: VideoChunkRequest,
        chunk_idx: Optional[int] = None,
    ) -> int:
        """Get the size of current chunk for a video chunk request.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L54
        """
        return self._video_data.get_chunk_size(chunk_request.level, chunk_idx or self.video_chunk_counter)

    def get_next_chunk_sizes(self) -> List[int]:
        """Get sizes of next chunk at all quality levels.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L152-L154
        """
        return self._video_data.get_next_chunk_sizes(self.video_chunk_counter)

    def get_next_chunk_qualities(self) -> List[float]:
        """Get qualities of next chunk at all quality levels."""
        return self._video_data.get_next_chunk_qualities(self.video_chunk_counter)

    @property
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels."""
        return self._video_data.bitrate_levels

    @property
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        return self._video_data.total_chunks

"""Quality ladder video player implementation."""

from dataclasses import dataclass
from typing import List, Optional

from ..player import VideoChunkRequest, VideoPlayer
from .envivio import load_video_size


# From src/env.py
VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]  # Kbps, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L13


@dataclass(frozen=True)
class QualityLadderRequest(VideoChunkRequest):
    """Request for a video chunk at a quality ladder level."""
    level: int


class QualityLadderVideoPlayer(VideoPlayer):
    """Video player backed by quality-ladder video chunk size data."""

    def __init__(self, quality: List[float] = VIDEO_BIT_RATE, *args, **kwargs):
        """Initialize the quality ladder video player.

        Args:
            quality: Quality metric list for each bitrate level.
            *args: Positional arguments passed to load_video_size.
            **kwargs: Keyword arguments passed to load_video_size.
        """
        self.quality = list(quality)
        # Video-size loading follows the original src/core.py fixed_env data layout.
        self.video_size = load_video_size(*args, **kwargs)
        if self.video_size.shape[0] != len(self.quality):
            raise ValueError(
                "video_size bitrate dimension "
                f"({self.video_size.shape[0]}) must match "
                f"quality length ({len(self.quality)})"
            )
        super().__init__()

    def get_chunk_quality(
        self,
        chunk_request: QualityLadderRequest,
        chunk_idx: Optional[int] = None,
    ) -> float:
        """Get the actual quality level for a chunk request."""
        return self.quality[chunk_request.level]

    def get_chunk_size(
        self,
        chunk_request: QualityLadderRequest,
        chunk_idx: Optional[int] = None,
    ) -> int:
        """Get the size of current chunk for a video chunk request.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L54
        """
        return int(
            self.video_size[
                chunk_request.level,
                chunk_idx or self.video_chunk_counter,
            ]
        )

    def get_next_chunk_sizes(self) -> List[int]:
        """Get sizes of next chunk at all quality levels.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L152-L154
        """
        return self.video_size[:, self.video_chunk_counter].tolist()

    def get_next_chunk_qualities(self) -> List[float]:
        """Get qualities of next chunk at all quality levels."""
        return list(self.quality)

    @property
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels."""
        return self.video_size.shape[0]

    @property
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        return self.video_size.shape[1]

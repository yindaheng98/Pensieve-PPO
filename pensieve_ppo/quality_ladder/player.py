"""Quality ladder video player implementation."""

from typing import List, Optional

from ..core.video.player import VideoPlayer
from .abc import QualityLadderLoader, QualityLadderRequest
from .envivio import load_envivio_video_size


LOAD_VIDEO_SIZE: dict[str, QualityLadderLoader] = {
    "envivio": load_envivio_video_size,
}


class QualityLadderVideoPlayer(VideoPlayer):
    """Video player backed by quality-ladder video chunk size data."""

    def __init__(self, name: str = "envivio", *args, **kwargs):
        """Initialize the quality ladder video player.

        Args:
            name: Registered quality ladder loader name.
            *args: Positional arguments passed to the selected loader.
            **kwargs: Keyword arguments passed to the selected loader.
        """
        # Video-size loading follows the original src/core.py fixed_env data layout.
        data = LOAD_VIDEO_SIZE[name](*args, **kwargs)
        self.video_size = data.video_size
        self.video_quality = data.video_quality
        if self.video_size.shape != self.video_quality.shape:
            raise ValueError(
                f"video_size shape {self.video_size.shape} must match "
                f"video_quality shape {self.video_quality.shape}"
            )
        super().__init__()

    def get_chunk_quality(
        self,
        chunk_request: QualityLadderRequest,
        chunk_idx: Optional[int] = None,
    ) -> float:
        """Get the actual quality level for a chunk request."""
        return float(
            self.video_quality[
                chunk_request.level,
                chunk_idx or self.video_chunk_counter,
            ]
        )

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
        return self.video_quality[:, self.video_chunk_counter].tolist()

    @property
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels."""
        return self.video_size.shape[0]

    @property
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        return self.video_size.shape[1]

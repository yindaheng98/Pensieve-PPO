"""Video chunk size data class."""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .loader import load_video_size


# From src/env.py
VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]  # Kbps, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L13


class EnvivioVideoData:
    """Container for loaded video chunk size data."""

    def __init__(
        self,
        video_size_file_prefix: str,
        bitrate_levels: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ):
        """Initialize Envivio video data from video size files.

        Args:
            video_size_file_prefix: Path prefix for video size files
            bitrate_levels: Number of bitrate levels. If None, auto-detect by
                       finding the maximum bitrate level with existing files.
            max_chunks: Maximum number of chunks to load. If specified, truncates
                   the loaded data to this limit. If None, load all chunks.
        """

        video_size: NDArray[np.int64] = load_video_size(
            video_size_file_prefix,
            bitrate_levels=bitrate_levels,
            max_chunks=max_chunks,
        )  # shape: [bitrate_levels, total_chunks]

        # Validate loaded video data matches the fixed quality ladder.
        if video_size.shape[0] != len(VIDEO_BIT_RATE):
            raise ValueError(
                "video_size bitrate dimension "
                f"({video_size.shape[0]}) must match "
                f"VIDEO_BIT_RATE length ({len(VIDEO_BIT_RATE)})"
            )

        self.video_size = video_size

    @property
    def bitrate_levels(self) -> int:
        """Number of bitrate levels."""
        return self.video_size.shape[0]

    @property
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        return self.video_size.shape[1]

    def get_chunk_size(self, level: int, chunk_idx: int) -> int:
        """Get size of a specific chunk at given bitrate level."""
        return int(self.video_size[level, chunk_idx])

    def get_chunk_quality(self, level: int, chunk_idx: int) -> float:
        """Get quality of a specific chunk at given bitrate level."""
        return VIDEO_BIT_RATE[level]

    def get_next_chunk_sizes(self, chunk_idx: int) -> List[int]:
        """Get sizes of next chunk for all bitrate levels."""
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L155-L157
        return self.video_size[:, chunk_idx].tolist()

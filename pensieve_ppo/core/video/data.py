"""Video chunk size data class."""

from typing import List
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class VideoData:
    """Container for loaded video chunk size data."""
    video_size: NDArray[np.int64]  # shape: [bitrate_levels, total_chunks]

    @property
    def bitrate_levels(self) -> int:
        """Number of bitrate levels."""
        return self.video_size.shape[0]

    @property
    def total_chunks(self) -> int:
        """Total number of video chunks."""
        return self.video_size.shape[1]

    def get_chunk_size(self, bitrate: int, chunk_idx: int) -> int:
        """Get size of a specific chunk at given bitrate."""
        return int(self.video_size[bitrate, chunk_idx])

    def get_next_chunk_sizes(self, chunk_idx: int) -> List[int]:
        """Get sizes of next chunk for all bitrate levels."""
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L155-L157
        return self.video_size[:, chunk_idx].tolist()

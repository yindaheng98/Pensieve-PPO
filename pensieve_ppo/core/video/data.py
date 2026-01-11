"""Video chunk size data class."""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class VideoData:
    """Container for loaded video chunk size data."""
    video_size: Dict[int, List[int]]  # bitrate_level -> list of chunk sizes
    bitrate_levels: int
    num_chunks: int

    def get_chunk_size(self, bitrate: int, chunk_idx: int) -> int:
        """Get size of a specific chunk at given bitrate."""
        return self.video_size[bitrate][chunk_idx]

    def get_next_chunk_sizes(self, chunk_idx: int) -> List[int]:
        """Get sizes of next chunk for all bitrate levels."""
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L152-L154
        return [self.get_chunk_size(i, chunk_idx) for i in range(self.bitrate_levels)]

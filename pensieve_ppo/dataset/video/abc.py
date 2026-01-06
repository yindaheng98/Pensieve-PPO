"""Video chunk size data class."""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class VideoData:
    """Container for loaded video chunk size data."""
    video_size: Dict[int, List[int]]  # bitrate_level -> list of chunk sizes
    bitrate_levels: int
    num_chunks: int

    def __len__(self) -> int:
        return self.num_chunks

    def get_chunk_size(self, bitrate: int, chunk_idx: int) -> int:
        """Get size of a specific chunk at given bitrate."""
        return self.video_size[bitrate][chunk_idx]

    def get_next_chunk_sizes(self, chunk_idx: int) -> List[int]:
        """Get sizes of next chunk for all bitrate levels."""
        return [self.video_size[i][chunk_idx] for i in range(self.bitrate_levels)]

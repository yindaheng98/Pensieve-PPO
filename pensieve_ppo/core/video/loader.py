"""Video chunk size data loader."""

import os
from typing import Optional

import numpy as np

from .data import VideoData


def load_video_size(
    video_size_file_prefix: str,
    bitrate_levels: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> VideoData:
    """
    Load video chunk sizes for all bitrate levels.

    Video size files should be named as:
    - {prefix}0, {prefix}1, ..., {prefix}{bitrate_levels-1}

    Each file contains one chunk size per line (in bytes).

    Args:
        video_size_file_prefix: Path prefix for video size files
                               (e.g., './envivio/video_size_')
        bitrate_levels: Number of bitrate levels. If None, auto-detect by
                       finding the maximum bitrate level with existing files.
        max_chunks: Maximum number of chunks to load. If specified, truncates
                   the loaded data to this limit. If None, load all chunks.

    Returns:
        VideoData object containing chunk sizes for all bitrates
    """
    # Auto-detect bitrate_levels if not specified
    if bitrate_levels is None:
        bitrate_levels = 0
        while os.path.exists(f"{video_size_file_prefix}{bitrate_levels}"):
            bitrate_levels += 1
        if bitrate_levels == 0:
            raise FileNotFoundError(
                f"No video size files found with prefix: {video_size_file_prefix}"
            )

    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L42
    video_size_lists = []
    for bitrate in range(bitrate_levels):
        sizes = []
        with open(f"{video_size_file_prefix}{bitrate}", 'r') as f:
            for line in f:
                sizes.append(int(line.split()[0]))
        video_size_lists.append(sizes)

    total_chunks = min(len(sizes) for sizes in video_size_lists)

    # Truncate to max_chunks if specified
    if max_chunks is not None:
        total_chunks = min(max_chunks, total_chunks)

    # Create matrix: [bitrate_levels, total_chunks]
    video_size = np.array(
        [sizes[:total_chunks] for sizes in video_size_lists],
        dtype=np.int64,
    )

    return VideoData(video_size=video_size)

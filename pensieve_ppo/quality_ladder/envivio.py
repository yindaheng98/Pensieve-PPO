"""Envivio-backed quality ladder data loader."""

import os
from typing import Optional

import numpy as np

from .abc import QualityLadderData


# From src/core.py
TOTAL_VIDEO_CHUNKS = 48  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L9
VIDEO_SIZE_FILE_PREFIX = './src/envivio/video_size_'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L17

# From src/env.py
VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]  # Kbps, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L13
DEFAULT_QUALITY = 1  # default video quality without agent, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L19


def load_envivio_video_size(
    video_size_file_prefix: str = VIDEO_SIZE_FILE_PREFIX,
    max_chunks: Optional[int] = TOTAL_VIDEO_CHUNKS,
    quality: list[float] = VIDEO_BIT_RATE,
) -> QualityLadderData:
    """
    Load video chunk sizes and quality values for all bitrate levels.

    Video size files should be named as:
    - {prefix}0, {prefix}1, ..., {prefix}{bitrate_levels-1}

    Each file contains one chunk size per line (in bytes).

    Args:
        video_size_file_prefix: Path prefix for video size files
                               (e.g., './envivio/video_size_')
        max_chunks: Maximum number of chunks to load. If specified, truncates
                   the loaded data to this limit. If None, load all chunks.
        quality: Quality metric list for each bitrate level.

    Returns:
        QualityLadderData with matching size and quality matrices.
    """
    # Auto-detect bitrate_levels
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

    # Create matrices: [bitrate_levels, total_chunks]
    video_size = np.array(
        [sizes[:total_chunks] for sizes in video_size_lists],
        dtype=np.int64,
    )
    video_quality = np.broadcast_to(
        np.asarray(quality, dtype=np.float64)[:, None],
        video_size.shape,
    ).copy()

    return QualityLadderData(
        video_size=video_size,
        video_quality=video_quality,
    )

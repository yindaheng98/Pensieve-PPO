"""Video chunk size data loader."""

from typing import Optional

from .data import VideoData


def load_video_size(
    video_size_file_prefix: str,
    bitrate_levels: int = 6,
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
        bitrate_levels: Number of bitrate levels (default: 6)
        max_chunks: Maximum number of chunks to load. If specified, truncates
                   the loaded data to this limit. If None, load all chunks.

    Returns:
        VideoData object containing chunk sizes for all bitrates
    """
    # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L42
    video_size = {}
    for bitrate in range(bitrate_levels):
        video_size[bitrate] = []
        with open(f"{video_size_file_prefix}{bitrate}", 'r') as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))

    total_chunks = min([len(video_size[bitrate]) for bitrate in range(bitrate_levels)])

    # Truncate to max_chunks if specified
    if max_chunks is not None:
        total_chunks = min(max_chunks, total_chunks)

    for bitrate in range(bitrate_levels):
        video_size[bitrate] = video_size[bitrate][:total_chunks]

    return VideoData(
        video_size=video_size,
        bitrate_levels=bitrate_levels,
        total_chunks=total_chunks,
    )

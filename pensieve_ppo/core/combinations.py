"""Convenience functions for creating combined Simulator configurations."""

from typing import Optional

from .simulator import Simulator
from .trace import load_trace
from .trace.ext import create_train_simulator, create_test_simulator
from .video import VideoPlayer, load_video_size


def create_simulator(
    trace_folder: str,
    video_size_file_prefix: str,
    bitrate_levels: Optional[int] = None,
    max_chunks: Optional[int] = None,
    train: bool = True,
    random_seed: Optional[int] = None,
) -> Simulator:
    """Create a Simulator configured for training or testing.

    Args:
        trace_folder: Path to folder containing network trace files
        video_size_file_prefix: Path prefix for video size files
                               (e.g., './envivio/video_size_')
        bitrate_levels: Number of bitrate levels. If None, auto-detect by
                       finding the maximum bitrate level with existing files.
        max_chunks: Maximum number of video chunks to load. If specified,
                   truncates the loaded data to this limit. If None, load all.
        train: If True, use training simulator with noise and random trace
               selection. If False, use deterministic test simulator.
        random_seed: Random seed for reproducibility. If None, uses global
                    np.random state.

    Returns:
        Configured Simulator instance
    """
    # Load trace data
    trace_data = load_trace(trace_folder)

    # Load video data
    video_data = load_video_size(
        video_size_file_prefix,
        bitrate_levels=bitrate_levels,
        max_chunks=max_chunks,
    )

    # Create video player
    video_player = VideoPlayer(video_data)

    # Create trace simulator based on training mode
    if train:
        trace_simulator = create_train_simulator(trace_data, random_seed=random_seed)
    else:
        trace_simulator = create_test_simulator(trace_data)

    return Simulator(video_player, trace_simulator)

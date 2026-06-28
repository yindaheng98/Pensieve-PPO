"""Convenience functions for creating combined Simulator configurations."""

from typing import Any, Optional

from .simulator import Simulator
from .trace import load_trace
from .trace.ext import create_train_simulator, create_test_simulator
from .video import VideoChunkRequestType, create_video_player


def create_simulator(
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
    video_player_name: str = 'envivio',
    **video_configs: Any,
) -> Simulator[VideoChunkRequestType]:
    """Create a Simulator configured for training or testing.

    Args:
        trace_folder: Path to folder containing network trace files
        train: If True, use training simulator with noise and random trace
               selection. If False, use deterministic test simulator.
        random_seed: Random seed for reproducibility. If None, uses global
                    np.random state.
        video_player_name: Registered video player name.
        **video_configs: Keyword arguments passed to the video player.

    Returns:
        Configured Simulator instance
    """
    # Load trace data
    trace_data = load_trace(trace_folder)

    # Create video player
    video_player = create_video_player(video_player_name, **video_configs)

    # Create trace simulator based on training mode
    if train:
        trace_simulator = create_train_simulator(trace_data, random_seed=random_seed)
    else:
        trace_simulator = create_test_simulator(trace_data)

    return Simulator(video_player, trace_simulator)

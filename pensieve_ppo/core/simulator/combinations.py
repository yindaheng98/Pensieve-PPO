"""Convenience functions for creating Simulator configurations."""

from typing import Optional

from .simulator import Simulator
from ..trace import load_trace
from ..trace.ext import create_train_simulator, create_test_simulator
from ..video import VideoPlayer


def create_simulator(
    trace_folder: str,
    video_player: VideoPlayer,
    train: bool = True,
    random_seed: Optional[int] = None,
) -> Simulator:
    """Create a Simulator configured for training or testing.

    Args:
        trace_folder: Path to folder containing network trace files
        video_player: Pre-configured video player instance.
        train: If True, use training simulator with noise and random trace
               selection. If False, use deterministic test simulator.
        random_seed: Random seed for reproducibility. If None, uses global
                    np.random state.

    Returns:
        Configured Simulator instance
    """
    # Load trace data
    trace_data = load_trace(trace_folder)

    # Create trace simulator based on training mode
    if train:
        trace_simulator = create_train_simulator(trace_data, random_seed=random_seed)
    else:
        trace_simulator = create_test_simulator(trace_data)

    return Simulator(video_player, trace_simulator)

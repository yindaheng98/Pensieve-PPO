"""Convenience functions for creating RL ABR environments.

This module provides factory functions for creating ABREnv instances
configured with RL-specific observers and reward structures.
"""


from ...gym import ABREnv, create_env
from .observer import (
    RLABRStateObserver,
    REBUF_PENALTY,
    SMOOTH_PENALTY,
    S_LEN,
    BUFFER_NORM_FACTOR,
)


def create_rl_env(
    levels_quality: list,
    trace_folder: str,
    video_size_file_prefix: str,
    max_chunks: int,
    train: bool = True,
    # Observer parameters
    rebuf_penalty: float = REBUF_PENALTY,
    smooth_penalty: float = SMOOTH_PENALTY,
    state_history_len: int = S_LEN,
    buffer_norm_factor: float = BUFFER_NORM_FACTOR,
    # Env parameters
    initial_level: int = 0,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv with RL-specific state observer.

    This function creates an ABREnv configured with RLABRStateObserver,
    which provides the state representation and reward calculation
    for reinforcement learning based ABR agents.

    Args:
        levels_quality: Quality metric list for each bitrate level.
        trace_folder: Path to network trace folder.
        video_size_file_prefix: Prefix for video size files.
        max_chunks: Maximum number of video chunks.
        train: Whether in training mode (affects trace iteration).
        rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3).
        smooth_penalty: Penalty coefficient for quality changes (default: 1.0).
        state_history_len: Number of past observations in state (default: 8).
        buffer_norm_factor: Normalization factor for buffer size (default: 10.0).
        initial_level: Initial quality level index on reset.
        **kwargs: Additional arguments passed to create_env/create_simulator.

    Returns:
        Configured ABREnv instance with RLABRStateObserver.
    """
    # Create the RL observer with specified parameters
    observer = RLABRStateObserver(
        levels_quality=levels_quality,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
        state_history_len=state_history_len,
        buffer_norm_factor=buffer_norm_factor,
    )

    return create_env(
        observer=observer,
        initial_level=initial_level,
        trace_folder=trace_folder,
        video_size_file_prefix=video_size_file_prefix,
        bitrate_levels=len(levels_quality),
        max_chunks=max_chunks,
        train=train,
        **kwargs,
    )

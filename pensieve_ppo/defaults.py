"""Default parameter combinations for ABR environment.

This module provides default parameter values and a convenience function
for creating ABREnv instances with Pensieve-PPO defaults.
"""

from typing import Optional

from .gym.combinations import create_env
from .gym.env import ABREnv


# Default constants from original Pensieve-PPO implementation
# Source: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/

# From src/core.py
TOTAL_VIDEO_CHUNKS = 48  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L9
VIDEO_SIZE_FILE_PREFIX = './envivio/video_size_'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L17

# From src/env.py
VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]  # Kbps, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L13

# From src/load_trace.py and src/test.py
TRAIN_TRACES = './train/'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/load_trace.py#L4
TEST_TRACES = './test/'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L26


def create_env_with_default(
    levels_quality: list = VIDEO_BIT_RATE,
    trace_folder: Optional[str] = None,
    video_size_file_prefix: str = VIDEO_SIZE_FILE_PREFIX,
    max_chunks: int = TOTAL_VIDEO_CHUNKS,
    train: bool = True,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv with default Pensieve parameters.

    This is a convenience function that wraps create_env with default values
    matching the original Pensieve implementation.

    Args:
        levels_quality: Quality metric list for each bitrate level
                       (default: VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]).
        trace_folder: Path to folder containing network trace files.
                     If None, auto-selects based on train (TRAIN_TRACES or TEST_TRACES).
        video_size_file_prefix: Path prefix for video size files
                               (default: './envivio/video_size_').
        max_chunks: Maximum number of video chunks (default: 48).
        train: If True, use training simulator with noise and random trace
               selection. If False, use deterministic test simulator (default: True).
        **kwargs: Additional arguments passed to create_env (e.g., random_seed,
                 rebuf_penalty, smooth_penalty, state_history_len, buffer_norm_factor,
                 initial_level).

    Returns:
        Configured ABREnv instance with default Pensieve parameters.

    Example:
        >>> # Create training environment with all defaults
        >>> env = create_env_with_default()
        >>>
        >>> # Create test environment
        >>> env = create_env_with_default(train=False)
        >>>
        >>> # Create environment with custom random seed
        >>> env = create_env_with_default(random_seed=42)
    """
    if trace_folder is None:
        trace_folder = TRAIN_TRACES if train else TEST_TRACES

    return create_env(
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        video_size_file_prefix=video_size_file_prefix,
        max_chunks=max_chunks,
        train=train,
        **kwargs,
    )

"""Default parameter combinations for ABR environment.

This module provides default parameter values that can be used when creating
ABREnv instances for training, testing, or evaluation.

All constants in this module are derived from the original Pensieve-PPO implementation:
https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/

These values are used as default parameters for create_env_with_default() and are
verified against the original implementation in tests/test_env_equivalence.py.
"""

from typing import Optional

import numpy as np

from .combinations import create_env
from .env import ABREnv


# ==============================================================================
# Default constants from original Pensieve-PPO implementation
# Source: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/
# ==============================================================================

# ------------------------------------------------------------------------------
# From src/core.py
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L8-L9
# ------------------------------------------------------------------------------
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNKS = 48

# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L17
VIDEO_SIZE_FILE_PREFIX = './envivio/video_size_'

# ------------------------------------------------------------------------------
# From src/env.py
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L8-L10
# ------------------------------------------------------------------------------
S_INFO = 6
S_LEN = 8
A_DIM = 6

# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L13
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps

# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14-L16
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0

# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L17-L20
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# From src/load_trace.py
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/load_trace.py#L4
# ------------------------------------------------------------------------------
TRAIN_TRACES = './train/'

# ------------------------------------------------------------------------------
# From src/test.py
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L26
# ------------------------------------------------------------------------------
TEST_TRACES = './test/'


def create_env_with_default(
    levels_quality: Optional[list] = None,
    trace_folder: str = TRAIN_TRACES,
    video_size_file_prefix: str = VIDEO_SIZE_FILE_PREFIX,
    bitrate_levels: int = BITRATE_LEVELS,
    max_chunks: int = TOTAL_VIDEO_CHUNKS,
    train: bool = True,
    random_seed: Optional[int] = None,
    rebuf_penalty: float = REBUF_PENALTY,
    smooth_penalty: float = SMOOTH_PENALTY,
    state_history_len: int = S_LEN,
    buffer_norm_factor: float = BUFFER_NORM_FACTOR,
    initial_bitrate: int = DEFAULT_QUALITY,
) -> ABREnv:
    """Create an ABREnv with default Pensieve parameters.

    This is a convenience function that wraps create_env with default values
    matching the original Pensieve implementation.

    Args:
        levels_quality: Quality metric list for each bitrate level. If None,
                       uses VIDEO_BIT_RATE (default: [300, 750, 1200, 1850, 2850, 4300]).
        trace_folder: Path to folder containing network trace files
                     (default: './train/').
        video_size_file_prefix: Path prefix for video size files
                               (default: './envivio/video_size_').
        bitrate_levels: Number of bitrate levels (default: 6).
        max_chunks: Maximum number of video chunks (default: 48).
        train: If True, use training simulator with noise and random trace
               selection. If False, use deterministic test simulator (default: True).
        random_seed: Random seed for reproducibility. If None, uses global
                    np.random state.
        rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3).
        smooth_penalty: Penalty coefficient for quality changes (default: 1).
        state_history_len: Number of past observations to keep in state (default: 8).
        buffer_norm_factor: Normalization factor for buffer size in seconds (default: 10.0).
        initial_bitrate: Initial bitrate level index on reset (default: 1).

    Returns:
        Configured ABREnv instance with default Pensieve parameters.

    Example:
        >>> # Create training environment with all defaults
        >>> env = create_env_with_default()
        >>>
        >>> # Create test environment
        >>> env = create_env_with_default(trace_folder='./test/', train=False)
        >>>
        >>> # Create environment with custom random seed
        >>> env = create_env_with_default(random_seed=42)
    """
    if levels_quality is None:
        levels_quality = VIDEO_BIT_RATE.tolist()

    return create_env(
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        video_size_file_prefix=video_size_file_prefix,
        bitrate_levels=bitrate_levels,
        max_chunks=max_chunks,
        train=train,
        random_seed=random_seed,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
        state_history_len=state_history_len,
        buffer_norm_factor=buffer_norm_factor,
        initial_bitrate=initial_bitrate,
    )

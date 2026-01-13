"""Convenience functions for creating ABR gymnasium environments."""

from typing import List

from ..core import create_simulator
from .env import ABREnv, REBUF_PENALTY, SMOOTH_PENALTY, S_LEN, BUFFER_NORM_FACTOR


def create_env(
    levels_quality: List[float],
    rebuf_penalty: float = REBUF_PENALTY,
    smooth_penalty: float = SMOOTH_PENALTY,
    state_history_len: int = S_LEN,
    buffer_norm_factor: float = BUFFER_NORM_FACTOR,
    initial_level: int = 0,
    **simulator_kwargs,
) -> ABREnv:
    """Create an ABREnv with a configured Simulator.

    Args:
        levels_quality: Quality metric list for each bitrate level, used for
                       state representation and reward calculation (e.g., bitrate
                       values in Kbps: [300, 750, 1200, ...]).
        rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3)
        smooth_penalty: Penalty coefficient for quality changes (default: 1.0)
        state_history_len: Number of past observations to keep in state (default: 8)
        buffer_norm_factor: Normalization factor for buffer size in seconds (default: 10.0)
        initial_level: Initial quality level index on reset (default: 0)
        **simulator_kwargs: Arguments passed to create_simulator (trace_folder,
                           video_size_file_prefix, max_chunks, train, random_seed).
                           Note: bitrate_levels is automatically set to len(levels_quality).

    Returns:
        Configured ABREnv instance
    """
    simulator = create_simulator(bitrate_levels=len(levels_quality), **simulator_kwargs)

    return ABREnv(
        simulator=simulator,
        levels_quality=levels_quality,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
        state_history_len=state_history_len,
        buffer_norm_factor=buffer_norm_factor,
        initial_level=initial_level,
    )

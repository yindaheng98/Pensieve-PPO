"""Convenience functions for creating ABR gymnasium environments."""

from ..core import create_simulator
from .env import ABREnv, AbstractABRStateObserver


def create_env(
    observer: AbstractABRStateObserver,
    initial_level: int = 0,
    **simulator_kwargs,
) -> ABREnv:
    """Create an ABREnv with a configured Simulator.

    Args:
        observer: ABRStateObserver instance for state observation and reward
                 calculation. Its bitrate_levels property determines the
                 number of bitrate levels for the simulator.
        initial_level: Initial quality level index on reset (default: 0)
        **simulator_kwargs: Arguments passed to create_simulator (trace_folder,
                           video_size_file_prefix, max_chunks, train, random_seed).
                           Note: bitrate_levels is automatically set from observer.

    Returns:
        Configured ABREnv instance
    """
    simulator = create_simulator(
        bitrate_levels=observer.bitrate_levels,
        **simulator_kwargs
    )

    return ABREnv(
        simulator=simulator,
        observer=observer,
        initial_level=initial_level,
    )

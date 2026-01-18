"""Convenience functions for creating ABR gymnasium environments."""

from ..core import create_simulator
from .env import ABREnv, AbstractABRStateObserver


def create_env(
    observer: AbstractABRStateObserver,
    *args,
    initial_level: int = 0,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv with a configured Simulator.

    Args:
        observer: ABRStateObserver instance for state observation and reward.
        *args: Positional arguments passed to create_simulator.
        initial_level: Initial quality level index on reset (default: 0).
        **kwargs: Keyword arguments passed to create_simulator.

    Returns:
        Configured ABREnv instance.
    """
    simulator = create_simulator(*args, **kwargs)

    return ABREnv(
        simulator=simulator,
        observer=observer,
        initial_level=initial_level,
    )

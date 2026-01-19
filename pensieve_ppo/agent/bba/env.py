"""Convenience functions for creating BBA ABR environments.

This module provides factory functions for creating ABREnv instances
configured with BBA-specific observers.
"""

from ...gym import ABREnv
from ..rl.env import create_rl_env_with_observer_cls
from .observer import BBAStateObserver


def create_bba_env(*args, **kwargs) -> ABREnv:
    """Create an ABREnv with BBA-specific state observer.

    This function creates an ABREnv configured with BBAStateObserver,
    which provides a simplified state representation (buffer_size only)
    suitable for Buffer-Based Adaptive (BBA) streaming algorithms.

    This is a convenience wrapper around create_rl_env_with_observer_cls
    with observer_cls=BBAStateObserver. See that function for full
    parameter documentation.

    Args:
        *args: Positional arguments passed to create_rl_env_with_observer_cls.
        **kwargs: Keyword arguments passed to create_rl_env_with_observer_cls.
            Common parameters include:
            - levels_quality: Quality metric list for each bitrate level.
            - train: Whether in training mode (affects trace iteration).
            - rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3).
            - smooth_penalty: Penalty coefficient for quality changes (default: 1.0).
            - initial_level: Initial quality level index on reset.

    Returns:
        Configured ABREnv instance with BBAStateObserver.
    """
    return create_rl_env_with_observer_cls(
        observer_cls=BBAStateObserver,
        *args,
        **kwargs,
    )

"""Convenience functions for creating MPC ABR environments.

This module provides factory functions for creating ABREnv instances
configured with MPC-specific observers.
"""

from ...gym import ABREnv
from ..rl.env import create_rl_env_with_observer_cls
from .observer import MPCABRStateObserver


def create_mpc_env(*args, **kwargs) -> ABREnv:
    """Create an ABREnv with MPC-specific state observer.

    This function creates an ABREnv configured with MPCABRStateObserver,
    which provides a PredictionState that includes methods for computing
    future download times, enabling the MPC algorithm to plan ahead using
    actual future bandwidth information.

    This is a convenience wrapper around create_rl_env_with_observer_cls
    with observer_cls=MPCABRStateObserver. See that function for full
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
        Configured ABREnv instance with MPCABRStateObserver.
    """
    return create_rl_env_with_observer_cls(
        observer_cls=MPCABRStateObserver,
        *args,
        **kwargs,
    )

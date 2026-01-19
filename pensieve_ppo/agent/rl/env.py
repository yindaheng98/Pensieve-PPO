"""Convenience functions for creating RL ABR environments.

This module provides factory functions for creating ABREnv instances
configured with RL-specific observers and reward structures.
"""

from typing import Type

from ...gym import ABREnv, create_env
from .observer import (
    RLABRStateObserver,
    REBUF_PENALTY,
    SMOOTH_PENALTY,
    S_LEN,
    BUFFER_NORM_FACTOR,
)


def create_rl_env_with_observer_cls(
    observer_cls: Type[RLABRStateObserver],
    levels_quality: list,
    *args,
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
    """Create an ABREnv with a custom RLABRStateObserver subclass.

    This function creates an ABREnv configured with a user-specified observer
    class that must be a subclass of RLABRStateObserver. This allows for custom
    state representations and reward calculations while maintaining compatibility
    with RL-based ABR agents.

    Args:
        observer_cls: Observer class to use, must be RLABRStateObserver or a subclass.
        levels_quality: Quality metric list for each bitrate level.
        train: Whether in training mode (affects trace iteration).
        rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3).
        smooth_penalty: Penalty coefficient for quality changes (default: 1.0).
        state_history_len: Number of past observations in state (default: 8).
        buffer_norm_factor: Normalization factor for buffer size (default: 10.0).
        initial_level: Initial quality level index on reset.
        **kwargs: Additional arguments passed to create_env/create_simulator.

    Returns:
        Configured ABREnv instance with the specified observer.

    Raises:
        TypeError: If observer_cls is not a subclass of RLABRStateObserver.
    """
    # Validate that observer_cls is a subclass of RLABRStateObserver
    if not (isinstance(observer_cls, type) and issubclass(observer_cls, RLABRStateObserver)):
        raise TypeError(
            f"observer_cls must be RLABRStateObserver or a subclass, "
            f"got {observer_cls!r}"
        )

    # Create the observer with specified parameters
    observer = observer_cls(
        levels_quality=levels_quality,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
        state_history_len=state_history_len,
        buffer_norm_factor=buffer_norm_factor,
    )

    return create_env(
        observer,
        *args,
        initial_level=initial_level,
        bitrate_levels=len(levels_quality),
        train=train,
        **kwargs,
    )


def create_rl_env(*args, **kwargs) -> ABREnv:
    """Create an ABREnv with RL-specific state observer.

    This function creates an ABREnv configured with RLABRStateObserver,
    which provides the state representation and reward calculation
    for reinforcement learning based ABR agents.

    This is a convenience wrapper around create_rl_env_with_observer_cls
    with observer_cls=RLABRStateObserver. See that function for full
    parameter documentation.

    Returns:
        Configured ABREnv instance with RLABRStateObserver.
    """
    return create_rl_env_with_observer_cls(
        observer_cls=RLABRStateObserver,
        *args,
        **kwargs,
    )

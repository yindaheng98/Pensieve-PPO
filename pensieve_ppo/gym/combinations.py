"""Convenience functions for creating ABR gymnasium environments."""

from typing import Type

from ..core import create_simulator
from .env import ABREnv, AbstractABRStateObserver
from .imitate import ImitationObserver


def create_env_with_observer_class(
    observer_class: Type[AbstractABRStateObserver],
    *args,
    initial_level: int = 0,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv by automatically constructing an observer from kwargs.

    This function automatically extracts the required constructor arguments
    for the given observer class from kwargs, creates the observer, and
    then creates the ABREnv with the remaining kwargs passed to create_simulator.

    Args:
        observer_class: The observer class to instantiate. Must define
                       REQUIRED_ARGS class variable listing required constructor
                       argument names.
        *args: Positional arguments passed to create_simulator.
        initial_level: Initial quality level index on reset (default: 0).
        **kwargs: Keyword arguments. Arguments matching the observer's
                 REQUIRED_ARGS will be extracted for observer construction,
                 and the rest will be passed to create_simulator.

    Returns:
        Configured ABREnv instance.

    Example:
        >>> from pensieve_ppo.agent.rl import RLABRStateObserver
        >>> env = create_env_with_observer_class(
        ...     RLABRStateObserver,
        ...     trace_folder=trace_folder,
        ...     video_size_file_prefix=video_size_file_prefix,
        ...     levels_quality=[300, 750, 1200, 1850, 2850, 4300],
        ... )
    """
    # Get required args for the observer
    required_args = observer_class.get_required_args()

    # Extract observer args from kwargs
    observer_kwargs = {arg: kwargs.pop(arg) for arg in required_args if arg in kwargs}

    # Create the observer
    observer = observer_class(**observer_kwargs)

    # Create and return the environment
    return create_env(observer, *args, initial_level=initial_level, **kwargs)


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


def create_imitation_env(
    student_observer: AbstractABRStateObserver,
    teacher_observer: AbstractABRStateObserver,
    *args,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv for imitation learning with student and teacher observers.

    This is a convenience function that combines two observers using
    ImitationObserver, enabling imitation learning where a student agent
    learns from a teacher agent's decisions.

    Example:
        >>> rl_observer = RLABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> bba_observer = BBAStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> env = create_imitation_env(
        ...     student_observer=rl_observer,
        ...     teacher_observer=bba_observer,
        ...     trace_folder=trace_folder,
        ...     video_size_file_prefix=video_size_file_prefix,
        ... )
        >>> state, info = env.reset()
        >>> # state.student_state is RLState (for training RL agent)
        >>> # state.teacher_state is BBAState (for BBA agent's decision)

    Args:
        student_observer: Observer for the student agent. Its observation_space
                         will be used as the environment's observation space.
        teacher_observer: Observer for the teacher agent. Used to generate
                         states for teacher's decision making.
        *args: Positional arguments passed to create_simulator.
        **kwargs: Keyword arguments passed to create_simulator.

    Returns:
        Configured ABREnv instance with ImitationObserver.
    """
    imitation_observer = ImitationObserver(
        student_observer=student_observer,
        teacher_observer=teacher_observer,
    )

    return create_env(
        imitation_observer,
        *args,
        **kwargs,
    )

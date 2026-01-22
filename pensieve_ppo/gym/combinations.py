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
        observer_class: The observer class to instantiate. Must implement
                       get_constructor_args() returning all constructor argument names.
        *args: Positional arguments passed to create_simulator.
        initial_level: Initial quality level index on reset (default: 0).
        **kwargs: Keyword arguments. Arguments matching the observer's
                 init args will be extracted for observer construction,
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
    # Get constructor args for the observer
    constructor_args = observer_class.get_constructor_args()

    # Extract observer args from kwargs
    observer_kwargs = {arg: kwargs.pop(arg) for arg in constructor_args if arg in kwargs}

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
    initial_level: int = 0,
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
        initial_level: Initial quality level index on reset (default: 0).
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
        initial_level=initial_level,
        **kwargs,
    )


def create_imitation_env_with_observer_class(
    student_observer_class: Type[AbstractABRStateObserver],
    teacher_observer_class: Type[AbstractABRStateObserver],
    *args,
    initial_level: int = 0,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv for imitation learning by auto-constructing observers from kwargs.

    This function automatically extracts required constructor arguments for both
    observer classes from kwargs, creates both observers, and creates an ABREnv
    with an ImitationObserver. Shared arguments are passed to both observers.

    Args:
        student_observer_class: The student observer class to instantiate.
        teacher_observer_class: The teacher observer class to instantiate.
        *args: Positional arguments passed to create_simulator.
        initial_level: Initial quality level index on reset (default: 0).
        **kwargs: Keyword arguments. Arguments matching either observer's
                 constructor args will be extracted for observer construction
                 (shared args go to both), and the rest will be passed to
                 create_simulator.

    Returns:
        Configured ABREnv instance with ImitationObserver.

    Example:
        >>> from pensieve_ppo.agent.rl import RLABRStateObserver
        >>> from pensieve_ppo.agent.bba import BBAStateObserver
        >>> env = create_imitation_env_with_observer_class(
        ...     RLABRStateObserver,
        ...     BBAStateObserver,
        ...     trace_folder=trace_folder,
        ...     video_size_file_prefix=video_size_file_prefix,
        ...     levels_quality=[300, 750, 1200, 1850, 2850, 4300],
        ... )
    """
    # Get constructor args for both observers
    student_args = set(student_observer_class.get_constructor_args())
    teacher_args = set(teacher_observer_class.get_constructor_args())
    all_observer_args = student_args | teacher_args

    # Extract observer args from kwargs (shared args go to both)
    student_kwargs = {arg: kwargs[arg] for arg in student_args if arg in kwargs}
    teacher_kwargs = {arg: kwargs[arg] for arg in teacher_args if arg in kwargs}

    # Remove all observer args from kwargs
    for arg in all_observer_args:
        kwargs.pop(arg, None)

    # Create observers
    student_observer = student_observer_class(**student_kwargs)
    teacher_observer = teacher_observer_class(**teacher_kwargs)

    # Create and return the environment
    return create_imitation_env(
        student_observer,
        teacher_observer,
        *args,
        initial_level=initial_level,
        **kwargs,
    )

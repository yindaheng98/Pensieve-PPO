"""Convenience functions for creating ABR gymnasium environments."""

from ..core import create_simulator
from .env import ABREnv, AbstractABRStateObserver
from .imitate import ImitationObserver


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

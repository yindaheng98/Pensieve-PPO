"""Convenience functions for creating ABR gymnasium environments."""

from typing import Any, Optional, Type

from ..core.simulator import create_simulator
from ..core.video import VideoPlayer
from .env import ABREnv, AbstractABRStateObserver
from .imitate import ImitationObserver


def create_env(
    video_player: VideoPlayer,
    observer: AbstractABRStateObserver,
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
) -> ABREnv:
    """Create an ABREnv with a configured Simulator.

    Args:
        video_player: Pre-configured video player instance.
        observer: ABRStateObserver instance for state observation and reward.
        trace_folder: Path to folder containing network trace files.
        train: Whether to use training trace behavior.
        random_seed: Random seed for training trace selection/noise.

    Returns:
        Configured ABREnv instance.
    """
    simulator = create_simulator(
        trace_folder=trace_folder,
        video_player=video_player,
        train=train,
        random_seed=random_seed,
    )

    return ABREnv(
        simulator=simulator,
        observer=observer,
    )


def create_env_with_class(
    video_player_class: Type[VideoPlayer],
    observer_class: Type[AbstractABRStateObserver],
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
    video_player_kwargs: dict[str, Any] = {},
    observer_kwargs: dict[str, Any] = {},
) -> ABREnv:
    """Create an ABREnv by constructing its player and observer classes.

    Args:
        video_player_class: Video player class to instantiate.
        observer_class: The observer class to instantiate.
        trace_folder: Path to folder containing network trace files.
        train: Whether to use training trace behavior.
        random_seed: Random seed for training trace selection/noise.
        video_player_kwargs: Keyword arguments passed to the video player constructor.
        observer_kwargs: Keyword arguments passed to the observer constructor.

    Returns:
        Configured ABREnv instance.

    Example:
        >>> from pensieve_ppo.agent.rl import RLABRStateObserver
        >>> env = create_env_with_class(
        ...     QualityLadderVideoPlayer,
        ...     RLABRStateObserver,
        ...     trace_folder=trace_folder,
        ...     observer_kwargs={"levels_quality": [300, 750, 1200, 1850, 2850, 4300]},
        ...     video_player_kwargs={"video_size_file_prefix": video_size_file_prefix},
        ... )
    """
    observer = observer_class(**observer_kwargs)
    video_player = video_player_class(**video_player_kwargs)

    return create_env(
        video_player,
        observer,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
    )


def create_imitation_env(
    video_player: VideoPlayer,
    student_observer: AbstractABRStateObserver,
    teacher_observer: AbstractABRStateObserver,
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
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
        video_player: Pre-configured video player instance.
        student_observer: Observer for the student agent. Its observation_space
                         will be used as the environment's observation space.
        teacher_observer: Observer for the teacher agent. Used to generate
                         states for teacher's decision making.
        trace_folder: Path to folder containing network trace files.
        train: Whether to use training trace behavior.
        random_seed: Random seed for training trace selection/noise.

    Returns:
        Configured ABREnv instance with ImitationObserver.
    """
    imitation_observer = ImitationObserver(
        student_observer=student_observer,
        teacher_observer=teacher_observer,
    )

    return create_env(
        video_player,
        imitation_observer,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
    )


def create_imitation_env_with_class(
    video_player_class: Type[VideoPlayer],
    student_observer_class: Type[AbstractABRStateObserver],
    teacher_observer_class: Type[AbstractABRStateObserver],
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
    video_player_kwargs: dict[str, Any] = {},
    student_observer_kwargs: dict[str, Any] = {},
    teacher_observer_kwargs: dict[str, Any] = {},
) -> ABREnv:
    """Create an imitation ABREnv by constructing player and observer classes.

    Args:
        video_player_class: Video player class to instantiate.
        student_observer_class: The student observer class to instantiate.
        teacher_observer_class: The teacher observer class to instantiate.
        trace_folder: Path to folder containing network trace files.
        train: Whether to use training trace behavior.
        random_seed: Random seed for training trace selection/noise.
        video_player_kwargs: Keyword arguments passed to the video player constructor.
        student_observer_kwargs: Keyword arguments passed to the student observer.
        teacher_observer_kwargs: Keyword arguments passed to the teacher observer.

    Returns:
        Configured ABREnv instance with ImitationObserver.

    Example:
        >>> from pensieve_ppo.agent.rl import RLABRStateObserver
        >>> from pensieve_ppo.agent.bba import BBAStateObserver
        >>> env = create_imitation_env_with_class(
        ...     QualityLadderVideoPlayer,
        ...     RLABRStateObserver,
        ...     BBAStateObserver,
        ...     trace_folder=trace_folder,
        ...     student_observer_kwargs={"levels_quality": [300, 750, 1200, 1850, 2850, 4300]},
        ...     teacher_observer_kwargs={"levels_quality": [300, 750, 1200, 1850, 2850, 4300]},
        ...     video_player_kwargs={"video_size_file_prefix": video_size_file_prefix},
        ... )
    """
    student_observer = student_observer_class(**student_observer_kwargs)
    teacher_observer = teacher_observer_class(**teacher_observer_kwargs)
    video_player = video_player_class(**video_player_kwargs)

    return create_imitation_env(
        video_player,
        student_observer,
        teacher_observer,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
    )

"""Agent and environment factory for registered ABR agent configurations."""

from dataclasses import dataclass
from typing import Optional, Type, Dict


from ..core.video import VideoPlayer
from ..gym import ABREnv, AbstractABRStateObserver
from ..gym.combinations import (
    create_env_with_class,
    create_imitation_env_with_class,
)
from .abc import AbstractAgent
from .trainable import AbstractTrainableAgent


@dataclass
class RegistryEntry:
    """Registry entry containing agent, observer, and player classes."""

    agent_cls: Type[AbstractAgent]
    trainable_agent_cls: Optional[Type[AbstractTrainableAgent]]
    observer_cls: Type[AbstractABRStateObserver]
    player_cls: Type[VideoPlayer]


# Unified registry of agents with their associated classes
REGISTRY: Dict[str, RegistryEntry] = {}


def register(
    name: str,
    agent_cls: Type[AbstractAgent],
    observer_cls: Type[AbstractABRStateObserver],
    player_cls: Type[VideoPlayer],
    trainable_agent_cls: Optional[Type[AbstractTrainableAgent]] = None,
) -> None:
    """Register an agent with its associated trainable agent and observer classes.

    This is the unified registration function that populates the REGISTRY.

    Args:
        name: Name to register the agent under (case-sensitive).
        agent_cls: The agent class to register.
        observer_cls: The observer class associated with this agent.
        trainable_agent_cls: Optional trainable agent class. If not provided,
            will be set to agent_cls if it's a subclass of AbstractTrainableAgent.
        player_cls: Video player class used by environments for this agent.

    Raises:
        ValueError: If the agent class is not a subclass of AbstractAgent,
            or if the observer class is not a subclass of AbstractABRStateObserver.
    """
    if not issubclass(agent_cls, AbstractAgent):
        raise ValueError(
            f"Agent class must be a subclass of AbstractAgent, got {agent_cls}"
        )
    if not issubclass(observer_cls, AbstractABRStateObserver):
        raise ValueError(
            f"Observer class must be a subclass of AbstractABRStateObserver, got {observer_cls}"
        )
    if not issubclass(player_cls, VideoPlayer):
        raise ValueError(
            f"Player class must be a subclass of VideoPlayer, got {player_cls}"
        )

    # Auto-detect trainable agent if not provided
    if trainable_agent_cls is None and issubclass(agent_cls, AbstractTrainableAgent):
        trainable_agent_cls = agent_cls

    # Validate trainable agent class if provided
    if trainable_agent_cls is not None:
        if not issubclass(trainable_agent_cls, AbstractTrainableAgent):
            raise ValueError(
                f"Trainable agent class must be a subclass of AbstractTrainableAgent, "
                f"got {trainable_agent_cls}"
            )

    # Register in unified REGISTRY
    entry = RegistryEntry(
        agent_cls=agent_cls,
        trainable_agent_cls=trainable_agent_cls,
        observer_cls=observer_cls,
        player_cls=player_cls,
    )
    REGISTRY[name] = entry


def get_available_agents() -> list[str]:
    """Get a list of available agent names.

    Returns:
        List of registered agent names.
    """
    return list(REGISTRY.keys())


def get_available_trainable_agents() -> list[str]:
    """Get a list of available trainable agent names.

    Returns:
        List of registered trainable agent names.
    """
    return [
        name for name, entry in REGISTRY.items()
        if entry.trainable_agent_cls is not None
    ]


def create_agent(
    name: str,
    *args,
    model_path: Optional[str] = None,
    **kwargs,
) -> AbstractAgent:
    """Create an agent by name.

    This factory function creates an agent instance based on the given name.
    Since different agents may have different constructor signatures, all
    positional and keyword arguments are passed directly to the agent class.

    Args:
        name: Name of the agent to create (case-insensitive).
            Available agents: "ppo".
        *args: Positional arguments passed to the agent constructor.
        model_path: Path to a saved model file. If provided, loads the model
            parameters after creating the agent. Only supported for trainable agents.
        **kwargs: Keyword arguments passed to the agent constructor.

    Returns:
        An instance of the requested agent.

    Raises:
        ValueError: If the agent name is not recognized, or if model_path is
            provided but the agent is not a trainable agent.

    Example:
        >>> agent = create_agent(
        ...     name="ppo",
        ...     state_dim=(6, 8),
        ...     action_dim=6,
        ...     device=torch.device("cuda"),
        ...     model_path="models/ppo_model.pt",
        ...     learning_rate=1e-4,
        ...     gamma=0.99,
        ... )
    """
    if name not in REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown agent: '{name}'. Available agents: {available}"
        )

    entry = REGISTRY[name]
    agent_class = entry.agent_cls

    # Check if model_path is provided but agent is not trainable
    if model_path is not None and entry.trainable_agent_cls is None:
        raise ValueError(
            f"Agent '{name}' is not a trainable agent and does not support loading models. "
            f"Available trainable agents: {', '.join(get_available_trainable_agents())}"
        )
    agent = agent_class(*args, **kwargs)

    if model_path is not None:
        agent.load(model_path)

    return agent


def create_env(
    name: str,
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
    observer_kwargs: dict = {},
    player_kwargs: dict = {},
) -> ABREnv:
    """Create an environment by name.

    This factory function creates an environment instance based on the given name.
    It retrieves registered agent, observer, and video player classes from REGISTRY,
    then constructs a compatible ABR environment.

    Args:
        name: Name of the agent/environment to create (case-sensitive).
            Must be registered in REGISTRY.
        trace_folder: Path to folder containing network trace files.
        train: Whether to use training trace behavior.
        random_seed: Random seed for training trace selection/noise.
        observer_kwargs: Keyword arguments passed to the observer constructor.
        player_kwargs: Keyword arguments passed to the video player constructor.

    Returns:
        An instance of the requested environment (ABREnv).

    Raises:
        ValueError: If the agent name is not recognized in REGISTRY.

    Example:
        >>> env = create_env(
        ...     name="ppo",
        ...     observer_kwargs={"levels_quality": VIDEO_BIT_RATE, "rebuf_penalty": 4.3},
        ...     trace_folder=trace_folder,
        ...     player_kwargs={"video_size_file_prefix": video_size_file_prefix},
        ... )
    """
    if name not in REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown agent/environment: '{name}'. Available agents: {available}"
        )

    entry = REGISTRY[name]
    return create_env_with_class(
        entry.player_cls,
        entry.observer_cls,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
        video_player_kwargs=player_kwargs,
        observer_kwargs=observer_kwargs,
    )


def create_imitation_env(
    student_name: str,
    teacher_name: str,
    trace_folder: str,
    train: bool = True,
    random_seed: Optional[int] = None,
    student_observer_kwargs: dict = {},
    teacher_observer_kwargs: dict = {},
    player_kwargs: dict = {},
) -> ABREnv:
    """Create an imitation learning environment by agent names.

    This factory function creates an environment for imitation learning where
    a student agent learns from a teacher agent. It retrieves observer and
    video player classes from REGISTRY, then constructs a compatible ABR
    environment.

    Args:
        student_name: Name of the student agent (case-sensitive).
            Must be registered in REGISTRY.
        teacher_name: Name of the teacher agent (case-sensitive).
            Must be registered in REGISTRY.
        trace_folder: Path to folder containing network trace files.
        train: Whether to use training trace behavior.
        random_seed: Random seed for training trace selection/noise.
        student_observer_kwargs: Keyword arguments passed to the student observer.
        teacher_observer_kwargs: Keyword arguments passed to the teacher observer.
        player_kwargs: Keyword arguments passed to the video player constructor.

    Returns:
        An instance of ABREnv with ImitationObserver.

    Raises:
        ValueError: If either agent name is not recognized in REGISTRY.

    Example:
        >>> env = create_imitation_env(
        ...     student_name="ppo",
        ...     teacher_name="bba",
        ...     student_observer_kwargs={"levels_quality": VIDEO_BIT_RATE},
        ...     teacher_observer_kwargs={"levels_quality": VIDEO_BIT_RATE},
        ...     trace_folder=trace_folder,
        ...     player_kwargs={"video_size_file_prefix": video_size_file_prefix},
        ... )
    """
    if student_name not in REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown student agent: '{student_name}'. Available agents: {available}"
        )
    if teacher_name not in REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown teacher agent: '{teacher_name}'. Available agents: {available}"
        )

    student_entry = REGISTRY[student_name]
    teacher_entry = REGISTRY[teacher_name]

    return create_imitation_env_with_class(
        student_entry.player_cls,
        student_entry.observer_cls,
        teacher_entry.observer_cls,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
        video_player_kwargs=player_kwargs,
        student_observer_kwargs=student_observer_kwargs,
        teacher_observer_kwargs=teacher_observer_kwargs,
    )

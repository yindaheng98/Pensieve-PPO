"""Agent and environment factory for creating different types of RL agents and environments.

This module provides factory functions to create agents and environments by name,
allowing for easy switching between different implementations.
"""

from dataclasses import dataclass
from typing import Any, Optional, Type, Dict

import gymnasium as gym

from ..gym import ABREnv, AbstractABRStateObserver
from ..gym.combinations import create_env as create_gym_env
from .abc import AbstractAgent
from .trainable import AbstractTrainableAgent


@dataclass
class RegistryEntry:
    """Registry entry containing agent, trainable agent, and observer classes."""

    agent_cls: Type[AbstractAgent]
    trainable_agent_cls: Optional[Type[AbstractTrainableAgent]]
    observer_cls: Type[AbstractABRStateObserver]


# Unified registry of agents with their associated classes
REGISTRY: Dict[str, RegistryEntry] = {}


def register(
    name: str,
    agent_cls: Type[AbstractAgent],
    observer_cls: Type[AbstractABRStateObserver],
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
    env_options: Dict[str, Any] = {},
    observer_options: Dict[str, Any] = {},
) -> ABREnv:
    """Create an environment by name.

    This factory function creates an environment instance based on the given name.
    It retrieves the observer class from REGISTRY, instantiates it with observer_options,
    and then creates the environment using the gym combinations.create_env function.

    Args:
        name: Name of the agent/environment to create (case-sensitive).
            Must be registered in REGISTRY.
        env_options: Dictionary of keyword arguments passed to create_gym_env
            (e.g., initial_level, trace_folder, video_size_file_prefix).
            Defaults to empty dict.
        observer_options: Dictionary of keyword arguments passed to observer constructor
            (e.g., levels_quality, rebuf_penalty). Defaults to empty dict.

    Returns:
        An instance of the requested environment (ABREnv).

    Raises:
        ValueError: If the agent name is not recognized in REGISTRY.

    Example:
        >>> env = create_env(
        ...     name="ppo",
        ...     env_options={
        ...         "initial_level": 0,
        ...         "trace_folder": trace_folder,
        ...         "video_size_file_prefix": video_size_file_prefix,
        ...     },
        ...     observer_options={
        ...         "levels_quality": VIDEO_BIT_RATE,
        ...         "rebuf_penalty": 4.3,
        ...     },
        ... )
    """
    if name not in REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown agent/environment: '{name}'. Available agents: {available}"
        )

    entry = REGISTRY[name]
    observer_cls = entry.observer_cls

    # Create observer instance
    observer = observer_cls(**observer_options)

    # Create environment using gym combinations.create_env
    env = create_gym_env(
        observer=observer,
        **env_options,
    )

    return env

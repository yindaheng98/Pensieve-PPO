"""Agent factory for creating different types of RL agents.

This module provides a factory function to create agents by name,
allowing for easy switching between different agent implementations.
"""

from typing import Optional, Type, Dict

import torch

from .abc import AbstractAgent
from .trainable import AbstractTrainableAgent


# Registry of available agents
AGENT_REGISTRY: Dict[str, Type[AbstractAgent]] = {}

# Registry of trainable agents (subset of AGENT_REGISTRY)
TRAINABLEAGENT_REGISTRY: Dict[str, Type[AbstractTrainableAgent]] = {}


def register_agent(name: str, agent_class: Type[AbstractAgent]) -> None:
    """Register a new agent type.

    Args:
        name: Name to register the agent under (case-insensitive).
        agent_class: The agent class to register.

    Raises:
        ValueError: If the agent class is not a subclass of AbstractAgent.
    """
    if not issubclass(agent_class, AbstractAgent):
        raise ValueError(f"Agent class must be a subclass of AbstractAgent, got {agent_class}")
    AGENT_REGISTRY[name.lower()] = agent_class

    # Also register in TRAINABLEAGENT_REGISTRY if it's a trainable agent
    if issubclass(agent_class, AbstractTrainableAgent):
        TRAINABLEAGENT_REGISTRY[name.lower()] = agent_class


def get_available_agents() -> list[str]:
    """Get a list of available agent names.

    Returns:
        List of registered agent names.
    """
    return list(AGENT_REGISTRY.keys())


def get_available_trainable_agents() -> list[str]:
    """Get a list of available trainable agent names.

    Returns:
        List of registered trainable agent names.
    """
    return list(TRAINABLEAGENT_REGISTRY.keys())


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
    if name not in AGENT_REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown agent: '{name}'. Available agents: {available}"
        )

    # Check if model_path is provided but agent is not trainable
    if model_path is not None and name not in TRAINABLEAGENT_REGISTRY:
        raise ValueError(
            f"Agent '{name}' is not a trainable agent and does not support loading models. "
            f"Available trainable agents: {', '.join(get_available_trainable_agents())}"
        )

    agent_class = AGENT_REGISTRY[name]
    agent = agent_class(*args, **kwargs)

    if model_path is not None:
        agent.load(model_path)

    return agent

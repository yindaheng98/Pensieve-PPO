"""Agent factory for creating different types of RL agents.

This module provides a factory function to create agents by name,
allowing for easy switching between different agent implementations.
"""

from typing import Optional, Type, Dict

import torch

from .abc import AbstractAgent


# Registry of available agents
AGENT_REGISTRY: Dict[str, Type[AbstractAgent]] = {}


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


def get_available_agents() -> list[str]:
    """Get a list of available agent names.

    Returns:
        List of registered agent names.
    """
    return list(AGENT_REGISTRY.keys())


def create_agent(
    name: str,
    state_dim: tuple[int, int],
    action_dim: int,
    device: Optional[torch.device] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> AbstractAgent:
    """Create an agent by name.

    This factory function creates an agent instance based on the given name.
    The AbstractAgent parameters (state_dim, action_dim, device) are explicitly
    defined, while agent-specific parameters are passed via **kwargs.

    Args:
        name: Name of the agent to create (case-insensitive).
            Available agents: "ppo".
        state_dim: State dimension as [num_features, sequence_length].
        action_dim: Number of discrete actions.
        device: PyTorch device for computations. If None, uses CPU.
        model_path: Path to a saved model file. If provided, loads the model
            parameters after creating the agent.
        **kwargs: Additional agent-specific parameters.
            For PPO agent:
                - learning_rate (float): Learning rate for the optimizer. Default: 1e-4.
                - gamma (float): Discount factor for future rewards. Default: 0.99.
                - eps (float): PPO clipping parameter. Default: 0.2.
                - ppo_training_epo (int): Number of PPO update epochs. Default: 5.
                - h_target (float): Target entropy for adaptive entropy weight. Default: 0.1.

    Returns:
        An instance of the requested agent.

    Raises:
        ValueError: If the agent name is not recognized.

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
    name_lower = name.lower()

    if name_lower not in AGENT_REGISTRY:
        available = ", ".join(get_available_agents())
        raise ValueError(
            f"Unknown agent: '{name}'. Available agents: {available}"
        )

    agent_class = AGENT_REGISTRY[name_lower]
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **kwargs,
    )

    if model_path is not None:
        agent.load(model_path)

    return agent

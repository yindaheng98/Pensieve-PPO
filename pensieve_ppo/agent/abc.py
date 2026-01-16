"""Abstract base classes for ABR agents.

This module provides the base abstract class for all agents:
- AbstractAgent: Minimal interface for any ABR agent (reset and step)

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


from ..gym.env import Observation


class AbstractAgent(ABC):
    """Abstract base class for any ABR agent.

    This is the minimal interface that all ABR agents must implement.
    It requires reset and step methods, similar to a gymnasium environment,
    allowing the agent to maintain internal state across episodes.
    This allows for simple rule-based agents, heuristic agents, or RL agents to all
    share the same interface for evaluation.

    The typical usage pattern is:
        agent.reset()
        while not done:
            action = agent.step(observation)
            observation, reward, done, truncated, info = env.step(action)
    """

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the agent's internal state.

        This method should be called at the beginning of each episode to reset
        any internal state the agent maintains (e.g., history buffers, counters).
        Similar to gymnasium.Env.reset(), but only resets internal agent state,
        not the environment.

        Args:
            seed: Optional random seed for reproducibility.
            options: Additional options for reset behavior. Common options include:
                - initial_level (int): Initial bitrate level index.
        """
        pass

    @abstractmethod
    def step(self, observation: Observation) -> int:
        """Select an action given an observation.

        This method is called at each timestep to get the agent's action.
        The agent may update its internal state based on the observation.

        Args:
            observation: Raw observation from the environment.

        Returns:
            Action index (bitrate level to select).
        """
        pass

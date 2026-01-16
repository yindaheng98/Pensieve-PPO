"""Abstract base classes for ABR agents.

This module provides the abstract base classes for all agents:
- AbstractAgent: Minimal interface for any ABR agent (predict only)
- AbstractRLAgent: Extended interface for reinforcement learning agents

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod


from ..gym.env import Observation


class AbstractAgent(ABC):
    """Abstract base class for any ABR agent.

    This is the minimal interface that all ABR agents must implement.
    It only requires a predict method that takes an observation and returns an action.
    This allows for simple rule-based agents, heuristic agents, or RL agents to all
    share the same interface for evaluation.
    """

    @abstractmethod
    def predict(self, observation: Observation) -> int:
        """Predict an action given an observation.

        Args:
            observation: Raw observation from the environment.

        Returns:
            Action index (bitrate level to select).
        """
        pass

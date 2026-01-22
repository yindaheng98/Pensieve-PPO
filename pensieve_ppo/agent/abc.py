"""Abstract base classes for agents.

This module provides the abstract base class hierarchy for all agents:
- AbstractAgent: Base class with select_action and reset

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from abc import ABC, abstractmethod
from typing import Tuple, List

from ..gym import State


class AbstractAgent(ABC):
    """Abstract base class for agents with action selection capability.

    This class defines the minimal interface for agents that can select
    actions from states.

    Design Note on Agent "Statefulness":
        Logically, an Agent should be stateless. All historical information needed
        by the Agent should be collected by the Observer and passed through the
        State object. However, in some special cases, an Agent may need to maintain
        its own "internal state".

        For example, in NetLLM (see `pensieve_ppo/agent/netllm`), the large language
        model needs to cache embeddings of historical states to avoid redundant
        computation. Since the embedding model is a trainable part of the policy,
        it cannot be moved into the Observer. In such cases, the Agent must maintain
        its own "internal state" (not the actual environment state), i.e., some
        special internal data structures (like an embedding cache) that accelerate
        computation.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py

        Furthermore, this "internal state" management is essentially maintaining an
        embedding cache rather than managing the actual environment state. Theoretically,
        this should support out-of-order `select_action` calls by querying pre-computed
        embeddings based on the input state. However, since the current codebase does
        not have out-of-order `select_action` calls, we can simplify the implementation
        by assuming sequential calls only.

        Using "state management" methods to manage the cache has the following tradeoffs:
        - Pros: Eliminates embedding cache lookup steps; allows precise control of cache
          size since we know exactly which embeddings are needed; better performance.
        - Cons: If the same state appears at distant timesteps, the embedding must be
          recomputed rather than retrieved from cache.

        Note: The agents in `pensieve_ppo/agent/rl/`, `pensieve_ppo/agent/mpc/`, and
        `pensieve_ppo/agent/bba/` are all stateless - they do not maintain any internal
        "internal state" between `select_action` calls.
    """

    @abstractmethod
    def select_action(self, state: State) -> Tuple[int, List[float]]:
        """Select an action for a given state.

        Args:
            state: Input state.

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        pass

    def reset(self) -> None:
        """Reset the agent's "internal state" for a new episode.

        This method should be called at the beginning of each episode to clear
        any "internal state" (e.g., embedding caches) that the agent may maintain.

        For stateless agents (e.g., those in `pensieve_ppo/agent/rl/`,
        `pensieve_ppo/agent/mpc/`, `pensieve_ppo/agent/bba/`), this method
        is a no-op.

        For "stateful" agents (e.g., NetLLM in `pensieve_ppo/agent/netllm/`),
        this method should clear embedding caches and other internal buffers.

        See the class docstring for more details on agent "statefulness".
        """
        pass

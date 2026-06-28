"""Gymnasium-based ABR (Adaptive Bitrate) environment.

This module implements the ABR environment using the gymnasium API,
wrapping the Pensieve simulator components.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.simulator import Simulator, StepResult
from ..core.video import (
    VideoChunkRequestType,
    VideoChunkRequestTyped,
)


@dataclass
class State:
    """Base class for state representation.

    This is the base dataclass for all state types used in ABR environments.
    Concrete state classes (e.g., RLState, MPCState, BBAState) should inherit
    from this class to ensure compatibility across different agent types.

    This enables imitation learning where trajectories collected by one type
    of agent (e.g., MPC, BBA) can be used to train another type (e.g., RL).
    """
    pass


class AbstractABRStateObserver(VideoChunkRequestTyped[VideoChunkRequestType], ABC):
    """Abstract base class for ABR state observers.

    This class defines the interface that ABREnv uses to interact with
    state observers. Subclasses must implement state observation and
    reward calculation logic.

    Subclasses must also implement the `get_constructor_args` class method,
    which returns all constructor argument names accepted by the observer.
    This enables automatic observer construction from keyword arguments.
    """

    @classmethod
    @abstractmethod
    def get_constructor_args(cls) -> List[str]:
        """Get the list of all constructor argument names.

        Returns:
            List of all argument names accepted by __init__ (excluding self).
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Box:
        """Gymnasium observation space for the state.

        Returns:
            Gymnasium Box space defining the observation shape and bounds.
        """
        pass

    @abstractmethod
    def reset(
        self,
        env: "ABREnv[VideoChunkRequestType]",
        initial_chunk_request: VideoChunkRequestType,
    ) -> Tuple[State, Dict[str, Any]]:
        """Reset observer state and return initial observation.

        Args:
            env: The ABREnv instance to observe.
            initial_chunk_request: Initial video chunk request.

        Returns:
            Tuple of (state, info_dict).
        """
        pass

    @abstractmethod
    def observe(
        self,
        env: "ABREnv[VideoChunkRequestType]",
        chunk_request: VideoChunkRequestType,
        result: StepResult,
    ) -> Tuple[State, float, Dict[str, Any]]:
        """Process simulator result: compute reward and update state.

        Args:
            env: The ABREnv instance to observe.
            chunk_request: Current video chunk request.
            result: Result from simulator.step().

        Returns:
            Tuple of (state, reward, info_dict).
        """
        pass


class ABREnv(gym.Env, Generic[VideoChunkRequestType]):
    """Gymnasium environment for Adaptive Bitrate Streaming.

    This environment simulates video streaming over a network with
    variable bandwidth. The agent must select bitrate levels to maximize
    video quality while minimizing rebuffering and bitrate oscillations.

    Observation Space:
        Box(S_INFO, state_history_len) containing:
        - [0, :] Last quality normalized by max quality
        - [1, :] Buffer size normalized by buffer_norm_factor
        - [2, :] Throughput (chunk_size / delay) in Mbps
        - [3, :] Delay normalized by buffer_norm_factor
        - [4, :bitrate_levels] Next chunk sizes at each bitrate level (in MB)
        - [5, :] Remaining chunks normalized by total_chunk_cap

    Action Space:
        Discrete(bitrate_levels) - select bitrate level

    Reward:
        quality - rebuf_penalty * rebuffer - smooth_penalty * |quality_change|
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: Simulator[VideoChunkRequestType],
        observer: AbstractABRStateObserver[VideoChunkRequestType],
        initial_chunk_request: VideoChunkRequestType,
    ):
        """Initialize the ABR environment.

        Args:
            simulator: Pre-configured Simulator instance (use create_simulator
                      from pensieve_ppo.core to create one)
            observer: ABRStateObserver instance for state observation and reward
                     calculation. Its levels_quality length must match
                     simulator.video_player.bitrate_levels.
            initial_chunk_request: Initial video chunk request on reset.
        """
        super().__init__()

        # Store simulator
        self.simulator = simulator

        # Store observer
        self.observer = observer

        # Store initial chunk request
        self.initial_chunk_request = initial_chunk_request

        observer.validate_request_cls_match(simulator.video_player)

        # timestamp in ms for logging purposes
        self.time_stamp = 0.0

        # Define observation and action spaces
        self.observation_space = observer.observation_space
        self.action_space = spaces.Discrete(simulator.video_player.bitrate_levels)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[State, Dict[str, Any]]:
        """Reset the environment to initial state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L42-L47
        This method only initializes state to zeros and does NOT execute the first chunk download.
        The first chunk should be downloaded by calling step(action) after reset().
        This follows proper Gymnasium API separation of concerns.

        Args:
            seed: Random seed for reproducibility
            options: Additional options:
                - reset_time_stamp (bool): Whether to reset time_stamp to 0.
                  Default is True. Set to False for testing mode where time_stamp
                  should accumulate across traces.
                - initial_chunk_request: Override initial video chunk request.
                  If not specified, uses initial_chunk_request from __init__.

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Parse options
        options = options or {}
        reset_time_stamp = options.get('reset_time_stamp', True)
        initial_chunk_request = options.get('initial_chunk_request', self.initial_chunk_request)

        if reset_time_stamp:
            self.time_stamp = 0

        # Reset observer and get initial state
        state, observer_info = self.observer.reset(self, initial_chunk_request)

        info = {
            "time_stamp": self.time_stamp,
            **observer_info
        }

        return state, info

    def step(
        self, action: VideoChunkRequestType,
    ) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L72-L106

        Args:
            action: Video chunk request for this step.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # the action is from the last decision
        # this is to make the framework similar to the real
        result = self.simulator.step(action)

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L80-L81
        self.time_stamp += result.delay  # in ms
        self.time_stamp += result.sleep_time  # in ms

        # Observe state and compute reward
        state, reward, observer_info = self.observer.observe(self, action, result)

        # Episode termination
        terminated = result.end_of_video
        truncated = False

        info = {
            "time_stamp": self.time_stamp,
            "delay": result.delay,
            "sleep_time": result.sleep_time,
            "buffer_size": result.buffer_size,
            "rebuffer": result.rebuffer,
            "video_chunk_size": result.video_chunk_size,
            "video_chunk_quality": result.video_chunk_quality,
            "next_video_chunk_sizes": result.next_video_chunk_sizes,
            "video_chunk_remain": result.video_chunk_remain,
            "end_of_video": result.end_of_video,
            **observer_info
        }
        return state, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented for this env)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass

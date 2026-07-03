"""Gymnasium-based ABR (Adaptive Bitrate) environment.

This module implements the ABR environment using the gymnasium API,
wrapping the Pensieve simulator components.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..core.simulator import Simulator, StepResult
from ..core.video import VideoChunkRequest


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


class AbstractABRStateObserver(ABC):
    """Abstract base class for ABR state observers.

    This class defines the interface that ABREnv uses to interact with
    state observers. Subclasses must implement state observation and
    reward calculation logic.

    Observer construction is handled by factory functions using explicit
    observer kwargs.
    """

    @abstractmethod
    def reset(
        self,
        env: "ABREnv",
    ) -> None:
        """Reset observer state.

        Args:
            env: The ABREnv instance to observe.
        """
        pass

    @abstractmethod
    def observe(
        self,
        env: "ABREnv",
        chunk_request: VideoChunkRequest,
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


class ABREnv(gym.Env):
    """Gymnasium environment for Adaptive Bitrate Streaming.

    This environment simulates video streaming over a network with
    variable bandwidth. The agent must select bitrate levels to maximize
    video quality while minimizing rebuffering and bitrate oscillations.

    Reward:
        quality - rebuf_penalty * rebuffer - smooth_penalty * |quality_change|
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: Simulator,
        observer: AbstractABRStateObserver,
    ):
        """Initialize the ABR environment.

        Args:
            simulator: Pre-configured Simulator instance (use create_simulator
                      from pensieve_ppo.core to create one)
            observer: ABRStateObserver instance for state observation and reward
                     calculation.
        """
        super().__init__()

        # Store simulator
        self.simulator = simulator

        # Store observer
        self.observer = observer

        # timestamp in ms for logging purposes
        self.time_stamp = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[None, Dict[str, Any]]:
        """Reset the environment to initial state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L42-L47
        This method only initializes environment and observer state. The first
        chunk should be downloaded explicitly by calling step() with the request
        returned by agent.reset().

        Args:
            seed: Random seed for reproducibility
            options: Additional options:
                - reset_time_stamp (bool): Whether to reset time_stamp to 0.
                  Default is True. Set to False for testing mode where time_stamp
                  should accumulate across traces.

        Returns:
            Tuple of (None, info_dict). The first usable observation is
            produced by an explicit step() call.
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Parse options
        options = options or {}
        reset_time_stamp = options.get('reset_time_stamp', True)

        if reset_time_stamp:
            self.time_stamp = 0

        self.observer.reset(self)

        info = {
            "time_stamp": self.time_stamp,
        }

        return None, info

    def step(
        self, action: VideoChunkRequest,
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

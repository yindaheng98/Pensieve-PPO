"""Gymnasium-based ABR (Adaptive Bitrate) environment.

This module implements the ABR environment using the gymnasium API,
wrapping the Pensieve simulator components.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.simulator import Simulator


# State dimensions
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L8
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past

# Default bitrate levels
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L12
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps

# Normalization constants
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0

# Reward parameters
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L17
REBUF_PENALTY = 4.3  # 1 sec rebuffering penalty
SMOOTH_PENALTY = 1.0  # penalty for bitrate changes

# Default values
DEFAULT_QUALITY = 1  # default video quality without agent


class ABREnv(gym.Env):
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
        - [4, :num_bitrates] Next chunk sizes at each bitrate level (in MB)
        - [5, :] Remaining chunks normalized by total_chunk_cap

    Action Space:
        Discrete(num_bitrates) - select bitrate level

    Reward:
        quality - rebuf_penalty * rebuffer - smooth_penalty * |quality_change|
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: Simulator,
        video_bit_rate: Optional[np.ndarray] = None,
        rebuf_penalty: float = REBUF_PENALTY,
        smooth_penalty: float = SMOOTH_PENALTY,
        state_history_len: int = S_LEN,
        buffer_norm_factor: float = BUFFER_NORM_FACTOR,
        total_chunk_cap: float = CHUNK_TIL_VIDEO_END_CAP,
        initial_bitrate: int = DEFAULT_QUALITY,
    ):
        """Initialize the ABR environment.

        Args:
            simulator: Pre-configured Simulator instance (use create_simulator
                      from pensieve_ppo.core to create one)
            video_bit_rate: Array of bitrate values in Kbps (default: Pensieve values)
            rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3)
            smooth_penalty: Penalty coefficient for quality changes (default: 1.0)
            state_history_len: Number of past observations to keep in state (default: 8)
            buffer_norm_factor: Normalization factor for buffer size in seconds (default: 10.0)
            total_chunk_cap: Cap value for remaining chunks normalization (default: 48.0)
            initial_bitrate: Initial bitrate level index on reset (default: 1)
        """
        super().__init__()

        # Store simulator
        self.simulator = simulator

        # Store reward parameters
        self.rebuf_penalty = rebuf_penalty
        self.smooth_penalty = smooth_penalty
        self.video_bit_rate = video_bit_rate if video_bit_rate is not None else VIDEO_BIT_RATE.copy()
        self.num_bitrates = len(self.video_bit_rate)

        # Store normalization parameters
        self.state_history_len = state_history_len
        self.buffer_norm_factor = buffer_norm_factor
        self.total_chunk_cap = total_chunk_cap

        # Store initial bitrate
        self.initial_bitrate = initial_bitrate

        # Initialize state
        self.last_bit_rate = self.initial_bitrate
        self.buffer_size = 0.0
        self.state = np.zeros((S_INFO, self.state_history_len), dtype=np.float32)
        self.time_stamp = 0.0

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(S_INFO, self.state_history_len),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_bitrates)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L72-L106

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.time_stamp = 0
        self.last_bit_rate = self.initial_bitrate
        self.state = np.zeros((S_INFO, self.state_history_len))
        self.buffer_size = 0.
        bit_rate = self.last_bit_rate
        result = self.simulator.step(bit_rate)
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = (
                result.delay,
                result.sleep_time,
                result.buffer_size,
                result.rebuffer,
                result.video_chunk_size,
                result.next_video_chunk_sizes,
                result.end_of_video,
                result.video_chunk_remain,
            )
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.video_bit_rate[bit_rate] / \
            float(np.max(self.video_bit_rate))  # last quality
        state[1, -1] = self.buffer_size / self.buffer_norm_factor
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / self.buffer_norm_factor
        state[4, :self.num_bitrates] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.total_chunk_cap) / float(self.total_chunk_cap)
        self.state = state

        info = {
            "bitrate": self.video_bit_rate[bit_rate],
            "rebuffer": 0.0,
        }

        return self.state.copy(), info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L72-L106

        Args:
            action: Bitrate level to select (0 to num_bitrates-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        result = self.simulator.step(bit_rate)
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = (
                result.delay,
                result.sleep_time,
                result.buffer_size,
                result.rebuffer,
                result.video_chunk_size,
                result.next_video_chunk_sizes,
                result.end_of_video,
                result.video_chunk_remain,
            )

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = self.video_bit_rate[bit_rate] / M_IN_K \
            - self.rebuf_penalty * rebuf \
            - self.smooth_penalty * np.abs(self.video_bit_rate[bit_rate] -
                                           self.video_bit_rate[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.video_bit_rate[bit_rate] / \
            float(np.max(self.video_bit_rate))  # last quality
        state[1, -1] = self.buffer_size / self.buffer_norm_factor
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / self.buffer_norm_factor
        state[4, :self.num_bitrates] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.total_chunk_cap) / float(self.total_chunk_cap)

        self.state = state

        # Episode termination
        terminated = end_of_video
        truncated = False

        info = {
            "bitrate": self.video_bit_rate[bit_rate],
            "rebuffer": result.rebuffer,
            "delay": result.delay,
            "sleep_time": result.sleep_time,
            "video_chunk_size": result.video_chunk_size,
            "video_chunk_remain": result.video_chunk_remain,
        }

        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented for this env)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass

"""Gymnasium-based ABR (Adaptive Bitrate) environment.

This module implements the ABR environment using the gymnasium API,
wrapping the Pensieve simulator components.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.simulator import Simulator, StepResult


# State dimensions
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L8
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past

# Normalization constants
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
BUFFER_NORM_FACTOR = 10.0
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
        simulator: Simulator,
        levels_quality: List[float],
        rebuf_penalty: float = REBUF_PENALTY,
        smooth_penalty: float = SMOOTH_PENALTY,
        state_history_len: int = S_LEN,
        buffer_norm_factor: float = BUFFER_NORM_FACTOR,
        initial_level: int = DEFAULT_QUALITY,
    ):
        """Initialize the ABR environment.

        Args:
            simulator: Pre-configured Simulator instance (use create_simulator
                      from pensieve_ppo.core to create one)
            levels_quality: Quality metric list for each bitrate level, used for
                           state representation and reward calculation. Length must
                           match simulator.video_player.bitrate_levels.
                           For example, bitrate values in Kbps: [300, 750, 1200, ...]
                           or other quality indicators like VMAF/PSNR scores.
            rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3)
            smooth_penalty: Penalty coefficient for quality changes (default: 1.0)
            state_history_len: Number of past observations to keep in state (default: 8)
            buffer_norm_factor: Normalization factor for buffer size in seconds (default: 10.0)
            initial_level: Initial quality level index on reset (default: 1)
        """
        super().__init__()

        # Store simulator
        self.simulator = simulator

        # Store reward parameters
        self.rebuf_penalty = rebuf_penalty
        self.smooth_penalty = smooth_penalty
        self.levels_quality = levels_quality

        # Validate levels_quality length matches bitrate_levels from simulator
        if len(self.levels_quality) != self.bitrate_levels:
            raise ValueError(
                f"levels_quality length ({len(self.levels_quality)}) must match "
                f"simulator.video_player.bitrate_levels ({self.bitrate_levels})"
            )

        # Store normalization parameters
        self.state_history_len = state_history_len
        self.buffer_norm_factor = buffer_norm_factor

        # Store initial bitrate
        self.initial_bitrate = initial_level

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
        self.action_space = spaces.Discrete(self.bitrate_levels)

    @property
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels from the video player."""
        return self.simulator.video_player.bitrate_levels

    @property
    def total_chunk_cap(self) -> int:
        """Total number of video chunks, used for normalization."""
        return self.simulator.video_player.total_chunks

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L41-L66
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L80-L84

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
        self.buffer_size = result.buffer_size

        # Accumulate time_stamp for the first chunk (consistent with step behavior)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L76-L77
        self.time_stamp += result.delay  # in ms
        self.time_stamp += result.sleep_time  # in ms

        _, info = self._process_step_result(bit_rate, result)

        return self.state.copy(), info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L72-L106

        Args:
            action: Bitrate level to select (0 to bitrate_levels-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        result = self.simulator.step(bit_rate)
        self.buffer_size = result.buffer_size

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L80-L81
        self.time_stamp += result.delay  # in ms
        self.time_stamp += result.sleep_time  # in ms

        reward, info = self._process_step_result(bit_rate, result)

        self.last_bit_rate = bit_rate

        # Episode termination
        terminated = result.end_of_video
        truncated = False

        return self.state.copy(), float(reward), terminated, truncated, info

    def _process_step_result(
        self,
        bit_rate: int,
        result: StepResult,
    ) -> Tuple[float, Dict[str, Any]]:
        """Process simulator result: compute reward, update state, and build info.

        Args:
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            Tuple of (reward, info_dict).
        """
        # Unpack result (matches original variable names)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L75-L78
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L48-L51
        delay, sleep_time, buffer_size, rebuf, \
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

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L83-L87
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L80-L84
        # reward is video quality - rebuffer penalty - smooth penalty
        reward = self.levels_quality[bit_rate] / M_IN_K \
            - self.rebuf_penalty * rebuf \
            - self.smooth_penalty * np.abs(self.levels_quality[bit_rate] -
                                           self.levels_quality[self.last_bit_rate]) / M_IN_K

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L90-L104
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L52-L66
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.levels_quality[bit_rate] / \
            float(np.max(self.levels_quality))  # last quality
        state[1, -1] = buffer_size / self.buffer_norm_factor  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / self.buffer_norm_factor  # 10 sec
        state[4, :self.bitrate_levels] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.total_chunk_cap) / float(self.total_chunk_cap)

        self.state = state

        # Build info dict with all step details
        info = {
            "time_stamp": self.time_stamp,
            "quality": self.levels_quality[bit_rate],
            "rebuffer": rebuf,
            "delay": delay,
            "sleep_time": sleep_time,
            "buffer_size": buffer_size,
            "video_chunk_size": video_chunk_size,
            "video_chunk_remain": video_chunk_remain,
            "reward": reward,
        }

        return reward, info

    def render(self) -> None:
        """Render the environment (not implemented for this env)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass

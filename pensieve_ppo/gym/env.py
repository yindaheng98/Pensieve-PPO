"""Gymnasium-based ABR (Adaptive Bitrate) environment.

This module implements the ABR environment using the gymnasium API,
wrapping the Pensieve simulator components.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.simulator import Simulator


# State dimensions (used by RL agents, not by env anymore)
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


@dataclass
class Observation:
    """Raw observation data from ABR environment.

    This dataclass contains all the information from a single environment step
    that agents can use to make decisions. Unlike the state matrix used by RL
    agents, this is raw unprocessed data.

    Attributes:
        delay: Download delay in milliseconds.
        sleep_time: Sleep time in milliseconds.
        buffer_size: Current buffer size in seconds.
        rebuffer: Rebuffering time in seconds.
        video_chunk_size: Downloaded chunk size in bytes.
        next_video_chunk_sizes: List of chunk sizes at each quality level in bytes.
        video_chunk_remain: Number of remaining video chunks.
        end_of_video: Whether this is the last chunk of the video.
    """
    delay: float
    sleep_time: float
    buffer_size: float
    rebuffer: float
    video_chunk_size: float
    next_video_chunk_sizes: np.ndarray
    video_chunk_remain: int
    end_of_video: bool


class ABREnv(gym.Env):
    """Gymnasium environment for Adaptive Bitrate Streaming.

    This environment simulates video streaming over a network with
    variable bandwidth. The agent must select bitrate levels to maximize
    video quality while minimizing rebuffering and bitrate oscillations.

    Observation:
        An Observation dataclass containing raw data from each step:
        - delay: Download delay in milliseconds
        - sleep_time: Sleep time in milliseconds
        - buffer_size: Current buffer size in seconds
        - rebuffer: Rebuffering time in seconds
        - video_chunk_size: Downloaded chunk size in bytes
        - next_video_chunk_sizes: Chunk sizes at each quality level in bytes
        - video_chunk_remain: Number of remaining video chunks
        - end_of_video: Whether this is the last chunk

        Note: State computation (history tracking, normalization) is handled
        by the agent, not the environment. This allows different agents to
        use different state representations.

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
        buffer_norm_factor: float = BUFFER_NORM_FACTOR,
        initial_level: int = 0,
    ):
        """Initialize the ABR environment.

        Args:
            simulator: Pre-configured Simulator instance (use create_simulator
                      from pensieve_ppo.core to create one)
            levels_quality: Quality metric list for each bitrate level, used for
                           reward calculation. Length must match
                           simulator.video_player.bitrate_levels.
                           For example, bitrate values in Kbps: [300, 750, 1200, ...]
                           or other quality indicators like VMAF/PSNR scores.
            rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3)
            smooth_penalty: Penalty coefficient for quality changes (default: 1.0)
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
        self.buffer_norm_factor = buffer_norm_factor

        # Store initial bitrate
        self.initial_bitrate = initial_level

        # Initialize environment state (not agent state)
        self.last_bit_rate = self.initial_bitrate
        self.buffer_size = 0.0
        self.time_stamp = 0.0

        # Define action space (observation space is not a simple Box anymore)
        self.observation_space = None  # Observation is a dataclass, not a gym space
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
    ) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment to initial state.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L42-L47
        This method only initializes state and does NOT execute the first chunk download.
        The first chunk should be downloaded by calling step(action) after reset().
        This follows proper Gymnasium API separation of concerns.

        Args:
            seed: Random seed for reproducibility
            options: Additional options:
                - reset_time_stamp (bool): Whether to reset time_stamp to 0.
                  Default is True. Set to False for testing mode where time_stamp
                  should accumulate across traces.
                - initial_level (int): Override initial bitrate level for last_bit_rate.
                  If not specified, uses initial_level from __init__.

        Returns:
            Tuple of (observation, info_dict)
            The initial observation has zeros for most fields since no step has been taken.
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Parse options
        options = options or {}
        reset_time_stamp = options.get('reset_time_stamp', True)
        initial_level = options.get('initial_level', self.initial_bitrate)

        if reset_time_stamp:
            self.time_stamp = 0

        self.last_bit_rate = initial_level
        self.buffer_size = 0.

        # Create initial observation with zeros (no step taken yet)
        observation = Observation(
            delay=0.0,
            sleep_time=0.0,
            buffer_size=0.0,
            rebuffer=0.0,
            video_chunk_size=0.0,
            next_video_chunk_sizes=np.zeros(self.bitrate_levels),
            video_chunk_remain=self.total_chunk_cap,
            end_of_video=False,
        )

        info = {
            "time_stamp": self.time_stamp,
        }

        return observation, info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
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

        # Create observation with raw data
        observation = Observation(
            delay=delay,
            sleep_time=sleep_time,
            buffer_size=buffer_size,
            rebuffer=rebuf,
            video_chunk_size=video_chunk_size,
            next_video_chunk_sizes=np.array(next_video_chunk_sizes),
            video_chunk_remain=video_chunk_remain,
            end_of_video=end_of_video,
        )

        # Build info dict with all step details
        info = {
            "time_stamp": self.time_stamp,
            "quality": self.levels_quality[bit_rate],
            "reward": reward,
        }

        self.last_bit_rate = bit_rate

        # Episode termination
        terminated = result.end_of_video
        truncated = False

        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented for this env)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass

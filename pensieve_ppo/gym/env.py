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
        observer: 'ABRStateObserver',
        initial_level: int = 0,
    ):
        """Initialize the ABR environment.

        Args:
            simulator: Pre-configured Simulator instance (use create_simulator
                      from pensieve_ppo.core to create one)
            observer: ABRStateObserver instance for state observation and reward
                     calculation. Its levels_quality length must match
                     simulator.video_player.bitrate_levels.
            initial_level: Initial quality level index on reset (default: 0)
        """
        super().__init__()

        # Store simulator
        self.simulator = simulator

        # Store observer
        self.observer = observer

        # Store initial bitrate
        self.initial_bitrate = initial_level

        # timestamp in ms for logging purposes
        self.time_stamp = 0.0

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(S_INFO, observer.state_history_len),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(simulator.video_player.bitrate_levels)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
                - initial_level (int): Override initial bitrate level for last_bit_rate.
                  If not specified, uses initial_level from __init__.

        Returns:
            Tuple of (observation, info_dict)
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

        # Reset observer and get initial state
        state = self.observer.reset(self, initial_level)

        info = {
            "time_stamp": self.time_stamp,
        }

        return state, info

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

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L80-L81
        self.time_stamp += result.delay  # in ms
        self.time_stamp += result.sleep_time  # in ms

        # Observe state and compute reward
        state, reward = self.observer.observe(self, bit_rate, result)

        # Episode termination
        terminated = result.end_of_video
        truncated = False

        info = {
            "time_stamp": self.time_stamp,
        }
        return state, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented for this env)."""
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass


class ABRStateObserver:
    """Observer for ABR environment state and reward calculation.

    This class handles state representation and reward computation,
    decoupled from the environment dynamics.

    Attributes:
        levels_quality: Quality metric list for each bitrate level.
        rebuf_penalty: Penalty coefficient for rebuffering.
        smooth_penalty: Penalty coefficient for quality changes.
        state_history_len: Number of past observations in state.
        buffer_norm_factor: Normalization factor for buffer size.
        state: Current state array.
        last_bit_rate: Last selected bitrate level.
    """

    def __init__(
        self,
        levels_quality: List[float],
        rebuf_penalty: float = REBUF_PENALTY,
        smooth_penalty: float = SMOOTH_PENALTY,
        state_history_len: int = S_LEN,
        buffer_norm_factor: float = BUFFER_NORM_FACTOR,
    ):
        """Initialize the ABR state observer.

        Args:
            levels_quality: Quality metric list for each bitrate level, used for
                           state representation and reward calculation.
                           For example, bitrate values in Kbps: [300, 750, 1200, ...]
                           or other quality indicators like VMAF/PSNR scores.
            rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3)
            smooth_penalty: Penalty coefficient for quality changes (default: 1.0)
            state_history_len: Number of past observations to keep in state (default: 8)
            buffer_norm_factor: Normalization factor for buffer size in seconds (default: 10.0)
        """

        # Store reward parameters
        self.rebuf_penalty = rebuf_penalty
        self.smooth_penalty = smooth_penalty
        self.levels_quality = levels_quality

        # Store normalization parameters
        self.state_history_len = state_history_len
        self.buffer_norm_factor = buffer_norm_factor

        # State tracking (initialized in reset)
        self.state: Optional[np.ndarray] = None
        self.last_bit_rate: int = 0

    @property
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels."""
        return len(self.levels_quality)

    def reset(
        self,
        env: "ABREnv",
        initial_bit_rate: int = 0,
    ) -> np.ndarray:
        """Reset observer state and return initial observation.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial state array (zeros).
        """
        # Validate levels_quality length matches env's bitrate_levels
        if self.bitrate_levels != env.simulator.video_player.bitrate_levels:
            raise ValueError(
                f"levels_quality length ({self.bitrate_levels}) must match "
                f"env.simulator.video_player.bitrate_levels ({env.simulator.video_player.bitrate_levels})"
            )

        self.last_bit_rate = initial_bit_rate
        self.state = np.zeros((S_INFO, self.state_history_len), dtype=np.float32)
        return self.state.copy()

    def observe(
        self,
        env: 'ABREnv',
        bit_rate: int,
        result: StepResult,
    ) -> Tuple[np.ndarray, float]:
        """Process simulator result: compute reward and update state.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            Tuple of (state, reward).
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
                                  env.simulator.video_player.total_chunks) / float(env.simulator.video_player.total_chunks)

        # Update internal state
        self.state = state
        self.last_bit_rate = bit_rate

        return self.state.copy(), reward

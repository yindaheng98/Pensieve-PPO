"""RL-specific ABR State Observer.

This module implements the state observer and reward calculation for
reinforcement learning based ABR (Adaptive Bitrate) agents.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from ...core.simulator import StepResult
from ...gym.env import AbstractABRStateObserver, ABREnv


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


class RLABRStateObserver(AbstractABRStateObserver):
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
    def observation_space(self) -> spaces.Box:
        """Gymnasium observation space for the state."""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(S_INFO, self.state_history_len),
            dtype=np.float32
        )

    @property
    def bitrate_levels(self) -> int:
        """Number of available bitrate levels."""
        return len(self.levels_quality)

    def reset(
        self,
        env: ABREnv,
        initial_bit_rate: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset observer state and return initial observation.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Tuple of (state, info_dict).
        """
        # Validate levels_quality length matches env's bitrate_levels
        if self.bitrate_levels != env.simulator.video_player.bitrate_levels:
            raise ValueError(
                f"levels_quality length ({self.bitrate_levels}) must match "
                f"env.simulator.video_player.bitrate_levels ({env.simulator.video_player.bitrate_levels})"
            )

        self.last_bit_rate = initial_bit_rate
        self.state = np.zeros((S_INFO, self.state_history_len), dtype=np.float32)
        return self.state.copy(), {}

    def observe(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Process simulator result: compute reward and update state.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            Tuple of (state, reward, info_dict).
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

        # Info dict with quality for logging (matching VIDEO_BIT_RATE[bit_rate] in src/test.py)
        info = {
            'quality': self.levels_quality[bit_rate],
        }

        return self.state.copy(), reward, info

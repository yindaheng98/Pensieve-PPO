"""MPC (Model Predictive Control) Agent implementation using RobustMPC.

This module implements the MPC algorithm for adaptive bitrate streaming,
which uses harmonic mean bandwidth prediction to make optimal bitrate decisions.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py
"""

import itertools
import logging
from typing import List, Tuple


from ..abc import AbstractAgent
from .observer import MPCState


# MPC Parameters
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L11
MPC_FUTURE_CHUNK_COUNT = 5

# Video chunk length in seconds
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L206
VIDEO_CHUNK_LEN = 4.0  # seconds

# Reward parameters
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L19-L21
M_IN_K = 1000.0

# Bandwidth history length for harmonic mean calculation
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L153
BANDWIDTH_HISTORY_LEN = 5

# State index for throughput measurement (kilo byte / ms = MB/s)
# In RLABRStateObserver, state[2, :] stores float(video_chunk_size) / float(delay) / M_IN_K
STATE_THROUGHPUT_IDX = 2


class MPCAgent(AbstractAgent):
    """MPC (Model Predictive Control) Agent using RobustMPC.

    This agent implements the MPC algorithm that uses harmonic mean of past
    bandwidths to predict future bandwidth, and then selects optimal bitrates
    by simulating all possible combinations of future chunk bitrates.

    The RobustMPC algorithm:
    1. Compute harmonic mean of last 5 bandwidths
    2. Track prediction errors and apply error correction
    3. For each combination of bitrates for the next N chunks:
       - Simulate the download time using predicted bandwidth
       - Calculate rebuffering time and smoothness penalty
       - Compute total reward for the combination
    4. Select the bitrate for the current chunk from the best combination

    Note:
        rebuf_penalty and smooth_penalty are obtained from MPCState, which
        inherits these values from RLABRStateObserver.

    Attributes:
        action_dim: Number of available bitrate levels.
        future_chunk_count: Number of future chunks to consider (default: 5).
        bandwidth_history_len: Number of past bandwidths for harmonic mean (default: 5).
        video_chunk_len: Video chunk length in seconds (default: 4.0).
        past_errors: List of past bandwidth prediction errors.
        past_bandwidth_ests: List of past bandwidth estimates (harmonic mean).
    """

    def __init__(
        self,
        action_dim: int,
        future_chunk_count: int = MPC_FUTURE_CHUNK_COUNT,
        bandwidth_history_len: int = BANDWIDTH_HISTORY_LEN,
        video_chunk_len: float = VIDEO_CHUNK_LEN,
        **kwargs,
    ):
        """Initialize the MPC agent.

        Args:
            action_dim: Number of discrete actions (bitrate levels).
            future_chunk_count: Number of future chunks to consider in MPC.
            bandwidth_history_len: Number of past bandwidths for harmonic mean calculation.
            video_chunk_len: Video chunk length in seconds.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self.action_dim = action_dim
        self.future_chunk_count = future_chunk_count
        self.bandwidth_history_len = bandwidth_history_len
        self.video_chunk_len = video_chunk_len

        # Pre-compute all possible combinations of chunk bitrates
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L82-L83
        self.chunk_combo_options = list(
            itertools.product(range(action_dim), repeat=future_chunk_count)
        )

        # Track bandwidth prediction errors for RobustMPC
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L33-L34
        self.past_errors: List[float] = []
        self.past_bandwidth_ests: List[float] = []

        if kwargs:
            logging.warning(f"kwargs are ignored in MPCAgent: {kwargs}")

    def reset(self) -> None:
        """Reset the agent state for a new episode.

        Clears the bandwidth prediction error history.
        """
        self.past_errors = []
        self.past_bandwidth_ests = []

    def compute_harmonic_mean_bandwidth(self, state: MPCState) -> float:
        """Compute harmonic mean of past bandwidths.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L151-L163

        Args:
            state: MPCState containing bandwidth history.

        Returns:
            Harmonic mean bandwidth in MB/s.
        """
        # Get past bandwidths from state_matrix (last N values based on bandwidth_history_len)
        # state_matrix[2, :] contains throughput: float(video_chunk_size) / float(delay) / M_IN_K
        # which is in units of kilo bytes / ms = MB/s
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L153
        past_bandwidths = state.state_matrix[STATE_THROUGHPUT_IDX, -self.bandwidth_history_len:]

        # Remove leading zeros (from initial state)
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L154-L155
        while len(past_bandwidths) > 0 and past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]

        # Compute harmonic mean
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L160-L163
        bandwidth_sum = 0.0
        for past_val in past_bandwidths:
            bandwidth_sum += (1.0 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        return harmonic_bandwidth

    def compute_robust_bandwidth(self, state: MPCState) -> float:
        """Compute robust bandwidth prediction using error correction.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L145-L173

        Args:
            state: MPCState containing bandwidth history.

        Returns:
            Predicted future bandwidth in MB/s with error correction.
        """
        # Get current actual throughput from state_matrix
        curr_throughput = state.state_matrix[STATE_THROUGHPUT_IDX, -1]

        # Compute current prediction error
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L146-L149
        curr_error = 0.0
        if len(self.past_bandwidth_ests) > 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - curr_throughput) / float(curr_throughput)
        self.past_errors.append(curr_error)

        # Compute harmonic mean of past bandwidths
        harmonic_bandwidth = self.compute_harmonic_mean_bandwidth(state)

        # Compute max error from last N errors (based on bandwidth_history_len)
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L167-L173
        max_error = 0.0
        error_pos = -self.bandwidth_history_len
        if len(self.past_errors) < self.bandwidth_history_len:
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1.0 + max_error)  # robustMPC here
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        return future_bandwidth

    def compute_combo_reward(
        self,
        state: MPCState,
        combo: Tuple[int, ...],
        last_index: int,
        start_buffer: float,
        future_bandwidth: float,
    ) -> float:
        """Compute the total reward for a combination of bitrate choices.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L190-L216

        Args:
            state: MPCState with video chunk information.
            combo: Tuple of bitrate levels for future chunks.
            last_index: Index of the last downloaded chunk.
            start_buffer: Starting buffer size in seconds.
            future_bandwidth: Predicted future bandwidth in MB/s.

        Returns:
            Total reward for this combination.
        """
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int(state.bit_rate)
        for position in range(len(combo)):
            chunk_quality = combo[position]
            index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = (state.get_chunk_size(chunk_quality, index) / M_IN_K / M_IN_K) / future_bandwidth  # this is MB/MB/s --> seconds
            if curr_buffer < download_time:
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += self.video_chunk_len
            bitrate_sum += state.levels_quality[chunk_quality]
            smoothness_diffs += abs(state.levels_quality[chunk_quality] - state.levels_quality[last_quality])
            # bitrate_sum += BITRATE_REWARD[chunk_quality]
            # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

        reward = (bitrate_sum / M_IN_K) - (state.rebuf_penalty * curr_rebuffer_time) - (state.smooth_penalty * smoothness_diffs / M_IN_K)
        # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)

        return reward

    def select_action(self, state: MPCState) -> Tuple[int, List[float]]:
        """Select an action using MPC algorithm with predicted bandwidth.

        This method iterates through all possible combinations of bitrate
        choices for future chunks, simulates each using the predicted
        bandwidth, and selects the best combination.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L145-L230

        Args:
            state: MPCState containing current observation and
                   video chunk information.

        Returns:
            Tuple of (selected_action_index, action_probabilities).
            The action_prob is a one-hot encoding since MPC is deterministic.
        """
        # Check if state is MPCState
        if not isinstance(state, MPCState):
            raise TypeError(
                f"MPCAgent requires MPCState, got {type(state).__name__}. "
                "Use MPCABRStateObserver with this agent."
            )

        # Compute robust bandwidth prediction
        future_bandwidth = self.compute_robust_bandwidth(state)

        # Calculate future chunk length
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L176-L180
        #
        # Bug fix: The original pensieve code has an off-by-one error.
        # After get_video_chunk(), video_chunk_counter is incremented, so it points to the
        # NEXT chunk to download. The original code computes:
        #   last_index = CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain = video_chunk_counter
        # Then uses: index = last_index + position + 1
        # This means for position=0, it predicts for chunk (video_chunk_counter + 1),
        # but the next download is actually for chunk video_chunk_counter.
        #
        # Fix: Subtract 1 so last_index represents the last DOWNLOADED chunk index.
        # Then index = last_index + position + 1 correctly gives video_chunk_counter for position=0.
        last_index = state.video_chunk_counter - 1

        # Determine how many future chunks to consider
        future_chunk_length = self.future_chunk_count
        if state.total_chunks - state.video_chunk_counter < self.future_chunk_count:
            future_chunk_length = state.total_chunks - state.video_chunk_counter

        # Handle edge case: no more chunks to download
        if future_chunk_length <= 0:
            # Return default action (lowest bitrate)
            action_prob = [0.0] * self.action_dim
            action_prob[0] = 1.0
            return 0, action_prob

        # Iterate over all possible combinations and find the best
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L182-L189
        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000.0
        best_combo: Tuple[int, ...] = ()
        start_buffer = state.buffer_size
        # start = time.time()
        for full_combo in self.chunk_combo_options:
            combo = full_combo[:future_chunk_length]

            # Compute reward for this combination
            reward = self.compute_combo_reward(
                state=state,
                combo=combo,
                last_index=last_index,
                start_buffer=start_buffer,
                future_bandwidth=future_bandwidth,
            )

            # Update best if this is better
            # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc.py#L219-L230
            if (reward >= max_reward):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                if best_combo != ():  # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data

        # Return one-hot probability distribution (MPC is deterministic)
        action_prob = [0.0] * self.action_dim
        action_prob[bit_rate] = 1.0

        return bit_rate, action_prob

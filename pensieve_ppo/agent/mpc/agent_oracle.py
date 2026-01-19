"""MPC (Model Predictive Control) Agent implementation.

This module implements the MPC algorithm for adaptive bitrate streaming,
which uses future bandwidth information to make optimal bitrate decisions.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py
"""

import itertools
import logging
from typing import List, Tuple


from ..abc import AbstractAgent
from .observer_oracle import OracleMPCState


# MPC Parameters
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L11
MPC_FUTURE_CHUNK_COUNT = 7

# Video chunk length in seconds
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L213
VIDEO_CHUNK_LEN = 4.0  # seconds

# Reward parameters
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L19-L21
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0


class OracleMPCAgent(AbstractAgent):
    """MPC (Model Predictive Control) Agent.

    This agent implements the MPC algorithm that uses future bandwidth
    information to select optimal bitrates by simulating all possible
    combinations of future chunk bitrates and selecting the one with
    the highest cumulative reward.

    The algorithm:
    1. For each combination of bitrates for the next N chunks
    2. Simulate the download time using actual future bandwidth
    3. Calculate rebuffering time and smoothness penalty
    4. Compute total reward for the combination
    5. Select the bitrate for the current chunk from the best combination

    Attributes:
        action_dim: Number of available bitrate levels.
        video_bit_rate: List of bitrate values in Kbps for each level.
        future_chunk_count: Number of future chunks to consider (default: 7).
        rebuf_penalty: Penalty coefficient for rebuffering (default: 4.3).
        smooth_penalty: Penalty coefficient for bitrate changes (default: 1.0).
        video_chunk_len: Video chunk length in seconds (default: 4.0).
    """

    def __init__(
        self,
        action_dim: int,
        future_chunk_count: int = MPC_FUTURE_CHUNK_COUNT,
        rebuf_penalty: float = REBUF_PENALTY,
        smooth_penalty: float = SMOOTH_PENALTY,
        video_chunk_len: float = VIDEO_CHUNK_LEN,
        **kwargs,
    ):
        """Initialize the MPC agent.

        Args:
            action_dim: Number of discrete actions (bitrate levels).
            future_chunk_count: Number of future chunks to consider in MPC.
            rebuf_penalty: Penalty coefficient for rebuffering.
            smooth_penalty: Penalty coefficient for bitrate changes.
            video_chunk_len: Video chunk length in seconds.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self.action_dim = action_dim
        self.future_chunk_count = future_chunk_count
        self.rebuf_penalty = rebuf_penalty
        self.smooth_penalty = smooth_penalty
        self.video_chunk_len = video_chunk_len

        # Pre-compute all possible combinations of chunk bitrates
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L82-L83
        self.chunk_combo_options = list(
            itertools.product(range(action_dim), repeat=future_chunk_count)
        )
        if kwargs:
            logging.warning(f"kwargs are ignored in OracleMPCAgent: {kwargs}")

    def compute_combo_reward(
        self,
        state: OracleMPCState,
        combo: Tuple[int, ...],
        last_index: int,
        start_buffer: float,
    ) -> float:
        """Compute the total reward for a combination of bitrate choices.

        Note: The state's virtual pointers should already be reset before
        calling this method. This method only reads from and modifies
        the state's internal virtual pointers.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L190-L223

        Args:
            state: OracleState with future prediction capabilities.
                   Virtual pointers should be reset before calling.
            combo: Tuple of bitrate levels for future chunks.
            last_index: Index of the last downloaded chunk.
            start_buffer: Starting buffer size in seconds.

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

        state.reset_download_time()  # so that the next future download time starts from now

        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4

            # download_time = (get_chunk_size(chunk_quality, index)/1000000.) / future_bandwidth # this is MB/MB/s --> seconds

            download_time = state.get_download_time(state.get_chunk_size(chunk_quality, index))  # poke env to get future download time

            if (curr_buffer < download_time):
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
        # Compute reward for this combination
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

        reward = (bitrate_sum / M_IN_K) - (self.rebuf_penalty * curr_rebuffer_time) - (smoothness_diffs / M_IN_K)
        # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)

        return reward

    def select_action(self, state: OracleMPCState) -> Tuple[int, List[float]]:
        """Select an action using MPC algorithm with future bandwidth.

        This method iterates through all possible combinations of bitrate
        choices for future chunks, simulates each using the actual future
        bandwidth, and selects the best combination.

        The state is used once and discarded after this method returns.
        Virtual pointers are reset at the start of each combination evaluation.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L145-L237

        Args:
            state: OracleState containing current observation and
                   methods for future prediction.

        Returns:
            Tuple of (selected_action_index, action_probabilities).
            The action_prob is a one-hot encoding since MPC is deterministic.
        """
        # Check if state is OracleState
        if not isinstance(state, OracleMPCState):
            raise TypeError(
                f"OracleMPCAgent requires OracleState, got {type(state).__name__}. "
                "Use OracleABRStateObserver with this agent."
            )

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L176-L189
        # future chunks length (try 4 if that many remaining)
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
        # Remaining chunks = total - video_chunk_counter = total - (last_index + 1)
        future_chunk_length = self.future_chunk_count
        if (state.total_chunks - state.video_chunk_counter < self.future_chunk_count):
            future_chunk_length = state.total_chunks - state.video_chunk_counter

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = float('-inf')
        best_combo = ()
        start_buffer = state.buffer_size
        # start = time.time()
        for full_combo in self.chunk_combo_options:
            combo = full_combo[:future_chunk_length]

            # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L190-L231
            # Compute reward for this combination
            reward = self.compute_combo_reward(
                state=state.copy(),
                combo=combo,
                last_index=last_index,
                start_buffer=start_buffer,
            )

            # Update best if this is better
            # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py#L226-L237
            if (reward >= max_reward):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                if (best_combo != ()):  # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data

        # Return one-hot probability distribution (MPC is deterministic)
        action_prob = [0.0] * self.action_dim
        action_prob[bit_rate] = 1.0

        return bit_rate, action_prob

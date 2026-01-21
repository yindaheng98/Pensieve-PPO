"""MPC State Observer with Future Bandwidth Prediction.

This module provides a state observer for MPC (Model Predictive Control) algorithm
that includes future bandwidth information for decision making.

For imitation learning (e.g., training RL from Oracle MPC demonstrations), use
ImitationObserver from pensieve_ppo.gym.imitate to combine OracleMPCABRStateObserver
with an RLABRStateObserver.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/mpc_future_bandwidth.py
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py
"""

from dataclasses import dataclass, asdict


from .observer import MPCState, MPCABRStateObserver, B_IN_MB, BITS_IN_BYTE
from ...core.simulator import StepResult
from ...gym import ABREnv


@dataclass
class OracleMPCState(MPCState):
    """State class for MPC algorithm with future prediction capabilities.

    This extends MPCState to include methods for computing future download times
    using virtual pointers. The virtual pointers are copied from the Observer
    when the state is created, and only modified within this instance.

    For imitation learning, use ImitationObserver to combine this with an
    RLState-producing observer.

    Attributes:
        virtual_mahimahi_ptr: Virtual pointer for future prediction (internal).
        virtual_last_mahimahi_time: Virtual time for future prediction (internal).
    """
    virtual_mahimahi_ptr: int = None
    virtual_last_mahimahi_time: float = None

    def reset_download_time(self) -> None:
        """Reset virtual pointers to current actual pointers for future prediction.

        This method synchronizes the virtual pointers with the actual
        trace simulator pointers, preparing for a new future prediction sequence.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L56-L58
        """
        self.virtual_mahimahi_ptr = self.trace_simulator.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.trace_simulator.last_mahimahi_time

    def get_download_time(self, video_chunk_size: int) -> float:
        """Compute download time for a chunk using virtual pointers.

        This method simulates the download time without affecting the actual
        simulator state, allowing the MPC algorithm to peek into the future.
        Only modifies internal virtual pointers.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L60-L92

        Args:
            video_chunk_size: Size of the video chunk in bytes.

        Returns:
            Download time in seconds.
        """
        cooked_time = self.trace_simulator.cooked_time
        cooked_bw = self.trace_simulator.cooked_bw
        packet_payload_portion = self.trace_simulator.packet_payload_portion

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/test/fixed_env_future_bandwidth.py#L62-L92
        delay = 0.0  # in seconds
        video_chunk_counter_sent = 0  # in bytes

        while True:
            throughput = cooked_bw[self.virtual_mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = cooked_time[self.virtual_mahimahi_ptr] \
                - self.virtual_last_mahimahi_time

            packet_payload = throughput * duration * packet_payload_portion

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / packet_payload_portion
                delay += fractional_time
                self.virtual_last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.virtual_last_mahimahi_time = cooked_time[self.virtual_mahimahi_ptr]
            self.virtual_mahimahi_ptr += 1

            if self.virtual_mahimahi_ptr >= len(cooked_bw):
                # loop back to the beginning
                # note: trace file starts with time 0
                self.virtual_mahimahi_ptr = 1
                self.virtual_last_mahimahi_time = 0

        return delay


class OracleMPCABRStateObserver(MPCABRStateObserver):
    """State observer for MPC algorithm with future bandwidth prediction.

    This observer extends MPCABRStateObserver to provide OracleMPCState objects
    that include methods for computing future download times, enabling the
    MPC algorithm to plan ahead using actual future bandwidth information.

    Example for standalone Oracle MPC:
        >>> observer = OracleMPCABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> env = ABREnv(simulator=simulator, observer=observer)

    Example for imitation learning:
        >>> from pensieve_ppo.gym.imitate import ImitationObserver
        >>> rl_observer = RLABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> oracle_observer = OracleMPCABRStateObserver(levels_quality=VIDEO_BIT_RATE)
        >>> imitation_observer = ImitationObserver(rl_observer, oracle_observer)
        >>> env = ABREnv(simulator=simulator, observer=imitation_observer)
    """

    def build_and_set_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> OracleMPCState:
        """Build initial OracleMPCState on reset.

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial OracleMPCState with synchronized virtual pointers.
        """
        state = OracleMPCState(
            **asdict(super().build_and_set_initial_state(env, initial_bit_rate)),
        )
        state.reset_download_time()
        return state

    def compute_and_update_state(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
    ) -> OracleMPCState:
        """Compute new OracleMPCState from simulator result.

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().

        Returns:
            New OracleMPCState with synchronized virtual pointers.
        """
        state = OracleMPCState(
            **asdict(super().compute_and_update_state(env, bit_rate, result)),
        )
        state.reset_download_time()
        return state

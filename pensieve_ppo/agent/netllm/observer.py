"""NetLLM State Observer for NetLLM-based ABR agents.

This module provides a state observer for NetLLM (Decision Transformer-based)
ABR agents that provides state arrays compatible with NetLLM's input format.

The observer extends RLABRStateObserver (which handles state computation and
reward calculation) and adds NetLLM-specific fields:
- timestep: Current timestep within episode (for positional embedding)
- target_return: Return-to-go value (for conditional generation)

Copy from:
    https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py
    https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..rl import RLABRStateObserver
from ...core.simulator import StepResult
from ...gym import ABREnv

# Reference: NetLLM/adaptive_bitrate_streaming/baseline_special/utils/constants.py#L15
S_LEN = 6


@dataclass
class NetLLMState:
    """State class for NetLLM agents.

    Wraps the numpy state array with NetLLM-specific fields for
    Decision Transformer inference.

    Attributes:
        state: Numpy array with shape (S_INFO, S_LEN).
        timestep: Current timestep within episode.
        target_return: Current return-to-go value.

    Copy from:
        https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L25-L29
        https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L28-L32
    """
    state: np.ndarray
    timestep: int
    target_return: float

    def copy(self) -> 'NetLLMState':
        """Create a copy of this NetLLMState."""
        return NetLLMState(
            state=self.state.copy(),
            timestep=self.timestep,
            target_return=self.target_return,
        )


class NetLLMABRStateObserver(RLABRStateObserver):
    """State observer for NetLLM-based ABR agents.

    Extends RLABRStateObserver to add NetLLM-specific tracking:
    - timestep: Tracks position within episode for timestep embedding
    - target_return: Tracks return-to-go for conditional generation

    The state computation and reward calculation are inherited from
    RLABRStateObserver, which implements the same logic as NetLLM's
    evaluate.py and test.py.

    Copy from:
        https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py
        https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py
    """

    def __init__(
        self,
        *args,
        state_history_len: int = S_LEN,
        max_return: float = 0.0,
        **kwargs,
    ):
        """Initialize the NetLLM state observer.

        Args:
            max_return: Initial target return-to-go value for episodes.
                       Copy from: evaluate.py#L13 target_return parameter
            *args, **kwargs: Passed to RLABRStateObserver.
        """
        super().__init__(*args, state_history_len=state_history_len, **kwargs)

        self.max_return = max_return

        # NetLLM-specific tracking
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L26-L27
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L29-L30
        self.timestep: int = 0
        self.target_return: float = max_return

    def build_and_set_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> NetLLMState:
        """Build initial NetLLMState on reset.

        Copy from:
            evaluate.py#L25: state = torch.zeros((1, 1, S_INFO, S_LEN), ...)
            evaluate.py#L26-27: timestep = 0, target_return_clone = copy.deepcopy(target_return)

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial NetLLMState with zero state array.
        """
        # Reset NetLLM-specific state
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L26-L27
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L29-L30
        self.timestep = 0
        self.target_return = self.max_return

        # Get initial state from parent (numpy array)
        # Parent sets self.state and returns it
        numpy_state = super().build_and_set_initial_state(env, initial_bit_rate)

        return NetLLMState(
            state=numpy_state,
            timestep=self.timestep,
            target_return=self.target_return,
        )

    def observe(
        self,
        env: ABREnv,
        bit_rate: int,
        result: StepResult,
        process_reward_fn: Optional[Callable[[float], float]] = None,
    ) -> Tuple[NetLLMState, float, Dict[str, Any]]:
        """Process simulator result: compute reward and update state.

        Follows the exact order of operations from NetLLM's evaluate.py/test.py:
        1. Compute reward (using last_bit_rate from previous step)
        2. Update state (roll and fill new values)
        3. Update target_return (skip first timestep, then subtract reward)
        4. Update last_bit_rate
        5. Increment timestep

        Copy from:
            evaluate.py#L38-L63
            test.py#L47-L77

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().
            process_reward_fn: Optional function to process reward before
                              updating target_return.
                              Copy from: evaluate.py#L14-L15, test.py#L15-L16

        Returns:
            Tuple of (NetLLMState_copy, reward, info_dict).
        """
        # Step 1: Compute reward
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L39-L41
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L48-L50
        reward = self.compute_reward(env, bit_rate, result)

        # Step 2: Update state (dequeue history record)
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L46-L54
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L60-L68
        numpy_state = self.compute_and_update_state(env, bit_rate, result)

        # Step 3: Update target_return (skip first timestep like Pensieve)
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L56-L58
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L70-L72
        if self.timestep > 0:
            processed_reward = process_reward_fn(reward) if process_reward_fn else reward
            self.target_return -= processed_reward

        # Step 4: Update last_bit_rate for next step's reward calculation
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L43
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L54
        self.last_bit_rate = bit_rate

        # Step 5: Increment timestep
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/evaluate.py#L63
        # https://github.com/duowuyms/NetLLM/blob/main/adaptive_bitrate_streaming/plm_special/test.py#L77
        self.timestep += 1

        # Build state object
        state = NetLLMState(
            state=numpy_state,
            timestep=self.timestep,
            target_return=self.target_return,
        )

        # Build info dict
        info = self.build_info_dict(env, bit_rate, result)

        return state.copy(), reward, info

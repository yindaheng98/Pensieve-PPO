"""NetLLM State Observer for NetLLM-based ABR agents.

This module provides a state observer for NetLLM (Decision Transformer-based)
ABR agents. The observer is responsible for collecting raw data needed for
both training and inference.

Raw data collection:
- state: State array (S_INFO x S_LEN)
- action: Bitrate level selected
- reward: Step reward
- done: Episode termination flag

For inference, additional tracking:
- timestep: Current timestep within episode
- target_return: Current return-to-go value

The processing of raw data into training batches (computing returns, normalizing
rewards, etc.) is done in AbstractNetLLMAgent.produce_training_batch(), not here.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


from ..rl import RLABRStateObserver, RLState
from ..rl.observer import S_LEN
from ...core.simulator import StepResult
from ...gym import ABREnv


@dataclass
class NetLLMState(RLState):
    """State class for NetLLM agents containing raw data.

    This dataclass extends RLState to include additional fields needed for
    NetLLM training and inference. It inherits state_matrix from RLState
    for compatibility with RL-based state observers.

    For training (ExperiencePool format):
    - state_matrix: State array (inherited from RLState)
    - action: Bitrate action
    - reward: Step reward
    - done: Episode termination flag

    For inference (rl_policy.sample() parameters):
    - timestep: Current timestep within episode
    - target_return: Return-to-go value

    Note: The processing of raw data into training batches (normalization,
    computing discounted returns, etc.) is done in AbstractNetLLMAgent.produce_training_batch().

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py#L2-L16
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L266-L269
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L145-L148

    Attributes:
        state_matrix: Numpy array with shape (S_INFO, S_LEN), e.g., (6, 6).
            Raw state observation from the environment. (Inherited from RLState)
            Reference: generate_exp_pool.py#L276-L283 (state computation)
        action: Current action (bitrate level).
            Reference: generate_exp_pool.py#L267
        reward: Current step reward (raw, unnormalized).
            Reference: generate_exp_pool.py#L260-L262, #L268
        done: Whether episode has ended (end_of_video).
            Reference: generate_exp_pool.py#L269
        timestep: Current timestep within episode (for positional embedding).
            Reference: evaluate.py#L26, dataset.py#L106
        target_return: Current return-to-go value (for conditional generation).
            Reference: evaluate.py#L27 (target_return_clone), rl_policy.py#L145
    """
    # Raw data fields for training (ExperiencePool format)
    # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py#L12-L16
    # state_matrix is inherited from RLState
    action: int         # Bitrate action
    reward: float       # Raw step reward (unnormalized)
    done: bool          # Episode termination flag

    # Raw data fields for inference
    # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L145-L148
    timestep: int           # Timestep within episode
    target_return: float    # Return-to-go value


class NetLLMABRStateObserver(RLABRStateObserver):
    """State observer for NetLLM-based ABR agents.

    This observer extends RLABRStateObserver to collect raw data needed for
    both training and inference. It is responsible ONLY for raw data collection,
    NOT for processing data into training batches.

    Raw data collection (done here):
    - State computation (inherited from RLABRStateObserver)
    - Reward calculation (inherited from RLABRStateObserver)
    - Timestep tracking for inference
    - Return-to-go tracking for inference

    Data processing (done in AbstractNetLLMAgent.produce_training_batch):
    - Reward normalization
    - Discounted return computation
    - Timestep computation for training
    - Tensor conversion

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/test.py
    """

    def __init__(
        self,
        *args,
        state_history_len: int = S_LEN,
        target_return: float = 0.0,
        **kwargs,
    ):
        """Initialize the NetLLM state observer.

        Args:
            state_history_len: History length for state (S_LEN).
                Default is 6 for NetLLM (vs 8 for Pensieve-PPO).
                Reference: baseline_special/utils/constants.py#L15
            target_return: Initial return-to-go value for inference episodes.
                Used as target_return in rl_policy.sample().
                Reference: evaluate.py#L13 target_return parameter
            *args, **kwargs: Passed to RLABRStateObserver.
        """
        super().__init__(*args, state_history_len=state_history_len, **kwargs)

        # Inference tracking variables (reset in build_and_set_initial_state)
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L26-L27
        self.timestep: int = 0
        self.target_return_clone = target_return
        self.target_return: float = target_return

        # Episode statistics tracking (reset in build_and_set_initial_state)
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L29
        self.episodes_return: float = 0.0
        self.episodes_len: int = 0

    def build_and_set_initial_state(
        self,
        env: ABREnv,
        initial_bit_rate: int,
    ) -> NetLLMState:
        """Build initial NetLLMState on reset.

        Initializes the raw data fields. For training, the initial state
        typically has reward=0.0 and done=False.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L25-L27
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L245-L247

        Args:
            env: The ABREnv instance to observe.
            initial_bit_rate: Initial bitrate level index.

        Returns:
            Initial NetLLMState with zero state array and initialized fields.
        """
        # Reset inference tracking variables
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L26-L27
        self.timestep = 0
        self.target_return = self.target_return_clone

        # Reset episode statistics
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L29
        self.episodes_return = 0.0
        self.episodes_len = 0

        # Get initial state from parent (creates zero state array)
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L247
        rl_state = super().build_and_set_initial_state(env, initial_bit_rate)

        # Build NetLLMState with raw data
        return NetLLMState(
            state_matrix=rl_state.state_matrix,
            action=initial_bit_rate,
            reward=0.0,  # No reward at initial state
            done=False,  # Episode just started
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
        """Collect raw data from environment step.

        This method collects raw data needed for both training and inference:
        1. Computes raw reward (using last_bit_rate from previous step)
        2. Updates state array (roll and fill new values)
        3. Updates return-to-go for inference (subtract processed reward)
        4. Updates last_bit_rate
        5. Increments timestep

        Note: The reward stored in NetLLMState is the RAW reward. Reward
        normalization and return computation for training is done in
        AbstractNetLLMAgent.produce_training_batch().

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L251-L270
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L38-L63

        Args:
            env: The ABREnv instance to observe.
            bit_rate: Current bitrate level selected.
            result: Result from simulator.step().
            process_reward_fn: Optional function to process reward before
                updating return-to-go for inference.
                Reference: evaluate.py#L14-L15

        Returns:
            Tuple of (NetLLMState, reward, info_dict).
        """
        # Step 1: Compute raw reward
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L260-L262
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L39-L41
        reward = self.compute_reward(env, bit_rate, result)

        # Step 2: Update state array (roll and fill new values)
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L273-L283
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L46-L54
        rl_state = self.compute_and_update_state(env, bit_rate, result)

        # Step 3: Update return-to-go for inference (skip first timestep like Pensieve)
        # Also accumulate episode statistics
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L56-L60
        if self.timestep > 0:
            processed_reward = process_reward_fn(reward) if process_reward_fn else reward
            self.target_return -= processed_reward
            self.episodes_return += processed_reward
            self.episodes_len += 1

        # Step 4: Update last_bit_rate for next step's reward calculation
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L264
        self.last_bit_rate = bit_rate

        # Get done flag from result
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L252-L254
        done = result.end_of_video

        # Build NetLLMState with raw data
        # Note: Use current timestep BEFORE incrementing, matching evaluate.py
        # where model.sample(state, target_return, timestep) is called before timestep += 1
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L62-L63
        state = NetLLMState(
            state_matrix=rl_state.state_matrix,
            action=bit_rate,
            reward=reward,  # Raw reward (unnormalized)
            done=done,
            timestep=self.timestep,  # Use timestep BEFORE increment
            target_return=self.target_return,
        )

        # Step 5: Increment timestep AFTER building state
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L63
        self.timestep += 1

        # Build info dict with episode statistics
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L76-L80
        info = self.build_info_dict(env, bit_rate, result)
        info['episodes_return'] = self.episodes_return
        info['episodes_len'] = self.episodes_len

        return state, reward, info

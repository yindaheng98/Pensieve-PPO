"""NetLLM Agent implementation.

This module provides the concrete NetLLMAgent class that combines:
- AbstractNetLLMAgent: Training and inference interface
- OfflineRLPolicy: Decision Transformer-style policy network

The NetLLMAgent serves as a container that delegates model operations
to the underlying OfflineRLPolicy while providing the training infrastructure
defined in AbstractNetLLMAgent.

Supports both LoRA (Low-Rank Adaptation) and full fine-tuning modes:
- LoRA mode (rank > 0): Freezes PLM parameters and only trains LoRA adapters
- Full fine-tuning mode (rank = -1): Trains all parameters

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/low_rank.py
"""

import os
from typing import Any, Optional

import torch
import torch.nn as nn

from .abc import AbstractNetLLMAgent
from .models.rl_policy import OfflineRLPolicy
from .models.state_encoder import EncoderNetwork
from .models.low_rank import peft_model

# Reference: NetLLM/adaptive_bitrate_streaming/baseline_special/utils/constants.py#L15
S_LEN = 6
A_DIM = 6  # Number of bitrate levels
S_INFO = 6  # Number of state information types


def _detect_plm_type(plm: nn.Module) -> str:
    """Auto-detect PLM type from the model class name.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/low_rank.py#L7-L14

    Args:
        plm: Pre-trained language model instance.

    Returns:
        PLM type string for LoRA target modules selection.

    Raises:
        ValueError: If PLM type cannot be detected.
    """
    class_name = plm.__class__.__name__.lower()

    # Map class names to plm_type
    if 'gpt2' in class_name:
        return 'gpt2'
    elif 'llama' in class_name:
        return 'llama'
    elif 'llava' in class_name:
        return 'llava'
    elif 'mistral' in class_name:
        return 'mistral'
    elif 'opt' in class_name:
        return 'opt'
    elif 't5' in class_name:
        return 't5-lm'
    else:
        raise ValueError(
            f"Cannot auto-detect PLM type from class name '{plm.__class__.__name__}'. "
            f"Supported types: gpt2, llama, llava, mistral, opt, t5-lm"
        )


class NetLLMAgent(AbstractNetLLMAgent):
    """Concrete NetLLM agent implementation.

    This class combines AbstractNetLLMAgent with OfflineRLPolicy to provide
    a complete Decision Transformer-style agent for ABR streaming.

    The class delegates model forward/sample operations to the internal
    OfflineRLPolicy while using AbstractNetLLMAgent's training infrastructure.

    Supports two training modes controlled by the `rank` parameter:
    - LoRA mode (rank > 0): Freezes PLM and only trains LoRA adapters + task modules
    - Full fine-tuning (rank = -1): Trains all parameters including PLM

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L130-L231
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L11-L63
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/low_rank.py

    Attributes:
        _policy: The underlying OfflineRLPolicy network.
        _optimizer: PyTorch optimizer for training.
        _lr_scheduler: Learning rate scheduler (optional).
        _rank: LoRA rank (-1 for full fine-tuning, >0 for LoRA).

    Example:
        >>> from pensieve_ppo.agent.netllm import NetLLMAgent
        >>> from pensieve_ppo.agent.netllm.models import GPT2Model
        >>>
        >>> # Create PLM backbone
        >>> plm = GPT2Model.from_pretrained('gpt2')
        >>> plm_embed_size = plm.config.n_embd
        >>>
        >>> # Create agent with LoRA (rank=128)
        >>> agent = NetLLMAgent(
        ...     action_dim=6,
        ...     min_reward=-10.0,
        ...     max_reward=10.0,
        ...     plm=plm,
        ...     plm_embed_size=plm_embed_size,
        ...     rank=128,  # Enable LoRA with rank 128
        ...     learning_rate=1e-4,
        ... )
        >>>
        >>> # Create agent without LoRA (full fine-tuning)
        >>> agent_full = NetLLMAgent(
        ...     action_dim=6,
        ...     min_reward=-10.0,
        ...     max_reward=10.0,
        ...     plm=plm,
        ...     plm_embed_size=plm_embed_size,
        ...     rank=-1,  # Disable LoRA
        ...     learning_rate=1e-4,
        ... )
    """

    def __init__(
        self,
        state_dim: tuple[int, int] = (S_INFO, S_LEN),
        action_dim: int = A_DIM,
        *args,
        plm: nn.Module,
        plm_embed_size: int,
        # Default values from NetLLM reference implementation:
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L249-L255
        state_feature_dim: int = 256,   # run_plm.py#L249
        max_length: int = 20,           # run_plm.py#L251
        gamma: float = 1.0,             # run_plm.py#L252
        learning_rate: float = 1e-4,    # run_plm.py#L253
        weight_decay: float = 1e-4,     # run_plm.py#L254
        warmup_steps: int = 2000,       # run_plm.py#L255
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_ep_len: int = 100,
        conv_size: int = 4,
        residual: bool = False,
        which_layer: int = -1,
        # LoRA settings
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L247
        rank: int = -1,
        **kwargs,
    ):
        """Initialize the NetLLM agent.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L71-L83
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L193
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L12-L21

        Args:
            state_dim: State dimension as (num_features, sequence_length).
                Must match (S_INFO, S_LEN) = (6, 6).
            action_dim: Number of discrete actions (bitrate levels).
            *args: Positional arguments passed to AbstractNetLLMAgent
                (min_reward, max_reward).
            plm: Pre-trained language model backbone (e.g., GPT2Model).
            plm_embed_size: Embedding dimension of the PLM.
            state_feature_dim: Dimension of state encoder features.
                Reference: run_plm.py#L249 (default 256)
            max_length: Maximum sequence length (w value in paper).
                Reference: run_plm.py#L251 (default 20)
            gamma: Discount factor for return computation.
                Reference: run_plm.py#L252 (default 1.0)
            learning_rate: Learning rate for optimizer.
                Reference: run_plm.py#L253 (default 1e-4)
            weight_decay: Weight decay for optimizer.
                Reference: run_plm.py#L254 (default 1e-4)
            warmup_steps: Number of warmup steps for learning rate scheduler.
                Reference: run_plm.py#L255 (default 2000)
                Set to 0 to disable warmup/scheduler.
            device: Device to run the model on ('cuda' or 'cpu').
            max_ep_len: Maximum episode length for timestep embedding.
                Reference: run_plm.py#L191
            conv_size: Convolution kernel size for state encoder.
                Reference: state_encoder.py#L14 (default 4)
            residual: Whether to use residual connection in policy.
                Reference: rl_policy.py#L26
            which_layer: Which PLM layer to stop at (-1 for all layers).
                Reference: run_plm.py#L260, rl_policy.py#L28
            rank: LoRA rank for parameter-efficient fine-tuning.
                Reference: run_plm.py#L247
                - rank = -1: Disable LoRA, full fine-tuning (all PLM params trainable)
                - rank > 0: Enable LoRA with specified rank (e.g., 128)
            **kwargs: Additional arguments passed to AbstractNetLLMAgent
                (return_scale, loss_fn, grad_clip, grad_accum_steps).
        """
        # Initialize AbstractNetLLMAgent
        super().__init__(
            action_dim,
            *args,
            device=device,
            max_length=max_length,
            gamma=gamma,
            **kwargs,
        )

        # Store LoRA configuration
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L182
        self._rank = rank

        # Validate state_dim matches expected dimensions
        num_features, sequence_length = state_dim
        assert sequence_length == S_LEN, f"sequence_length ({sequence_length}) must equal S_LEN ({S_LEN})"
        assert num_features == S_INFO, f"num_features ({num_features}) must equal S_INFO ({S_INFO})"
        assert action_dim == A_DIM, f"action_dim ({action_dim}) must equal A_DIM ({A_DIM})"

        # Apply LoRA if enabled (before creating OfflineRLPolicy)
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L181-L182
        # This freezes PLM parameters and injects LoRA adapters
        if self._rank != -1:
            plm_type = _detect_plm_type(plm)
            plm = peft_model(plm, plm_type, rank=rank, print_trainable=True)

        # Create state encoder
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L186-L187
        state_encoder = EncoderNetwork(
            conv_size=conv_size,
            bitrate_levels=action_dim,
            embed_dim=state_feature_dim,
        ).to(device)

        # Create the OfflineRLPolicy
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L192-L193
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L14-L74
        self._policy = OfflineRLPolicy(
            state_feature_dim=state_feature_dim,
            bitrate_levels=action_dim,
            state_encoder=state_encoder,
            plm=plm,
            plm_embed_size=plm_embed_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            device=device,
            residual=residual,
            conv_size=conv_size,
            which_layer=which_layer,
        )

        # Create optimizer
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L72-L76
        # Note: In LoRA mode, model.parameters() only returns trainable parameters
        # (LoRA adapters + task-specific modules like state_encoder, action_head, etc.)
        # because PLM parameters have requires_grad=False after peft_model() call
        self._optimizer = torch.optim.AdamW(
            self._policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Create learning rate scheduler (optional warmup)
        # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L77-L80
        if warmup_steps > 0:
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
            )
        else:
            self._lr_scheduler = None

    # =========================================================================
    # AbstractNetLLMAgent Abstract Method Implementations
    # =========================================================================

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model for training.

        Delegates to OfflineRLPolicy.forward().

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L76-L144
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L60

        Args:
            states: Batch of states with shape (1, seq_len, S_INFO, S_LEN).
            actions: Batch of actions with shape (1, seq_len, 1).
            returns: Batch of return-to-go values with shape (1, seq_len, 1).
            timesteps: Batch of timesteps with shape (1, seq_len).

        Returns:
            Predicted action logits with shape (1, seq_len, action_dim).
        """
        return self._policy.forward(states, actions, returns, timesteps)

    def sample(
        self,
        state: torch.Tensor,
        target_return: float,
        timestep: int,
    ) -> int:
        """Sample an action from the model for inference.

        Delegates to OfflineRLPolicy.sample().

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L146-L215
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L62

        Args:
            state: Current state tensor with shape (1, 1, S_INFO, S_LEN).
            target_return: Current return-to-go value.
            timestep: Current timestep within the episode.

        Returns:
            Selected bitrate index.
        """
        return self._policy.sample(state, target_return, timestep)

    def reset(self) -> None:
        """Reset internal state for new episode.

        Clears the embedding deques in OfflineRLPolicy for autoregressive inference.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L217-L224
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L68-L70
        """
        self._policy.clear_dq()

    def get_params(self) -> Any:
        """Get the current network parameters.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L48-L56

        Returns:
            State dict of the policy network.
        """
        return self._policy.state_dict()

    def set_params(self, params: Any) -> None:
        """Set the network parameters.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L59-L68

        Args:
            params: State dict of the policy network.
        """
        self._policy.load_state_dict(params)

    # =========================================================================
    # Save/Load Methods (Override AbstractTrainableAgent)
    # =========================================================================

    def save(self, save_dir: str) -> None:
        """Save the model to a directory.

        In LoRA mode (rank > 0), saves LoRA weights and other modules separately.
        In full fine-tuning mode (rank = -1), saves the entire model.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L48-L56

        Args:
            save_dir: Directory to save the model.
        """
        if self._rank > 0:
            # save lora weights
            self._policy.plm.save_pretrained(save_dir)
            # save other modules except plm
            torch.save(self._policy.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
        else:
            # lora is disabled, save whole model
            torch.save(self._policy.state_dict(), os.path.join(save_dir, 'model.bin'))

    def load(self, model_dir: str) -> None:
        """Load the model from a directory.

        In LoRA mode (rank > 0), loads LoRA weights and other modules separately.
        In full fine-tuning mode (rank = -1), loads the entire model.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L59-L68

        Args:
            model_dir: Directory to load the model from.
        """
        if self._rank > 0:
            # load lora weights
            self._policy.plm.load_adapter(model_dir, adapter_name='default')
            # load other modules except plm
            self._policy.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
        else:
            # lora is disabled, load whole model
            self._policy.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))

    # =========================================================================
    # AbstractNetLLMAgent Abstract Property Implementations
    # =========================================================================

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer for training.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L72-L76
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L15

        Returns:
            PyTorch AdamW optimizer instance.
        """
        return self._optimizer

    @property
    def model(self) -> nn.Module:
        """Get the model module for training.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L14
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L33

        Returns:
            The OfflineRLPolicy nn.Module instance.
        """
        return self._policy

    @property
    def lr_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Get the learning rate scheduler for training.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L77-L80
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L21
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L45-L46

        Returns:
            PyTorch lr_scheduler instance or None if no scheduler is used.
        """
        return self._lr_scheduler

"""NetLLM Agent implementation.

This module provides the concrete NetLLMAgent class that combines:
- AbstractNetLLMAgent: Training and inference interface
- OfflineRLPolicy: Decision Transformer-style policy network

The NetLLMAgent serves as a container that delegates model operations
to the underlying OfflineRLPolicy while providing the training infrastructure
defined in AbstractNetLLMAgent.

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from .abc import AbstractNetLLMAgent
from .models.rl_policy import OfflineRLPolicy
from .models.state_encoder import EncoderNetwork


class NetLLMAgent(AbstractNetLLMAgent):
    """Concrete NetLLM agent implementation.

    This class combines AbstractNetLLMAgent with OfflineRLPolicy to provide
    a complete Decision Transformer-style agent for ABR streaming.

    The class delegates model forward/sample operations to the internal
    OfflineRLPolicy while using AbstractNetLLMAgent's training infrastructure.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L130-L231
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L11-L63

    Attributes:
        _policy: The underlying OfflineRLPolicy network.
        _optimizer: PyTorch optimizer for training.
        _lr_scheduler: Learning rate scheduler (optional).

    Example:
        >>> from pensieve_ppo.agent.netllm import NetLLMAgent
        >>> from pensieve_ppo.agent.netllm.models import GPT2Model
        >>>
        >>> # Create PLM backbone
        >>> plm = GPT2Model.from_pretrained('gpt2')
        >>> plm_embed_size = plm.config.n_embd
        >>>
        >>> # Create agent
        >>> agent = NetLLMAgent(
        ...     action_dim=6,
        ...     min_reward=-10.0,
        ...     max_reward=10.0,
        ...     plm=plm,
        ...     plm_embed_size=plm_embed_size,
        ...     learning_rate=1e-4,
        ... )
    """

    def __init__(
        self,
        action_dim: int,
        min_reward: float,
        max_reward: float,
        plm: nn.Module,
        plm_embed_size: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        gamma: float = 1.0,
        return_scale: float = 10.0,
        max_length: int = 30,
        max_ep_len: int = 100,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        grad_clip: float = 0.25,
        grad_accum_steps: int = 1,
        state_feature_dim: int = 128,
        conv_size: int = 4,
        residual: bool = False,
        which_layer: int = -1,
    ):
        """Initialize the NetLLM agent.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L71-L83
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L184-L193
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L12-L21

        Args:
            action_dim: Number of discrete actions (bitrate levels).
            min_reward: Global minimum reward for normalization.
            max_reward: Global maximum reward for normalization.
            plm: Pre-trained language model backbone (e.g., GPT2Model).
            plm_embed_size: Embedding dimension of the PLM.
            learning_rate: Learning rate for optimizer.
                Reference: run_plm.py#L253 (default 1e-4)
            weight_decay: Weight decay for optimizer.
                Reference: run_plm.py#L254 (default 1e-4)
            warmup_steps: Number of warmup steps for learning rate scheduler.
                Reference: run_plm.py#L255 (default 2000)
                Set to 0 to disable warmup/scheduler.
            device: Device to run the model on ('cuda' or 'cpu').
            gamma: Discount factor for return computation.
                Reference: run_plm.py#L252 (default 1.)
            return_scale: Scale factor for returns.
                Reference: run_plm.py#L266 (default 1000)
            max_length: Maximum sequence length (w value in paper).
                Reference: run_plm.py#L251 (default 20)
            max_ep_len: Maximum episode length for timestep embedding.
                Reference: run_plm.py#L191
            loss_fn: Loss function for training.
                Reference: run_plm.py#L81
            grad_clip: Gradient clipping value.
                Reference: trainer.py#L41 (0.25)
            grad_accum_steps: Number of steps to accumulate gradients.
                Reference: run_plm.py#L264 (default 32)
            state_feature_dim: Dimension of state encoder features.
                Reference: run_plm.py#L249 (default 256)
            conv_size: Convolution kernel size for state encoder.
                Reference: state_encoder.py#L14 (default 4)
            residual: Whether to use residual connection in policy.
                Reference: rl_policy.py#L26
            which_layer: Which PLM layer to stop at (-1 for all layers).
                Reference: run_plm.py#L260, rl_policy.py#L28
        """
        # Initialize AbstractNetLLMAgent
        super().__init__(
            action_dim=action_dim,
            min_reward=min_reward,
            max_reward=max_reward,
            device=device,
            gamma=gamma,
            return_scale=return_scale,
            max_length=max_length,
            loss_fn=loss_fn,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
        )

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
            Dictionary containing:
            - 'policy_state_dict': State dict of the policy network
            - 'optimizer_state_dict': State dict of the optimizer
            - 'lr_scheduler_state_dict': State dict of the LR scheduler (if exists)
        """
        params = {
            'policy_state_dict': self._policy.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }
        if self._lr_scheduler is not None:
            params['lr_scheduler_state_dict'] = self._lr_scheduler.state_dict()
        return params

    def set_params(self, params: Any) -> None:
        """Set the network parameters.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/run_plm.py#L59-L68

        Args:
            params: Dictionary containing:
            - 'policy_state_dict': State dict of the policy network
            - 'optimizer_state_dict': State dict of the optimizer (optional)
            - 'lr_scheduler_state_dict': State dict of the LR scheduler (optional)
        """
        self._policy.load_state_dict(params['policy_state_dict'])
        if 'optimizer_state_dict' in params:
            self._optimizer.load_state_dict(params['optimizer_state_dict'])
        if self._lr_scheduler is not None and 'lr_scheduler_state_dict' in params:
            self._lr_scheduler.load_state_dict(params['lr_scheduler_state_dict'])

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

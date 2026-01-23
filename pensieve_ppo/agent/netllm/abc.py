"""Abstract base classes for NetLLM agents.

This module provides the abstract base classes and data structures for NetLLM-based
ABR agents that use Decision Transformer-style architecture.

The module implements the separation between:
- Raw data collection: Done by NetLLMABRStateObserver (observer.py)
- Data processing: Done by produce_training_batch() (this module)
- Model interface: Abstract methods forward() and sample() (this module)
- Training: Done by train_batch() (this module)
- Inference: Done by select_action() (this module)

Reference:
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/utils/utils.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..trainable import Step, TrainingBatch, AbstractTrainableAgent
from .observer import NetLLMState


def discount_returns(rewards: List[float], gamma: float, scale: float) -> List[float]:
    """Compute discounted returns from rewards.

    This function computes the discounted cumulative return for each timestep,
    using the standard RL return formula: R_t = r_t + gamma * R_{t+1}

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L5-L12

    Args:
        rewards: List of rewards for each timestep.
        gamma: Discount factor (0 < gamma <= 1).
        scale: Scale factor to normalize returns.

    Returns:
        List of discounted returns, scaled by 1/scale.
    """
    returns = [0.0 for _ in range(len(rewards))]
    returns[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        returns[i] = rewards[i] + gamma * returns[i + 1]
    # Scale down return
    for i in range(len(returns)):
        returns[i] /= scale  # scale down return
    return returns


@dataclass
class NetLLMTrainingBatch(TrainingBatch):
    """A batch of processed training data for NetLLM.

    This class contains the processed tensors ready for model.forward().
    The processing includes:
    - Reward normalization
    - Discounted return computation
    - Timestep computation
    - Conversion to torch tensors with correct shapes

    The tensor shapes follow the format expected by rl_policy.forward():
    - states: (1, seq_len, S_INFO, S_LEN)
    - actions: (1, seq_len, 1) - normalized to (action + 1) / action_dim
    - returns: (1, seq_len, 1)
    - timesteps: (1, seq_len)
    - labels: (1, seq_len) - original action indices for loss computation

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/utils/utils.py#L11-L24
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L75-L143

    Attributes:
        states: State tensors with shape (1, seq_len, S_INFO, S_LEN).
        actions: Action tensors with shape (1, seq_len, 1), normalized.
        returns: Return-to-go tensors with shape (1, seq_len, 1).
        timesteps: Timestep tensors with shape (1, seq_len).
        labels: Label tensors with shape (1, seq_len) for loss computation.
    """
    states: torch.Tensor      # (1, seq_len, S_INFO, S_LEN)
    actions: torch.Tensor     # (1, seq_len, 1) - normalized
    returns: torch.Tensor     # (1, seq_len, 1)
    timesteps: torch.Tensor   # (1, seq_len)
    labels: torch.Tensor      # (1, seq_len)


class AbstractNetLLMAgent(AbstractTrainableAgent):
    """Abstract base class for NetLLM agents.

    This class extends AbstractTrainableAgent with NetLLM-specific interface
    for Decision Transformer-style training and inference.

    The class implements the separation of concerns:
    - Raw data collection: Done by NetLLMABRStateObserver
    - Data processing: Done by produce_training_batch()
    - Model interface: Abstract methods forward() and sample()
    - Training: Done by train_batch()
    - Inference: Done by select_action()

    Subclasses must implement:
    - forward(): Forward pass through model (for training)
    - sample(): Sample action from model (for inference)
    - reset(): Reset internal state for new episode
    - get_params() / set_params(): Model parameter access

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py
    """

    def __init__(
        self,
        action_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        gamma: float = 1.0,
        return_scale: float = 10.0,
        max_length: int = 30,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        grad_clip: float = 0.25,
    ):
        """Initialize the NetLLM agent.

        Args:
            action_dim: Number of discrete actions (bitrate levels).
            device: Device to run the model on ('cuda' or 'cpu').
            gamma: Discount factor for return computation.
                Reference: dataset.py#L19 (gamma=1.)
            return_scale: Scale factor for returns.
                Reference: dataset.py#L19 (scale=10)
            max_length: Maximum sequence length (w value in paper).
                Reference: dataset.py#L19 (max_length=30)
            loss_fn: Loss function for training. Defaults to CrossEntropyLoss.
            grad_clip: Gradient clipping value.
                Reference: trainer.py#L41 (clip_grad_norm_ 0.25)
        """
        self.action_dim = action_dim
        self.device = device

        # Training parameters
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L19
        self.gamma = gamma
        self.return_scale = return_scale
        self.max_length = max_length
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L41
        self.grad_clip = grad_clip

        # Loss function
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L62
        self.loss_fn = loss_fn

        # Statistics for reward normalization (updated in produce_training_batch)
        self._min_reward: Optional[float] = None
        self._max_reward: Optional[float] = None

    # =========================================================================
    # Abstract Methods (Model Interface)
    # =========================================================================

    @abstractmethod
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model for training.

        This method is called by train_batch() to get action predictions.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L75-L143

        Args:
            states: Batch of states with shape (1, seq_len, S_INFO, S_LEN).
            actions: Batch of actions with shape (1, seq_len, 1).
            returns: Batch of return-to-go values with shape (1, seq_len, 1).
            timesteps: Batch of timesteps with shape (1, seq_len).

        Returns:
            Predicted action logits with shape (1, seq_len, action_dim).
        """
        pass

    @abstractmethod
    def sample(
        self,
        state: torch.Tensor,
        target_return: float,
        timestep: int,
    ) -> int:
        """Sample an action from the model for inference.

        This method is called by select_action() to get the action.
        It uses the Decision Transformer-style conditional generation
        based on state and return-to-go.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L145-L215

        Args:
            state: Current state tensor with shape (1, 1, S_INFO, S_LEN).
            target_return: Current return-to-go value.
            timestep: Current timestep within the episode.

        Returns:
            Selected bitrate index.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new episode.

        This method should clear any internal buffers (e.g., embedding deques)
        used for autoregressive inference.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L217-L224
        """
        pass

    @abstractmethod
    def get_params(self) -> Any:
        """Get the current network parameters.

        Returns:
            Network parameters in a format suitable for set_params.
        """
        pass

    @abstractmethod
    def set_params(self, params: Any) -> None:
        """Set the network parameters.

        Args:
            params: Network parameters from get_params.
        """
        pass

    # =========================================================================
    # AbstractTrainableAgent Interface Implementation
    # =========================================================================

    def select_action(self, state: NetLLMState) -> Tuple[int, List[float]]:
        """Select an action for inference using model.sample().

        This method implements the inference process from evaluate.py:
        1. Convert numpy state to torch tensor
        2. Call model.sample() with state, return-to-go, and timestep
        3. Return the selected action

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L62
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/models/rl_policy.py#L145-L215

        Args:
            state: NetLLMState object containing raw data.

        Returns:
            Tuple of (selected_action_index, action_probability_distribution).
            Action probability is one-hot for NetLLM (argmax action selection).
        """
        # Convert numpy state to torch tensor with shape (1, 1, S_INFO, S_LEN)
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L25
        state_tensor = torch.as_tensor(
            state.state_matrix,
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (S_INFO, S_LEN) -> (1, 1, S_INFO, S_LEN)

        # Call sample for inference
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/evaluate.py#L62
        action = self.sample(
            state_tensor,
            state.target_return,
            state.timestep,
        )

        # Return one-hot probability distribution
        # NetLLM uses argmax for action selection, so we return 1.0 for selected action
        action_prob = [0.0] * self.action_dim
        action_prob[action] = 1.0

        return action, action_prob

    def select_action_for_training(self, state: NetLLMState) -> Tuple[int, List[float]]:
        """Select an action for training.

        For NetLLM, training uses offline datasets with supervised learning,
        so this method is the same as select_action (no exploration noise needed).

        Args:
            state: NetLLMState object containing raw data.

        Returns:
            Tuple of (selected_action_index, action_probability_distribution).
            Action probability is one-hot for NetLLM (argmax action selection).
        """
        return self.select_action(state)

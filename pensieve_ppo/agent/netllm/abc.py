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
        min_reward: float,
        max_reward: float,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        gamma: float = 1.0,
        return_scale: float = 10.0,
        max_length: int = 30,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        grad_clip: float = 0.25,
        grad_accum_steps: int = 1,
    ):
        """Initialize the NetLLM agent.

        Args:
            action_dim: Number of discrete actions (bitrate levels).
            min_reward: Global minimum reward for normalization.
                Reference: dataset.py#L86-L93
            max_reward: Global maximum reward for normalization.
                Reference: dataset.py#L86-L93
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
            grad_accum_steps: Number of steps to accumulate gradients before updating.
                Reference: trainer.py#L20 (grad_accum_steps=1)
        """
        self.action_dim = action_dim
        self.device = device

        # Global reward normalization parameters
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L86-L93
        assert max_reward > min_reward, "max_reward must be greater than min_reward"
        self.min_reward = min_reward
        self.max_reward = max_reward

        # Training parameters
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L19
        self.gamma = gamma
        self.return_scale = return_scale
        self.max_length = max_length
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L41
        self.grad_clip = grad_clip
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L20
        self.grad_accum_steps = grad_accum_steps

        # Loss function
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L62
        self.loss_fn = loss_fn

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

    def produce_training_batch(
        self,
        trajectory: List[Step],
        done: bool,
    ) -> NetLLMTrainingBatch:
        """Process raw trajectory data into training batch.

        This method implements the data processing from dataset.py:
        1. Extract raw data from trajectory (states, actions, rewards, dones)
        2. Normalize rewards
        3. Compute discounted returns
        4. Compute timesteps for each step
        5. Convert to tensors with correct shapes

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L15-L117
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/utils/utils.py#L11-L24

        Args:
            trajectory: List of Step objects collected during environment rollout.
                Each Step.state is a NetLLMState containing raw data.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            NetLLMTrainingBatch with processed tensors.
        """
        # Step 1: Extract raw data from trajectory
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L66-L76
        states: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []

        for step in trajectory:
            netllm_state: NetLLMState = step.state  # type: ignore
            states.append(netllm_state.state_matrix)
            actions.append(netllm_state.action)
            rewards.append(netllm_state.reward)
            dones.append(netllm_state.done)

        # Step 2: Normalize rewards using global min/max
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L86-L93
        rewards = [(r - self.min_reward) / (self.max_reward - self.min_reward) for r in rewards]

        # Step 3: Compute discounted returns
        returns: List[float] = []
        timesteps: List[int] = []
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/dataset.py#L95-L107
        episode_start = 0
        while episode_start < len(rewards):
            try:  # Find episode end
                episode_end = dones.index(True, episode_start) + 1
            except ValueError:
                episode_end = len(rewards)
            returns.extend(discount_returns(rewards[episode_start:episode_end], self.gamma, self.return_scale))
            timesteps.extend(list(range(episode_end - episode_start)))
            episode_start = episode_end
        assert len(returns) == len(timesteps)

        # Step 4: Convert to tensors
        # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/utils/utils.py#L11-L24
        states_array = np.stack(states, axis=0)  # (seq_len, S_INFO, S_LEN)
        states = torch.as_tensor(states_array, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, seq_len, S_INFO, S_LEN)
        actions_array = np.array(actions, dtype=np.float32)  # (seq_len,)
        labels = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
        actions_normalized = (actions_array + 1) / self.action_dim  # (seq_len,)
        actions = torch.as_tensor(actions_normalized, dtype=torch.float32, device=self.device).reshape(1, -1, 1)  # (1, seq_len, 1)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device).reshape(1, -1, 1)  # (1, seq_len, 1)
        timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=self.device).unsqueeze(0)  # (1, seq_len)

        return NetLLMTrainingBatch(
            states=states,
            actions=actions,
            returns=returns,
            timesteps=timesteps,
            labels=labels,
        )

    def train_batch(
        self,
        training_batches: List[NetLLMTrainingBatch],
        epoch: int,
    ) -> Dict[str, float]:
        """Train on multiple training batches.

        This method implements the training step from trainer.py:
        1. For each batch, call train_step() to get loss
        2. Backpropagate and update parameters with gradient clipping

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L26-L56
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L58-L63

        Args:
            training_batches: List of NetLLMTrainingBatch from workers.
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics (logs).
        """
        train_losses = []
        logs = dict()

        dataset_size = len(training_batches)

        self.model.train()
        for step, batch in enumerate(training_batches):
            train_loss = self.train_step(batch)
            train_losses.append(train_loss.item())

            # perform gradient accumulation update
            # https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L38-L46
            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        logs['training/train_loss_mean'] = np.mean(train_losses) if train_losses else 0.0
        logs['training/train_loss_std'] = np.std(train_losses) if train_losses else 0.0

        logs['training/train_losses'] = train_losses

        return logs

    def train_step(self, batch: NetLLMTrainingBatch) -> torch.Tensor:
        """Perform a single training step.

        This method computes the loss for a single batch following trainer.py.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L58-L63

        Args:
            batch: A NetLLMTrainingBatch containing states, actions, returns, timesteps, labels.

        Returns:
            Loss tensor for this batch.
        """
        states, actions, returns, timesteps, labels = batch.states, batch.actions, batch.returns, batch.timesteps, batch.labels
        actions_pred = self.forward(states, actions, returns, timesteps)
        actions_pred = actions_pred.permute(0, 2, 1)
        loss = self.loss_fn(actions_pred, labels)
        return loss

    # =========================================================================
    # Abstract Properties (to be implemented by subclass)
    # =========================================================================

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer for training.

        Subclasses must implement this property to return their optimizer.

        Returns:
            PyTorch optimizer instance.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """Get the model module for training.

        Subclasses must implement this property to return their model.
        Used for model.train() and gradient clipping.

        Returns:
            PyTorch nn.Module instance.
        """
        pass

    @property
    @abstractmethod
    def lr_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Get the learning rate scheduler for training.

        Subclasses must implement this property to return their lr_scheduler.
        Can return None if no scheduler is used.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L21
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/trainer.py#L45-L46

        Returns:
            PyTorch lr_scheduler instance or None.
        """
        pass

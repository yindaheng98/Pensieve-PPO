"""A3C Agent implementation.

This module implements the A3C (Asynchronous Advantage Actor-Critic) agent
following the architecture from the original Pensieve implementation.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .. import AbstractRLAgent
from ..observer import RLState
from .model import Actor, Critic


# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L6
GAMMA = 0.99

# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L8-L9
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6

# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L15-L16
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L29
RAND_RANGE = 1000


class A3CAgent(AbstractRLAgent):
    """A3C (Asynchronous Advantage Actor-Critic) Agent.

    This agent implements the A3C algorithm with:
    - Separate Actor and Critic networks
    - Policy gradient with entropy regularization
    - TD (Temporal Difference) error for advantage estimation
    - RMSProp optimizer (following original implementation)

    Reference:
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L79-L200 (central_agent)
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L202-L341 (agent)
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        actor_lr: float = ACTOR_LR_RATE,
        critic_lr: float = CRITIC_LR_RATE,
        gamma: float = GAMMA,
        entropy_weight: float = ENTROPY_WEIGHT,
        entropy_eps: float = ENTROPY_EPS,
        device: Optional[torch.device] = None,
    ):
        """Initialize the A3C agent.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L90-L95

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            actor_lr: Learning rate for the actor optimizer.
            critic_lr: Learning rate for the critic optimizer.
            gamma: Discount factor for future rewards.
            entropy_weight: Weight for entropy regularization in actor loss.
            entropy_eps: Small epsilon for numerical stability in entropy calculation.
            device: PyTorch device for computations.
        """
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.device = device if device is not None else torch.device('cpu')

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L6
        self.gamma = gamma

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L8-L9
        self.entropy_weight = entropy_weight
        self.entropy_eps = entropy_eps

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L90-L95
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L57-L59
        # Using RMSprop as in original TensorFlow implementation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=critic_lr)

    def train(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        p_batch: np.ndarray,
        v_batch: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """Train the A3C agent on a batch of experiences.

        This implements the A3C training step with:
        - Policy gradient loss with entropy regularization for actor
        - Mean squared error loss for critic

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L218-L245
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L138-L170

        Args:
            s_batch: Batch of states with shape (batch_size, s_dim[0], s_dim[1]).
            a_batch: Batch of actions (one-hot) with shape (batch_size, action_dim).
            p_batch: Batch of old action probabilities (not used in A3C, kept for interface compatibility).
            v_batch: Batch of computed returns with shape (batch_size, 1).
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics.
        """
        # Convert to tensors
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(self.device)
        a_batch = torch.from_numpy(a_batch).to(torch.float32).to(self.device)
        v_batch = torch.from_numpy(v_batch).to(torch.float32).to(self.device)

        # Forward pass
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L228
        action_probs = self.actor(s_batch)  # (batch_size, action_dim)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L240
        values = self.critic(s_batch)  # (batch_size, 1)

        # Compute TD error (advantage)
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L240
        # Original: td_batch = R_batch - v_batch
        td_batch = v_batch - values.detach()  # (batch_size, 1)

        # Actor loss: policy gradient with entropy regularization
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L46-L52
        # log_prob = log(sum(action_prob * action_one_hot))
        selected_action_prob = torch.sum(action_probs * a_batch, dim=1, keepdim=True)
        log_prob = torch.log(selected_action_prob + self.entropy_eps)

        # Policy gradient: -log_prob * advantage (we minimize, so negate)
        # The original uses -act_grad_weights which is -td_batch
        # obj = sum(log(selected_prob) * (-td)) + entropy_weight * sum(prob * log(prob))
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L46-L52
        policy_loss = torch.sum(log_prob * (-td_batch))

        # Entropy: -sum(prob * log(prob))
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L51-L52
        entropy = torch.sum(action_probs * torch.log(action_probs + self.entropy_eps))
        actor_loss = policy_loss + self.entropy_weight * entropy

        # Update actor
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L57-L59
        # Original: tf.train.RMSPropOptimizer(self.lr_rate).apply_gradients(...)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic loss: Mean squared error
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L149-L150
        values = self.critic(s_batch)  # Recompute after actor update
        critic_loss = F.mse_loss(values, v_batch)

        # Update critic
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L155-L157
        # Original: tf.train.RMSPropOptimizer(self.lr_rate).apply_gradients(...)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute entropy for logging
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L263-L272
        avg_entropy = -entropy.item() / s_batch.shape[0]

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_loss": torch.mean(td_batch ** 2).item(),
            "entropy": avg_entropy,
        }

    def predict(self, state: np.ndarray) -> List[float]:
        """Predict action probabilities for a given state.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L91-L94

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).
                   The batch dimension will be added internally.

        Returns:
            Action probability distribution as a 1D list.
        """
        s_info, s_len = self.s_dim
        with torch.no_grad():
            # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L287
            state = np.reshape(state, (1, s_info, s_len))
            state = torch.from_numpy(state).to(torch.float32).to(self.device)
            # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L91-L94
            action_prob = self.actor(state)[0]
            return action_prob.cpu().tolist()

    def select_action(self, state: RLState) -> Tuple[int, List[float]]:
        """Select an action using probability-based sampling.

        This method uses cumulative distribution function sampling as in the
        original Pensieve implementation for action selection during testing.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/rl_test.py#L126-L128

        Args:
            state: RLState containing state_matrix with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        action_prob = self.predict(state.state_matrix)
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/rl_test.py#L127-L128
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L288-L289
        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states
        action_cumsum = np.cumsum(action_prob)
        action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        return int(action), action_prob

    def select_action_for_training(self, state: RLState) -> Tuple[int, List[float]]:
        """Select an action using probability-based sampling for training.

        Same as select_action for A3C (no separate exploration strategy like Gumbel noise).

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/multi_agent.py#L287-L289

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, action_probabilities).
        """
        return self.select_action(state)

    def compute_v(
        self,
        s_batch: List[RLState],
        a_batch: List[List[int]],
        r_batch: List[float],
        terminal: bool,
    ) -> List[float]:
        """Compute value targets (returns) for a trajectory.

        This implements the n-step return computation from the original A3C.
        For terminal states, bootstraps from 0. For non-terminal, bootstraps
        from the critic's value estimate.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L218-L238

        Args:
            s_batch: List of states in the trajectory.
            a_batch: List of actions in the trajectory.
            r_batch: List of rewards in the trajectory.
            terminal: Whether the trajectory ended in a terminal state.

        Returns:
            List of computed returns for each timestep.
        """
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L230
        R_batch = np.zeros((len(r_batch), 1))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L232-L235
        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            with torch.no_grad():
                # Bootstrap from last state's value
                # Extract state_matrix arrays from RLState objects
                s_tensor = torch.from_numpy(np.array([state.state_matrix for state in s_batch])).to(torch.float32).to(self.device)
                val = self.critic(s_tensor)
                R_batch[-1, 0] = val[-1, 0].item()  # boot strap from last state

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L237-L238
        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t, 0] = r_batch[t] + self.gamma * R_batch[t + 1, 0]

        return list(R_batch)

    def get_params(self) -> Tuple[Dict, Dict]:
        """Get the current network parameters.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L108-L109
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L209-L210

        Returns:
            Tuple of (actor_state_dict, critic_state_dict).
        """
        return [self.actor.state_dict(), self.critic.state_dict()]

    def set_params(self, params: Tuple[Dict, Dict]) -> None:
        """Set the network parameters.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L111-L114
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L212-L215

        Args:
            params: Tuple of (actor_state_dict, critic_state_dict).
        """
        actor_net_params, critic_net_params = params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def get_actor_gradients(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        td_batch: np.ndarray,
    ) -> List[torch.Tensor]:
        """Compute actor gradients without applying them.

        This is useful for distributed training where gradients are aggregated
        before being applied.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L96-L101

        Args:
            s_batch: Batch of states.
            a_batch: Batch of actions (one-hot).
            td_batch: Batch of TD errors (advantages).

        Returns:
            List of gradient tensors for actor parameters.
        """
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(self.device)
        a_batch = torch.from_numpy(a_batch).to(torch.float32).to(self.device)
        td_batch = torch.from_numpy(td_batch).to(torch.float32).to(self.device)

        action_probs = self.actor(s_batch)
        selected_action_prob = torch.sum(action_probs * a_batch, dim=1, keepdim=True)
        log_prob = torch.log(selected_action_prob + self.entropy_eps)

        policy_loss = torch.sum(log_prob * (-td_batch))
        entropy = torch.sum(action_probs * torch.log(action_probs + self.entropy_eps))
        actor_loss = policy_loss + self.entropy_weight * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        gradients = [p.grad.clone() for p in self.actor.parameters() if p.grad is not None]
        return gradients

    def get_critic_gradients(
        self,
        s_batch: np.ndarray,
        v_batch: np.ndarray,
    ) -> List[torch.Tensor]:
        """Compute critic gradients without applying them.

        This is useful for distributed training where gradients are aggregated
        before being applied.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L198-L202

        Args:
            s_batch: Batch of states.
            v_batch: Batch of computed returns (TD targets).

        Returns:
            List of gradient tensors for critic parameters.
        """
        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(self.device)
        v_batch = torch.from_numpy(v_batch).to(torch.float32).to(self.device)

        values = self.critic(s_batch)
        critic_loss = F.mse_loss(values, v_batch)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        gradients = [p.grad.clone() for p in self.critic.parameters() if p.grad is not None]
        return gradients

    def apply_actor_gradients(self, gradients: List[torch.Tensor]) -> None:
        """Apply pre-computed gradients to actor parameters.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L103-L106

        Args:
            gradients: List of gradient tensors matching actor parameters.
        """
        self.actor_optimizer.zero_grad()
        for param, grad in zip(self.actor.parameters(), gradients):
            param.grad = grad.to(self.device)
        self.actor_optimizer.step()

    def apply_critic_gradients(self, gradients: List[torch.Tensor]) -> None:
        """Apply pre-computed gradients to critic parameters.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L204-L207

        Args:
            gradients: List of gradient tensors matching critic parameters.
        """
        self.critic_optimizer.zero_grad()
        for param, grad in zip(self.critic.parameters(), gradients):
            param.grad = grad.to(self.device)
        self.critic_optimizer.step()

    def tensorboard_logging(self, writer: SummaryWriter, epoch: int) -> None:
        """Log A3C-specific metrics to TensorBoard.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L275-L286

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch.
        """
        writer.add_scalar('Entropy Weight', self.entropy_weight, epoch)


def compute_entropy(x: np.ndarray) -> float:
    """Compute the entropy of a probability distribution.

    Given vector x, computes the entropy H(x) = -sum(p * log(p)).

    Reference:
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L263-L272

    Args:
        x: Probability distribution array.

    Returns:
        Entropy value.
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def discount(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted cumulative sums.

    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 * x[i+2] + ...

    Reference:
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L248-L260

    Args:
        x: Input array of values.
        gamma: Discount factor.

    Returns:
        Array of discounted cumulative sums.
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out

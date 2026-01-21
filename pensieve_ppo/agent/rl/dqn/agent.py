"""DQN Agent implementation.

This module implements the DQN (Deep Q-Network) agent with Double DQN
and experience replay, following the architecture from the original
Pensieve-PPO DQN implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .. import AbstractRLAgent
from ..observer import RLState
from .model import QNetwork


# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L11
GAMMA = 0.99

# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L12
MAX_POOL_NUM = 500000

# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L13
TAU = 1e-5

# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L15
ACTOR_LR_RATE = 1e-4

# Minimum pool size before training starts
# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L136
MIN_POOL_SIZE = 4096

# Batch size for training
# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L140
BATCH_SIZE = 1024


class DQNAgent(AbstractRLAgent):
    """DQN (Deep Q-Network) Agent.

    This agent implements the Double DQN algorithm with:
    - Separate Eval and Target Q-networks
    - Experience replay buffer
    - Soft target network updates
    - Epsilon-greedy exploration (handled externally in training loop)

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L15-L157
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        learning_rate: float = ACTOR_LR_RATE,
        gamma: float = GAMMA,
        tau: float = TAU,
        max_pool_size: int = MAX_POOL_NUM,
        min_pool_size: int = MIN_POOL_SIZE,
        batch_size: int = BATCH_SIZE,
        device: Optional[torch.device] = None,
    ):
        """Initialize the DQN agent.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L70-L119

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            learning_rate: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            tau: Soft update parameter for target network.
            max_pool_size: Maximum size of replay buffer.
            min_pool_size: Minimum pool size before training starts.
            batch_size: Batch size for training.
            device: PyTorch device for computations.
        """
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.device = device if device is not None else torch.device('cpu')

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L11
        self.gamma = gamma

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L13
        self.tau = tau

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L12
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.batch_size = batch_size

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L82-L84
        # Create Eval and Target networks
        self.eval_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        # Initialize target network with same weights as eval network
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L73
        self.lr_rate = learning_rate

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L119
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L90
        # Experience replay buffer
        self.pool: List[List] = []

    def _soft_update(self) -> None:
        """Soft update target network parameters.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L96-L97
        """
        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * eval_param.data
            )

    def train(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        p_batch: np.ndarray,
        v_batch: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """Train the DQN agent on a batch of experiences.

        Note: For DQN, the interface is different from policy gradient methods.
        This method adds experiences to the replay buffer and performs training
        when enough experiences are collected.

        In DQN context:
        - p_batch is repurposed as next_state_batch
        - v_batch is repurposed as reward_batch (with done flags handled separately)

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L127-L157

        Args:
            s_batch: Batch of states.
            a_batch: Batch of actions (one-hot).
            p_batch: Batch of next states (repurposed from action probabilities).
            v_batch: Batch of rewards (repurposed from value targets).
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics.
        """
        # For DQN, we need done flags. Since the standard interface doesn't include them,
        # we'll assume non-terminal transitions. For proper DQN training, use train_dqn method.
        d_batch = np.zeros((len(s_batch), 1))
        return self.train_dqn(s_batch, a_batch, p_batch, v_batch, d_batch, epoch)

    def train_dqn(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        ns_batch: np.ndarray,
        r_batch: np.ndarray,
        d_batch: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """Train the DQN agent with full DQN interface.

        This is the proper DQN training interface that includes done flags.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L127-L157

        Args:
            s_batch: Batch of current states.
            a_batch: Batch of actions (one-hot).
            ns_batch: Batch of next states.
            r_batch: Batch of rewards.
            d_batch: Batch of done flags (1.0 if terminal, 0.0 otherwise).
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics.
        """
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L129-L134
        # Add experiences to replay buffer
        for (s, a, r, ns, d) in zip(s_batch, a_batch, r_batch, ns_batch, d_batch):
            if len(self.pool) > self.max_pool_size:
                # Random replacement when buffer is full
                pop_item = np.random.randint(len(self.pool))
                self.pool[pop_item] = [s, a, r, ns, d]
            else:
                self.pool.append([s, a, r, ns, d])

        loss_value = 0.0

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L136
        # Only train when we have enough experiences
        if len(self.pool) > self.min_pool_size:
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L137-L147
            # Sample from replay buffer
            s_samples, a_samples, r_samples, ns_samples, d_samples = [], [], [], [], []
            pop_items = np.random.randint(len(self.pool), size=self.batch_size)
            for pop_item in pop_items:
                s_, a_, r_, ns_, d_ = self.pool[pop_item]
                s_samples.append(s_)
                a_samples.append(a_)
                r_samples.append(r_)
                ns_samples.append(ns_)
                d_samples.append(d_)

            # Convert to tensors
            s_tensor = torch.from_numpy(np.array(s_samples)).to(torch.float32).to(self.device)
            a_tensor = torch.from_numpy(np.array(a_samples)).to(torch.float32).to(self.device)
            r_tensor = torch.from_numpy(np.array(r_samples)).to(torch.float32).to(self.device)
            ns_tensor = torch.from_numpy(np.array(ns_samples)).to(torch.float32).to(self.device)
            d_tensor = torch.from_numpy(np.array(d_samples)).to(torch.float32).to(self.device)

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L82-L87
            # Forward pass through networks
            eval_q = self.eval_net(s_tensor)  # Q(s, a) for current state
            eval_q_ns = self.eval_net(ns_tensor)  # Q(s', a) for next state (for Double DQN action selection)

            with torch.no_grad():
                target_q = self.target_net(ns_tensor)  # Target Q(s', a) for next state

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L86-L87
            # Double DQN: use eval network to select action, target network to evaluate
            # double_target = target_q[argmax(eval_q_ns)]
            best_actions = torch.argmax(eval_q_ns, dim=1)
            best_actions_onehot = F.one_hot(best_actions, num_classes=self.a_dim).float()
            double_target = torch.sum(target_q * best_actions_onehot, dim=1, keepdim=True)

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L89
            # Compute TD target: R = r + gamma * (1 - done) * Q_target(s', argmax_a Q_eval(s', a))
            td_target = r_tensor + self.gamma * (1 - d_tensor) * double_target

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L115-L117
            # Compute loss: MSE between Q(s, a) and TD target
            q_values = torch.sum(eval_q * a_tensor, dim=1, keepdim=True)
            loss = F.mse_loss(q_values, td_target.detach())

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L149-L155
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L156
            # Soft update target network
            self._soft_update()

            loss_value = loss.item()

        return {
            "loss": loss_value,
            "pool_size": len(self.pool),
        }

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a given state.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L121-L125

        Args:
            state: Input state with shape (s_dim[0], s_dim[1]).
                   The batch dimension will be added internally.

        Returns:
            Q-values for each action as a 1D array.
        """
        s_info, s_len = self.s_dim
        with torch.no_grad():
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L121-L125
            state = np.reshape(state, (1, s_info, s_len))
            state = torch.from_numpy(state).to(torch.float32).to(self.device)
            q_values = self.eval_net(state)[0]
            return q_values.cpu().numpy()

    def select_action(self, state: RLState) -> Tuple[int, List[float]]:
        """Select an action using greedy policy (for testing).

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/test_dqn.py#L125-L127

        Args:
            state: RLState containing state_matrix with shape (s_dim[0], s_dim[1]).

        Returns:
            Tuple of (selected_action_index, q_values_as_list).
        """
        q_values = self.predict(state.state_matrix)
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/test_dqn.py#L127
        action = int(np.argmax(q_values))
        return action, q_values.tolist()

    def select_action_for_training(
        self,
        state: RLState,
        epsilon: float = 0.0,
    ) -> Tuple[int, List[float]]:
        """Select an action using epsilon-greedy policy (for training).

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L164-L173

        Args:
            state: RLState containing state_matrix with shape (s_dim[0], s_dim[1]).
            epsilon: Probability of selecting random action.

        Returns:
            Tuple of (selected_action_index, q_values_as_list).
        """
        q_values = self.predict(state.state_matrix)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L170-L173
        if np.random.uniform() < epsilon:
            action = np.random.randint(self.a_dim)
        else:
            action = int(np.argmax(q_values))

        return action, q_values.tolist()

    def compute_v(
        self,
        s_batch: List[RLState],
        a_batch: List[List[int]],
        r_batch: List[float],
        terminal: bool,
    ) -> List[float]:
        """Compute value targets for a trajectory.

        Note: This method is provided for interface compatibility with AbstractRLAgent.
        For DQN, value computation is done differently (via TD targets with target network).
        This returns the rewards as-is since DQN uses experience replay.

        Args:
            s_batch: List of states in the trajectory.
            a_batch: List of actions in the trajectory.
            r_batch: List of rewards in the trajectory.
            terminal: Whether the trajectory ended in a terminal state.

        Returns:
            List of rewards (DQN handles value computation internally).
        """
        # For DQN, we return rewards as-is. The actual TD computation
        # happens in the train method using the replay buffer.
        return [[r] for r in r_batch]

    def get_params(self) -> Tuple[Dict, Dict]:
        """Get the current network parameters.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L54-L56
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L99-L103

        Returns:
            Tuple of (eval_net_state_dict, target_net_state_dict).
        """
        return [self.eval_net.state_dict(), self.target_net.state_dict()]

    def set_params(self, params: Tuple[Dict, Dict]) -> None:
        """Set the network parameters.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L65-L68

        Args:
            params: Tuple of (eval_net_state_dict, target_net_state_dict).
        """
        eval_net_params, target_net_params = params
        self.eval_net.load_state_dict(eval_net_params)
        self.target_net.load_state_dict(target_net_params)

    def tensorboard_logging(self, writer: SummaryWriter, epoch: int) -> None:
        """Log DQN-specific metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch.
        """
        writer.add_scalar('Pool Size', len(self.pool), epoch)

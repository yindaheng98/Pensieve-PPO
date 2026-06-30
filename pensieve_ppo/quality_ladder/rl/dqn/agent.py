"""DQN Agent implementation.

This module implements the DQN (Deep Q-Network) agent with Double DQN
and experience replay, following the architecture from the original
Pensieve-PPO DQN implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ....agent import Step, TrainBatchInfo
from .. import AbstractRLAgent, RLTrainingBatch
from ..abc import RLActionDecision
from ..observer import RLState
from .model import QNetwork


@dataclass
class DQNTrainingBatch(RLTrainingBatch):
    """Training batch for DQN extending RLTrainingBatch with next states and done flags.

    Inherits s_batch, a_batch, p_batch, v_batch from RLTrainingBatch.
    For DQN, v_batch contains rewards (from compute_v which returns rewards as-is).
    Adds ns_batch (next states) and d_batch (done flags) for TD target computation.

    Attributes:
        ns_batch: List of next states.
        d_batch: List of done flags (1.0 if terminal, 0.0 otherwise).
    """
    ns_batch: List[RLState]
    d_batch: List[float]


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
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L71
        self.s_dim = state_dim
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L72
        self.a_dim = action_dim
        self.device = device if device is not None else torch.device('cpu')

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L11
        self.gamma = gamma

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L13
        self.tau = tau

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L12
        self.max_pool_size = max_pool_size
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L136
        self.min_pool_size = min_pool_size
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L140
        self.batch_size = batch_size

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L82
        self.eval_net = QNetwork(state_dim, action_dim).to(self.device)
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L84
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

    def train(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        p_batch: np.ndarray,
        v_batch: np.ndarray,
        epoch: int,
    ) -> TrainBatchInfo:
        """Not used for DQN. Use train_batch / train_dqn instead.

        DQN requires (s, a, r, next_s, done) transitions which don't fit the
        standard RL train interface. The overridden produce_training_batch and
        train_batch methods handle proper DQN training.

        Raises:
            NotImplementedError: Always. DQN uses train_batch -> train_dqn path.
        """
        raise NotImplementedError(
            "DQN does not use the standard train() interface. "
            "Use train_batch() which calls train_dqn() directly."
        )

    def train_dqn(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        ns_batch: np.ndarray,
        r_batch: np.ndarray,
        d_batch: np.ndarray,
        epoch: int,
    ) -> TrainBatchInfo:
        """Train the DQN agent with full DQN interface.

        This is the proper DQN training interface that includes done flags.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L127-L156

        Args:
            s_batch: Batch of current states.
            a_batch: Batch of actions (one-hot).
            ns_batch: Batch of next states.
            r_batch: Batch of rewards.
            d_batch: Batch of done flags (1.0 if terminal, 0.0 otherwise).
            epoch: Current training epoch.

        Returns:
            TrainBatchInfo containing training metrics.
        """
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L128-L134
        # Original: for (s, a, v, ns, d) in zip(s_batch, a_batch, r_batch, p_batch, d_batch):
        # Note: in original, p_batch = next states, r_batch = rewards (named 'v' in loop)
        for (s, a, r, ns, d) in zip(s_batch, a_batch, r_batch, ns_batch, d_batch):
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L130-L134
            if len(self.pool) > self.max_pool_size:
                pop_item = np.random.randint(len(self.pool))
                self.pool[pop_item] = [s, a, r, ns, d]
            else:
                self.pool.append([s, a, r, ns, d])

        loss_value = 0.0

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L136
        if len(self.pool) > self.min_pool_size:
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L137
            s_samples, a_samples, r_samples, ns_samples, d_samples = [], [], [], [], []

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L140
            pop_items = np.random.randint(len(self.pool), size=self.batch_size)
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L141-L147
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

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L82
            eval_q = self.eval_net(s_tensor)       # self.eval = self.CreateEval(inputs=self.inputs)
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L83
            eval_q_ns = self.eval_net(ns_tensor)   # self.eval_ns = self.CreateEval(inputs=self.ns_inputs)

            with torch.no_grad():
                # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L84
                target_q = self.target_net(ns_tensor)  # self.target = self.CreateTarget(inputs=self.ns_inputs)

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L86-L87
            # self.double_target = tf.reduce_sum(tf.multiply(self.target,
            #     tf.one_hot(tf.argmax(self.eval_ns, axis=-1), self.a_dim)), reduction_indices=1, keepdims=True)
            best_actions = torch.argmax(eval_q_ns, dim=1)
            best_actions_onehot = F.one_hot(best_actions, num_classes=self.a_dim).float()
            double_target = torch.sum(target_q * best_actions_onehot, dim=1, keepdim=True)

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L89
            # self.R = tf.stop_gradient(self.r + GAMMA * (1 - self.done) * self.double_target)
            td_target = r_tensor + self.gamma * (1 - d_tensor) * double_target

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L115-L117
            # self.loss = tflearn.mean_square(
            #     tf.reduce_sum(tf.multiply(self.eval, self.acts), reduction_indices=1, keepdims=True),
            #     self.R)
            q_values = torch.sum(eval_q * a_tensor, dim=1, keepdim=True)
            loss = F.mse_loss(q_values, td_target.detach())

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L149-L155
            # self.sess.run(self.val_opt, ...)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L156
            # self.sess.run(self.soft_update)
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L92-L97
            # self.eval_params = \
            #     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval')
            # self.target_params = \
            #     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
            # self.soft_update = [tf.assign(ta, (1 - TAU) * ta + TAU * ea)
            #         for ta, ea in zip(self.target_params, self.eval_params)]
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(
                    (1 - self.tau) * target_param.data + self.tau * eval_param.data
                )

            loss_value = loss.item()

        return TrainBatchInfo(
            loss=loss_value,
            extra={
                "pool_size": len(self.pool),
            },
        )

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
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L122-L124
            # action = self.sess.run(self.eval, feed_dict={
            #     self.inputs: input
            # })
            state = np.reshape(state, (1, s_info, s_len))
            state = torch.from_numpy(state).to(torch.float32).to(self.device)
            q_values = self.eval_net(state)[0]
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L125
            # return action[0]
            return q_values.cpu().numpy()

    def select_action(self, state: RLState) -> RLActionDecision:
        """Select an action using greedy policy (for testing).

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/test_dqn.py#L125-L127

        Args:
            state: RLState containing state_matrix with shape (s_dim[0], s_dim[1]).

        Returns:
            Selected quality ladder action with q-values.
        """
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/test_dqn.py#L125
        # action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        q_values = self.predict(state.state_matrix)
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/test_dqn.py#L127
        # bit_rate = np.argmax(action_prob)
        action = int(np.argmax(q_values))
        return RLActionDecision.from_index(action, q_values.tolist())

    def select_action_for_training(
        self,
        state: RLState,
        epsilon: float = 0.0,
    ) -> RLActionDecision:
        """Select an action using epsilon-greedy policy (for training).

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L164-L173

        Args:
            state: RLState containing state_matrix with shape (s_dim[0], s_dim[1]).
            epsilon: Probability of selecting random action.

        Returns:
            Selected quality ladder action with q-values.
        """
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L164-L165
        # action_prob = actor.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
        q_values = self.predict(state.state_matrix)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L170-L173
        # if np.random.uniform() < prob_:
        #     bit_rate = np.random.randint(A_DIM)
        # else:
        #     bit_rate = np.argmax(action_prob)
        if np.random.uniform() < epsilon:
            action = np.random.randint(self.a_dim)
        else:
            action = int(np.argmax(q_values))

        return RLActionDecision.from_index(action, q_values.tolist())

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

    def produce_training_batch(
        self,
        trajectory: List[Step],
        done: bool,
    ) -> DQNTrainingBatch:
        """Produce a DQN training batch from a trajectory.

        Calls super() to build the base RLTrainingBatch (s, a, p, v), then
        adds next states and done flags. For each step i, the next state is
        trajectory[i+1].state. If the episode is not done, the last entry is
        dropped since there's no next state available for it.

        In the original code, the worker collects (s, a, ns, r, d) directly:

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L160-L189

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            DQNTrainingBatch with base fields plus ns_batch and d_batch.
        """
        base_batch = super().produce_training_batch(trajectory, done)

        ns_batch: List[RLState] = []
        d_batch: List[float] = []

        # Build next states and done flags for all steps except the last one
        for i in range(len(trajectory) - 1):
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L181
            # next_s_batch.append(obs)  # obs is the state AFTER env.step
            ns_batch.append(trajectory[i + 1].state)
            # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L182
            # d_batch.append([float(done)])
            d_batch.append(1.0 if trajectory[i].done else 0.0)

        if done and len(trajectory) > 0:
            # Last step is terminal: use dummy next state, masked by done=1
            ns_batch.append(trajectory[-1].state)
            d_batch.append(1.0)
        elif len(trajectory) > 0:
            # Not done: drop the last entry from base batch since we have no next state
            base_batch.s_batch = base_batch.s_batch[:-1]
            base_batch.a_batch = base_batch.a_batch[:-1]
            base_batch.p_batch = base_batch.p_batch[:-1]
            base_batch.v_batch = base_batch.v_batch[:-1]

        return DQNTrainingBatch(
            s_batch=base_batch.s_batch,
            a_batch=base_batch.a_batch,
            p_batch=base_batch.p_batch,
            v_batch=base_batch.v_batch,
            ns_batch=ns_batch,
            d_batch=d_batch,
        )

    def train_batch(
        self,
        training_batches: List[DQNTrainingBatch],
        epoch: int,
    ) -> TrainBatchInfo:
        """Train on multiple DQN training batches.

        Concatenates data from all training batches and performs a DQN training
        step. Uses v_batch as rewards (compute_v returns rewards as-is for DQN).

        In the original code, central_agent calls actor.train() per worker:

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/train_dqn.py#L112-L114

        Args:
            training_batches: List of DQN training batches from workers.
            epoch: Current training epoch.

        Returns:
            TrainBatchInfo containing training metrics.
        """
        s: List[RLState] = []
        a: List[List[int]] = []
        v: List[List[float]] = []  # rewards wrapped as [[r], ...] from compute_v
        ns: List[RLState] = []
        d: List[float] = []

        for batch in training_batches:
            s += batch.s_batch
            a += batch.a_batch
            v += batch.v_batch
            ns += batch.ns_batch
            d += batch.d_batch

        if len(s) == 0:
            return TrainBatchInfo(loss=0.0, extra={"pool_size": len(self.pool)})

        s_batch = np.stack([state.state_matrix for state in s], axis=0)
        a_batch = np.vstack(a)
        r_batch = np.vstack(v)  # v_batch contains [[r], ...] from compute_v
        ns_batch = np.stack([state.state_matrix for state in ns], axis=0)
        d_batch = np.array(d).reshape(-1, 1)

        return self.train_dqn(s_batch, a_batch, ns_batch, r_batch, d_batch, epoch)

    def get_params(self) -> Tuple[Dict, Dict]:
        """Get the current network parameters.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L54-L55
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L100-L103

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

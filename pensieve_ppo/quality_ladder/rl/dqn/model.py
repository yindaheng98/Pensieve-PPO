"""PyTorch neural network models for DQN algorithm.

This module implements the Q-network (both Eval and Target) for the DQN algorithm
following the architecture from the original Pensieve-PPO DQN implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L9
FEATURE_NUM = 128


class QNetwork(nn.Module):
    """Q-Network for value estimation.

    This network takes a state as input and outputs Q-values for each action.
    The architecture processes different parts of the state separately
    using fully connected layers and 1D convolutions, then merges them.

    Note: The original TensorFlow implementation uses tflearn.conv_1d with
    kernel_size=1. In TF, conv_1d interprets input as (batch, steps, channels),
    so (batch, 1, s_len) means 1 step and s_len channels. With kernel_size=1,
    this is equivalent to a Linear layer from s_len to feature_num.
    We use Linear layers here to match this behavior exactly.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L16-L52
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        feature_num: int = FEATURE_NUM,
    ):
        """Initialize the Q-Network.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L16-L52

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            feature_num: Number of features in hidden layers. Defaults to 128.
        """
        super(QNetwork, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.feature_num = feature_num

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L18
        # split_0: last quality (scalar -> feature_num)
        # inputs[:, 0:1, -1] -> (batch, 1)
        self.fc0 = nn.Linear(1, feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L19
        # split_1: buffer size (scalar -> feature_num)
        # inputs[:, 1:2, -1] -> (batch, 1)
        self.fc1 = nn.Linear(1, feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L20
        # split_2: throughput history
        # In TF: tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, 1)
        # Input (batch, 1, s_len) in TF is (batch, 1 step, s_len channels)
        # conv_1d with kernel=1 outputs (batch, 1, FEATURE_NUM), flattened to (batch, FEATURE_NUM)
        # This is equivalent to Linear(s_len, feature_num)
        self.fc2 = nn.Linear(state_dim[1], feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L21
        # split_3: download time history
        # Same structure as split_2
        self.fc3 = nn.Linear(state_dim[1], feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L22
        # split_4: next chunk sizes
        # Input: inputs[:, 4:5, :a_dim] -> (batch, 1, a_dim)
        # Equivalent to Linear(a_dim, feature_num)
        self.fc4 = nn.Linear(action_dim, feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L23
        # split_5: chunks remaining (scalar -> feature_num)
        # inputs[:, 5:6, -1] -> (batch, 1)
        self.fc5 = nn.Linear(1, feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L29
        # Merge layer input size:
        # All 6 splits output feature_num each, so total is 6 * feature_num
        merge_size = feature_num * 6

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L30
        self.fc_merge = nn.Linear(merge_size, feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L31
        # Output: Q-values for each action
        self.fc_value = nn.Linear(feature_num, action_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Q-network.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L16-L33

        Args:
            inputs: Input tensor with shape (batch_size, s_dim[0], s_dim[1]).

        Returns:
            Q-values with shape (batch_size, action_dim).
        """
        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L18
        # split_0: inputs[:, 0:1, -1] -> last quality (batch, 1)
        split_0 = F.relu(self.fc0(inputs[:, 0:1, -1]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L19
        # split_1: inputs[:, 1:2, -1] -> buffer size (batch, 1)
        split_1 = F.relu(self.fc1(inputs[:, 1:2, -1]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L20
        # split_2: inputs[:, 2, :] -> throughput history (batch, s_len)
        split_2 = F.relu(self.fc2(inputs[:, 2, :]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L21
        # split_3: inputs[:, 3, :] -> download time history (batch, s_len)
        split_3 = F.relu(self.fc3(inputs[:, 3, :]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L22
        # split_4: inputs[:, 4, :a_dim] -> next chunk sizes (batch, a_dim)
        split_4 = F.relu(self.fc4(inputs[:, 4, :self.a_dim]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L23
        # split_5: inputs[:, 5:6, -1] -> chunks remaining (batch, 1)
        split_5 = F.relu(self.fc5(inputs[:, 5:6, -1]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L29
        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], dim=1)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L30-L31
        net = F.relu(self.fc_merge(merge_net))
        value = self.fc_value(net)

        return value

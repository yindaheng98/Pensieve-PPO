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
    using fully connected layers and then merges them.

    Note: Unlike the A3C implementation which uses 1D convolutions, the original
    DQN implementation uses fully connected layers for all state components.
    This follows the exact architecture from dqn.py.

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

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L18-L23
        # split_0: last quality (scalar -> feature_num)
        # inputs[:, 0:1, -1] -> (batch, 1)
        self.fc1 = nn.Linear(1, feature_num)

        # split_1: buffer size (scalar -> feature_num)
        # inputs[:, 1:2, -1] -> (batch, 1)
        self.fc2 = nn.Linear(1, feature_num)

        # split_2: throughput history (1D conv with kernel=1)
        # inputs[:, 2:3, :] -> (batch, 1, s_len)
        # tflearn.conv_1d with kernel=1 is equivalent to Linear on last dim
        self.conv1 = nn.Conv1d(1, feature_num, kernel_size=1)

        # split_3: download time history (1D conv with kernel=1)
        # inputs[:, 3:4, :] -> (batch, 1, s_len)
        self.conv2 = nn.Conv1d(1, feature_num, kernel_size=1)

        # split_4: next chunk sizes (1D conv with kernel=1)
        # inputs[:, 4:5, :a_dim] -> (batch, 1, a_dim)
        self.conv3 = nn.Conv1d(1, feature_num, kernel_size=1)

        # split_5: chunks remaining (scalar -> feature_num)
        # inputs[:, 5:6, -1] -> (batch, 1)
        self.fc3 = nn.Linear(1, feature_num)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L29
        # Merge layer input size:
        # split_0 (feature_num) + split_1 (feature_num) +
        # split_2 (feature_num * s_len) + split_3 (feature_num * s_len) +
        # split_4 (feature_num * a_dim) + split_5 (feature_num)
        merge_size = (feature_num +  # split_0
                      feature_num +  # split_1
                      feature_num * self.s_dim[1] +  # split_2 flattened
                      feature_num * self.s_dim[1] +  # split_3 flattened
                      feature_num * action_dim +  # split_4 flattened
                      feature_num)  # split_5

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
        split_0 = F.relu(self.fc1(inputs[:, 0:1, -1]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L19
        # split_1: inputs[:, 1:2, -1] -> buffer size (batch, 1)
        split_1 = F.relu(self.fc2(inputs[:, 1:2, -1]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L20
        # split_2: inputs[:, 2:3, :] -> throughput history
        # Conv1d expects (batch, channels, length)
        split_2 = F.relu(self.conv1(inputs[:, 2:3, :]))
        split_2_flat = split_2.view(split_2.size(0), -1)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L21
        # split_3: inputs[:, 3:4, :] -> download time history
        split_3 = F.relu(self.conv2(inputs[:, 3:4, :]))
        split_3_flat = split_3.view(split_3.size(0), -1)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L22
        # split_4: inputs[:, 4:5, :self.a_dim] -> next chunk sizes
        split_4 = F.relu(self.conv3(inputs[:, 4:5, :self.a_dim]))
        split_4_flat = split_4.view(split_4.size(0), -1)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L23
        # split_5: inputs[:, 5:6, -1] -> chunks remaining
        split_5 = F.relu(self.fc3(inputs[:, 5:6, -1]))

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L29
        merge_net = torch.cat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], dim=1)

        # https://github.com/godka/Pensieve-PPO/blob/ed429e475a179bc346c76f66dc0cf6d3f2f0914d/src/dqn.py#L30-L31
        net = F.relu(self.fc_merge(merge_net))
        value = self.fc_value(net)

        return value

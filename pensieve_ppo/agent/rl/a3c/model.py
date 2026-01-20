"""PyTorch neural network models for A3C algorithm.

This module implements the Actor and Critic networks for the A3C algorithm
following the architecture from the original Pensieve implementation.

Reference:
    https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L7
A_DIM = 6

# Feature number for hidden layers (from original TF implementation)
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L65-L70
FEATURE_NUM = 128

# Convolution kernel size (from original TF implementation)
# https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L67-L69
KERNEL_SIZE = 4


class Actor(nn.Module):
    """Actor network for policy estimation.

    Input to the network is the state, output is the distribution
    of all actions.

    This network takes a state as input and outputs action probabilities.
    The architecture processes different parts of the state separately
    using fully connected layers and 1D convolutions, then merges them.

    Reference:
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L13-L82
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L61-L81
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        feature_num: int = FEATURE_NUM,
        kernel_size: int = KERNEL_SIZE,
    ):
        """Initialize the Actor network.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L18-L59

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            feature_num: Number of features in hidden layers. Defaults to 128.
            kernel_size: Kernel size for 1D convolutions. Defaults to 4.
        """
        super(Actor, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.feature_num = feature_num
        self.kernel_size = kernel_size

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L65-L66
        # split_0: last quality (scalar -> feature_num)
        self.fc1 = nn.Linear(1, feature_num)
        # split_1: buffer size (scalar -> feature_num)
        self.fc2 = nn.Linear(1, feature_num)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L67-L69
        # split_2: throughput history (1D conv)
        # Input: (batch, 1, s_len) -> Output: (batch, feature_num, conv_out_len)
        self.conv1 = nn.Conv1d(1, feature_num, kernel_size)
        # split_3: download time history (1D conv)
        self.conv2 = nn.Conv1d(1, feature_num, kernel_size)
        # split_4: next chunk sizes (1D conv)
        self.conv3 = nn.Conv1d(1, feature_num, kernel_size)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L70
        # split_5: chunks remaining (scalar -> feature_num)
        self.fc3 = nn.Linear(1, feature_num)

        # Calculate convolution output sizes
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L72-L74
        conv_out_len = self.s_dim[1] - kernel_size + 1  # For s_len history
        conv_out_len_action = action_dim - kernel_size + 1  # For action dim input

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L76
        # Merge layer input size: 2 fc outputs + 3 flattened conv outputs + 1 fc output
        merge_size = (feature_num +  # split_0 (fc)
                      feature_num +  # split_1 (fc)
                      feature_num * conv_out_len +  # split_2 (conv flattened)
                      feature_num * conv_out_len +  # split_3 (conv flattened)
                      feature_num * conv_out_len_action +  # split_4 (conv flattened)
                      feature_num)  # split_5 (fc)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L78
        self.fc4 = nn.Linear(merge_size, feature_num)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L79
        self.out = nn.Linear(feature_num, action_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L61-L81

        Args:
            inputs: Input tensor with shape (batch_size, s_dim[0], s_dim[1]).

        Returns:
            Action probabilities with shape (batch_size, action_dim).
        """
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L65
        # split_0: inputs[:, 0:1, -1] -> last quality (batch, 1)
        split_0 = F.relu(self.fc1(inputs[:, 0:1, -1]))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L66
        # split_1: inputs[:, 1:2, -1] -> buffer size (batch, 1)
        split_1 = F.relu(self.fc2(inputs[:, 1:2, -1]))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L67
        # split_2: inputs[:, 2:3, :] -> throughput history
        # Reshape from (batch, 1, s_len) for Conv1d
        split_2 = F.relu(self.conv1(inputs[:, 2:3, :]))
        split_2_flat = split_2.view(split_2.size(0), -1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L68
        # split_3: inputs[:, 3:4, :] -> download time history
        split_3 = F.relu(self.conv2(inputs[:, 3:4, :]))
        split_3_flat = split_3.view(split_3.size(0), -1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L69
        # split_4: inputs[:, 4:5, :A_DIM] -> next chunk sizes
        split_4 = F.relu(self.conv3(inputs[:, 4:5, :self.a_dim]))
        split_4_flat = split_4.view(split_4.size(0), -1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L70
        # split_5: inputs[:, 5:6, -1] -> chunks remaining
        # Note: In original TF code this was inputs[:, 4:5, -1], but that seems like an error
        # since channel 4 is for next chunk sizes. Following the state definition, channel 5
        # should be chunks remaining. Using 5:6 for consistency with state definition.
        split_5 = F.relu(self.fc3(inputs[:, 5:6, -1]))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L76
        merge_net = torch.cat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], dim=1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L78-L79
        dense_net = F.relu(self.fc4(merge_net))
        out = F.softmax(self.out(dense_net), dim=-1)

        return out


class Critic(nn.Module):
    """Critic network for value estimation.

    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.

    This network takes a state as input and outputs a value estimate.
    The architecture is similar to the Actor but outputs a single value.

    Reference:
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L117-L180
        https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L159-L179
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        feature_num: int = FEATURE_NUM,
        kernel_size: int = KERNEL_SIZE,
    ):
        """Initialize the Critic network.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L122-L157

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            feature_num: Number of features in hidden layers. Defaults to 128.
            kernel_size: Kernel size for 1D convolutions. Defaults to 4.
        """
        super(Critic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.feature_num = feature_num
        self.kernel_size = kernel_size

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L163-L164
        # split_0: last quality (scalar -> feature_num)
        self.fc1 = nn.Linear(1, feature_num)
        # split_1: buffer size (scalar -> feature_num)
        self.fc2 = nn.Linear(1, feature_num)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L165-L167
        # split_2: throughput history (1D conv)
        self.conv1 = nn.Conv1d(1, feature_num, kernel_size)
        # split_3: download time history (1D conv)
        self.conv2 = nn.Conv1d(1, feature_num, kernel_size)
        # split_4: next chunk sizes (1D conv)
        self.conv3 = nn.Conv1d(1, feature_num, kernel_size)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L168
        # split_5: chunks remaining (scalar -> feature_num)
        self.fc3 = nn.Linear(1, feature_num)

        # Calculate convolution output sizes
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L170-L172
        conv_out_len = self.s_dim[1] - kernel_size + 1  # For s_len history
        conv_out_len_action = action_dim - kernel_size + 1  # For action dim input

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L174
        # Merge layer input size
        merge_size = (feature_num +  # split_0 (fc)
                      feature_num +  # split_1 (fc)
                      feature_num * conv_out_len +  # split_2 (conv flattened)
                      feature_num * conv_out_len +  # split_3 (conv flattened)
                      feature_num * conv_out_len_action +  # split_4 (conv flattened)
                      feature_num)  # split_5 (fc)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L176
        self.fc4 = nn.Linear(merge_size, feature_num)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L177
        # Output: single value estimate
        self.out = nn.Linear(feature_num, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network.

        Reference:
            https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L159-L179

        Args:
            inputs: Input tensor with shape (batch_size, s_dim[0], s_dim[1]).

        Returns:
            Value estimates with shape (batch_size, 1).
        """
        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L163
        # split_0: inputs[:, 0:1, -1] -> last quality (batch, 1)
        split_0 = F.relu(self.fc1(inputs[:, 0:1, -1]))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L164
        # split_1: inputs[:, 1:2, -1] -> buffer size (batch, 1)
        split_1 = F.relu(self.fc2(inputs[:, 1:2, -1]))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L165
        # split_2: inputs[:, 2:3, :] -> throughput history
        split_2 = F.relu(self.conv1(inputs[:, 2:3, :]))
        split_2_flat = split_2.view(split_2.size(0), -1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L166
        # split_3: inputs[:, 3:4, :] -> download time history
        split_3 = F.relu(self.conv2(inputs[:, 3:4, :]))
        split_3_flat = split_3.view(split_3.size(0), -1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L167
        # split_4: inputs[:, 4:5, :A_DIM] -> next chunk sizes
        split_4 = F.relu(self.conv3(inputs[:, 4:5, :self.a_dim]))
        split_4_flat = split_4.view(split_4.size(0), -1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L168
        # split_5: inputs[:, 5:6, -1] -> chunks remaining
        split_5 = F.relu(self.fc3(inputs[:, 5:6, -1]))

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L174
        merge_net = torch.cat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], dim=1)

        # https://github.com/hongzimao/pensieve/blob/1120bb173958dc9bc9f2ebff1a8fe688b6f4e93c/sim/a3c.py#L176-L177
        dense_net = F.relu(self.fc4(merge_net))
        out = self.out(dense_net)

        return out

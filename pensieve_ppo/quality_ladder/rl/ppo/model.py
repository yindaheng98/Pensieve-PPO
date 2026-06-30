"""PyTorch neural network models for PPO algorithm.

This module implements the Actor and Critic networks for the PPO algorithm
following the architecture from the original Pensieve-PPO implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L7-L8
FEATURE_NUM = 128
ACTION_EPS = 1e-4


class Actor(nn.Module):
    """Actor network for policy estimation.

    This network takes a state as input and outputs action probabilities.
    The architecture processes different parts of the state separately
    and then merges them.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L12-L41
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        feature_num: int = FEATURE_NUM,
        action_eps: float = ACTION_EPS,
    ):
        """Initialize the Actor network.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L14-L26

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            feature_num: Number of features in hidden layers. Defaults to 128.
            action_eps: Small epsilon for action probability clipping. Defaults to 1e-4.
        """
        super(Actor, self).__init__()
        # Actor network
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, feature_num)
        self.fc2_actor = nn.Linear(1, feature_num)
        self.conv1_actor = nn.Linear(self.s_dim[1], feature_num)
        self.conv2_actor = nn.Linear(self.s_dim[1], feature_num)
        self.conv3_actor = nn.Linear(self.a_dim, feature_num)
        self.fc3_actor = nn.Linear(1, feature_num)
        self.fc4_actor = nn.Linear(feature_num * self.s_dim[0], feature_num)
        self.pi_head = nn.Linear(feature_num, action_dim)

        self.feature_num = feature_num
        self.action_eps = action_eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L28-L41

        Args:
            inputs: Input tensor with shape (batch_size, s_dim[0], s_dim[1]).

        Returns:
            Action probabilities with shape (batch_size, action_dim).
        """
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, self.feature_num))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, self.feature_num))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, self.feature_num))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        pi_net = F.relu(self.fc4_actor(merge_net))
        pi = F.softmax(self.pi_head(pi_net), dim=-1)
        pi = torch.clamp(pi, self.action_eps, 1. - self.action_eps)
        return pi


class Critic(nn.Module):
    """Critic network for value estimation.

    This network takes a state as input and outputs a value estimate.
    The architecture is similar to the Actor but outputs a single value.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L44-L72
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        feature_num: int = FEATURE_NUM,
    ):
        """Initialize the Critic network.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L46-L58

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            feature_num: Number of features in hidden layers. Defaults to 128.
        """
        super(Critic, self).__init__()
        # Critic network
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.feature_num = feature_num

        self.fc1_actor = nn.Linear(1, feature_num)
        self.fc2_actor = nn.Linear(1, feature_num)
        self.conv1_actor = nn.Linear(self.s_dim[1], feature_num)
        self.conv2_actor = nn.Linear(self.s_dim[1], feature_num)
        self.conv3_actor = nn.Linear(self.a_dim, feature_num)
        self.fc3_actor = nn.Linear(1, feature_num)
        self.fc4_actor = nn.Linear(feature_num * self.s_dim[0], feature_num)
        self.val_head = nn.Linear(feature_num, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L60-L72

        Args:
            inputs: Input tensor with shape (batch_size, s_dim[0], s_dim[1]).

        Returns:
            Value estimates with shape (batch_size, 1).
        """
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, self.feature_num))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, self.feature_num))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, self.feature_num))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        value_net = F.relu(self.fc4_actor(merge_net))
        value = self.val_head(value_net)
        return value

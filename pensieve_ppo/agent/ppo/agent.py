"""PPO Agent implementation.

This module implements the PPO (Proximal Policy Optimization) agent
following the architecture from the original Pensieve-PPO implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..abc import AbstractAgent
from .model import Actor, Critic


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L9-L10
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon


class PPOAgent(AbstractAgent):
    """PPO (Proximal Policy Optimization) Agent.

    This agent implements the PPO algorithm with:
    - Separate Actor and Critic networks
    - Clipped surrogate objective
    - Dual-PPO loss for improved stability
    - Adaptive entropy weight

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L74-L160
    """

    def __init__(
        self,
        state_dim: tuple[int, int],
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = GAMMA,
        eps: float = EPS,
        ppo_training_epo: int = 5,
        h_target: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """Initialize the PPO agent.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L79-L87

        Args:
            state_dim: State dimension as [num_features, sequence_length].
            action_dim: Number of discrete actions.
            learning_rate: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            eps: PPO clipping parameter.
            ppo_training_epo: Number of PPO update epochs per batch.
            h_target: Target entropy for adaptive entropy weight.
            device: PyTorch device for computations.
        """
        super().__init__(state_dim, action_dim, device)

        self._entropy_weight = np.log(action_dim)
        self.H_target = h_target
        self.PPO_TRAINING_EPO = ppo_training_epo

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=learning_rate)

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L9-L10
        self.gamma = gamma
        self.eps = eps

    def _r(
        self,
        pi_new: torch.Tensor,
        pi_old: torch.Tensor,
        acts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance sampling ratio.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L97-L99

        Args:
            pi_new: New policy probabilities.
            pi_old: Old policy probabilities.
            acts: One-hot encoded actions.

        Returns:
            Importance sampling ratio.
        """
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
            torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        p_batch: np.ndarray,
        v_batch: np.ndarray,
        epoch: int,
    ) -> Dict[str, float]:
        """Train the PPO agent on a batch of experiences.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L101-L129

        Args:
            s_batch: Batch of states.
            a_batch: Batch of actions (one-hot).
            p_batch: Batch of old action probabilities.
            v_batch: Batch of computed returns.
            epoch: Current training epoch.

        Returns:
            Dictionary containing training metrics.
        """
        total_loss = 0.0

        s_batch = torch.from_numpy(s_batch).to(torch.float32).to(self.device)
        a_batch = torch.from_numpy(a_batch).to(torch.float32).to(self.device)
        p_batch = torch.from_numpy(p_batch).to(torch.float32).to(self.device)
        v_batch = torch.from_numpy(v_batch).to(torch.float32).to(self.device)

        for _ in range(self.PPO_TRAINING_EPO):
            pi = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss
            adv = v_batch - val.detach()
            ratio = self._r(pi, p_batch, a_batch)
            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            loss_entropy = torch.sum(-pi * torch.log(pi), dim=1, keepdim=True)

            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)

        return {
            "loss": total_loss / self.PPO_TRAINING_EPO,
            "entropy_weight": self._entropy_weight,
            "entropy": _H,
        }

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action probabilities for a given state.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L131-L135

        Args:
            state: Input state with shape (1, s_dim[0], s_dim[1]).

        Returns:
            Action probability distribution.
        """
        with torch.no_grad():
            state = torch.from_numpy(state).to(torch.float32).to(self.device)
            pi = self.actor.forward(state)[0]
            return pi.cpu().numpy()

    def compute_v(
        self,
        s_batch: List[np.ndarray],
        a_batch: List[np.ndarray],
        r_batch: List[float],
        terminal: bool,
    ) -> List[float]:
        """Compute value targets (returns) for a trajectory.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L146-L159

        Args:
            s_batch: List of states in the trajectory.
            a_batch: List of actions in the trajectory.
            r_batch: List of rewards in the trajectory.
            terminal: Whether the trajectory ended in a terminal state.

        Returns:
            List of computed returns for each timestep.
        """
        R_batch = np.zeros_like(r_batch)

        if terminal:
            # in this case, the terminal reward will be assigned as r_batch[-1]
            R_batch[-1] = r_batch[-1]  # terminal state
        else:
            with torch.no_grad():
                s_tensor = torch.from_numpy(np.array(s_batch)).to(torch.float32).to(self.device)
                val = self.critic.forward(s_tensor)
                R_batch[-1] = val[-1].item()  # bootstrap from last state

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + self.gamma * R_batch[t + 1]

        return list(R_batch)

    def get_network_params(self) -> Tuple[Dict, Dict]:
        """Get the current network parameters.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L89-L90

        Returns:
            Tuple of (actor_state_dict, critic_state_dict).
        """
        return [self.actor.state_dict(), self.critic.state_dict()]

    def set_network_params(self, params: Tuple[Dict, Dict]) -> None:
        """Set the network parameters.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L92-L95

        Args:
            params: Tuple of (actor_state_dict, critic_state_dict).
        """
        actor_net_params, critic_net_params = params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def save_model(self, path: str) -> None:
        """Save the model to a file.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L142-L144

        Args:
            path: Path to save the model.
        """
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        torch.save(model_params, path)

    def load_model(self, path: str) -> None:
        """Load the model from a file.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/ppo2.py#L137-L140

        Args:
            path: Path to load the model from.
        """
        actor_model_params, critic_model_params = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)

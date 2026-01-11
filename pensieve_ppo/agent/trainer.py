"""Trainer class for distributed RL training.

This module implements the Trainer class that coordinates distributed training
using multiple parallel agents, following the architecture from the original
Pensieve-PPO implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from .abc import AbstractAgent


class Trainer:
    """Distributed trainer for RL agents.

    This class implements a distributed training architecture with:
    - A central coordinator that aggregates experiences and updates the model
    - Multiple parallel agents that collect experiences

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
    """

    def __init__(
        self,
        agent_class: Type[AbstractAgent],
        agent_kwargs: Dict[str, Any],
        env_factory: Callable[[int], Any],
        num_agents: int = 16,
        train_seq_len: int = 1000,
        train_epochs: int = 500000,
        model_save_interval: int = 300,
        random_seed: int = 42,
        summary_dir: str = './ppo',
        nn_model: Optional[str] = None,
    ):
        """Initialize the trainer.

        Args:
            agent_class: The agent class to instantiate (e.g., PPOAgent).
            agent_kwargs: Keyword arguments for creating agents.
            env_factory: Factory function that takes agent_id and returns an env.
            num_agents: Number of parallel agents.
            train_seq_len: Sequence length for each training batch.
            train_epochs: Total number of training epochs.
            model_save_interval: Interval for saving model checkpoints.
            random_seed: Random seed for reproducibility.
            summary_dir: Directory for saving logs and models.
            nn_model: Path to pre-trained model to load (optional).
        """
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.env_factory = env_factory
        self.num_agents = num_agents
        self.train_seq_len = train_seq_len
        self.train_epochs = train_epochs
        self.model_save_interval = model_save_interval
        self.random_seed = random_seed
        self.summary_dir = summary_dir
        self.nn_model = nn_model

        # Callbacks for logging (set by train.py)
        self.on_epoch_end: Optional[Callable[[int, AbstractAgent, Dict], None]] = None
        self.on_save_model: Optional[Callable[[int, str, AbstractAgent], Tuple[float, float]]] = None

    def _central_agent(
        self,
        net_params_queues: List[mp.Queue],
        exp_queues: List[mp.Queue],
    ) -> None:
        """Central coordinator that aggregates experiences and updates model.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L127

        Args:
            net_params_queues: Queues for sending network parameters to agents.
            exp_queues: Queues for receiving experiences from agents.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L79-L80
        assert len(net_params_queues) == self.num_agents
        assert len(exp_queues) == self.num_agents

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L83-L85
        actor = self.agent_class(**self.agent_kwargs)

        # restore neural net parameters
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L90-L114
        if self.nn_model is not None:
            actor.load_model(self.nn_model)
            print('Model restored.')

        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(self.train_epochs):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(self.num_agents):
                net_params_queues[i].put(actor_net_params)

            s, a, p, r = [], [], [], []
            for i in range(self.num_agents):
                s_, a_, p_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                r += r_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(r)

            train_info = actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            # Callback for epoch end
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, actor, train_info)

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L116-L127
            if epoch % self.model_save_interval == 0:
                # Save the neural net parameters to disk.
                model_path = f'{self.summary_dir}/nn_model_ep_{epoch}.pth'
                actor.save_model(model_path)

                # Callback for model saving (e.g., testing and logging)
                if self.on_save_model is not None:
                    self.on_save_model(epoch, model_path, actor)

    def _agent_worker(
        self,
        agent_id: int,
        net_params_queue: mp.Queue,
        exp_queue: mp.Queue,
    ) -> None:
        """Worker agent that collects experiences.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L130-L165

        Args:
            agent_id: Unique identifier for this agent.
            net_params_queue: Queue for receiving network parameters.
            exp_queue: Queue for sending experiences.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L131-L141
        env = self.env_factory(agent_id)
        actor = self.agent_class(**self.agent_kwargs)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        for epoch in range(self.train_epochs):
            obs, _ = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []

            done = False
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L142-L152
            for step in range(self.train_seq_len):
                s_batch.append(obs)

                action, action_prob = actor.select_action(obs)

                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L154-L160
                action_vec = actor.create_action_vector(action)
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)
                if done:
                    break
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L161-L165
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
            exp_queue.put([s_batch, a_batch, p_batch, v_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

    def train(self) -> None:
        """Start distributed training.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L167-L198
        """
        np.random.seed(self.random_seed)
        torch.set_num_threads(1)
        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(self.num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        # create a coordinator and multiple agent processes
        # (note: threading is not desirable due to python GIL)
        coordinator = mp.Process(target=self._central_agent,
                                 args=(net_params_queues, exp_queues))
        coordinator.start()

        agents = []
        for i in range(self.num_agents):
            agents.append(mp.Process(target=self._agent_worker,
                                     args=(i,
                                           net_params_queues[i],
                                           exp_queues[i])))
        for i in range(self.num_agents):
            agents[i].start()

        # wait until training is done
        coordinator.join()

"""Trainer class for distributed RL training.

This module implements the Trainer class that coordinates distributed training
using multiple parallel agents, following the architecture from the original
Pensieve-PPO implementation.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Union

import gymnasium as gym
import torch

from .abc import AbstractAgent
from .trainable import AbstractTrainableAgent, Step, TrainingBatch


class EpochEndCallback:
    """No-op callback for trainer events."""

    def __call__(self, epoch: int, agent: AbstractTrainableAgent, info: Dict) -> None:
        pass


class SaveModelCallback:
    """No-op callback for trainer events."""

    def __call__(self, epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None:
        pass


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
        env_factory: Callable[[int], gym.Env],
        agent_factory: Callable[[], AbstractTrainableAgent],
        parallel_workers: int = 16,
        steps_per_epoch: int = 1000,
        train_epochs: int = 500000,
        model_save_interval: int = 300,
        output_dir: str = './ppo',
        pretrained_model_path: Optional[str] = None,
        on_epoch_end: Callable[[int, AbstractTrainableAgent, Dict], None] = EpochEndCallback(),
        on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = SaveModelCallback(),
        # additional parameter to allow different agent implementations for workers
        agent_factory_for_worker: Optional[Callable[[], Union[AbstractAgent, AbstractTrainableAgent]]] = None,
        sync_params: bool = True,
    ):
        """Initialize the trainer.

        Args:
            env_factory: Factory function (agent_id: int) -> env.
            agent_factory: Factory function () -> AbstractTrainableAgent for central agent.
            parallel_workers: Number of parallel worker agents for distributed training.
            steps_per_epoch: Number of environment steps each worker collects per epoch.
            train_epochs: Total number of training epochs.
            model_save_interval: Interval for saving model checkpoints.
            output_dir: Directory for saving logs and model checkpoints.
            pretrained_model_path: Path to pre-trained model to load (optional).
            on_epoch_end: Callback invoked at the end of each epoch.
            on_save_model: Callback invoked when model is saved.
            agent_factory_for_worker: Factory function () -> AbstractAgent | AbstractTrainableAgent
                for worker agents. If None, uses agent_factory. This allows workers to use a
                different (possibly lighter) agent implementation than the central agent.
            sync_params: Whether to synchronize network parameters from central agent to workers.
                If True, workers must use AbstractTrainableAgent (requires set_params).
                If False, workers can use any AbstractAgent and params sync is skipped.
        """
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.num_agents = parallel_workers
        self.train_seq_len = steps_per_epoch
        self.train_epochs = train_epochs
        self.model_save_interval = model_save_interval
        self.summary_dir = output_dir
        self.nn_model = pretrained_model_path
        self.on_epoch_end = on_epoch_end
        self.on_save_model = on_save_model

        # additional parameter to allow different agent implementations for workers
        # If agent_factory_for_worker is not provided, default to agent_factory
        self.agent_factory_for_worker = agent_factory_for_worker if agent_factory_for_worker is not None else agent_factory
        self.sync_params = sync_params

    def central_agent(
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
        actor = self.agent_factory()

        # restore neural net parameters
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L90-L100
        if self.nn_model is not None:
            actor.load(self.nn_model)
            print('Model restored.')

        # while True:  # assemble training batches from agents, compute the gradients
        for epoch in range(self.train_epochs):
            # Synchronize the network parameters to worker agents.
            # If sync_params=True, send actual params for workers to update their models.
            # If sync_params=False, send None as a signal to proceed (no param sync needed,
            # e.g., when workers use a different non-trainable agent like LLM-based agent).
            actor_net_params = actor.get_params() if self.sync_params else None
            for i in range(self.num_agents):
                net_params_queues[i].put(actor_net_params)

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L102-L112
            training_batches: List[TrainingBatch] = []
            for i in range(self.num_agents):
                trajectory, done = exp_queues[i].get()
                # as the actor in central agent and worker agent may be different, we should produce the training batch here
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L161-L165
                training_batch = actor.produce_training_batch(trajectory, done)
                training_batches.append(training_batch)

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L102-L114
            train_info = actor.train_batch(training_batches, epoch)

            # Callback for epoch end
            self.on_epoch_end(epoch, actor, train_info)

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L116-L127
            if epoch % self.model_save_interval == 0:
                # Save the neural net parameters to disk.
                model_path = f'{self.summary_dir}/nn_model_ep_{epoch}.pth'
                actor.save(model_path)

                # Callback for model saving (e.g., testing and logging)
                self.on_save_model(epoch, model_path, actor)

    def agent_worker(
        self,
        agent_id: int,
        net_params_queue: mp.Queue,
        exp_queue: mp.Queue,
    ) -> None:
        """Worker agent that collects experiences.

        The worker can use either AbstractTrainableAgent or AbstractAgent depending on
        agent_factory_for_worker and sync_params settings:

        - If sync_params=True: Worker agent must be AbstractTrainableAgent because we need
          set_params() to synchronize network parameters from central agent. Action selection
          uses select_action_for_training() for exploration.

        - If sync_params=False: Worker agent can be any AbstractAgent (including non-trainable
          agents like LLM-based agents). No parameter sync occurs. Action selection uses
          select_action() if the agent is not AbstractTrainableAgent, or
          select_action_for_training() if it is.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L130-L165

        Args:
            agent_id: Unique identifier for this agent.
            net_params_queue: Queue for receiving network parameters (or None if sync_params=False).
            exp_queue: Queue for sending experiences.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L131-L140
        env = self.env_factory(agent_id)
        # Use agent_factory_for_worker to create the worker agent.
        # This allows workers to use a different agent implementation than the central agent.
        actor = self.agent_factory_for_worker()

        actor_net_params = net_params_queue.get()
        if self.sync_params:
            # If sync_params=True, the worker agent MUST be AbstractTrainableAgent because we need set_params() to synchronize network parameters.
            assert isinstance(actor, AbstractTrainableAgent), (
                "sync_params=True requires worker agent to be AbstractTrainableAgent "
                "(needs set_params() for parameter synchronization). "
                f"Got {type(actor).__name__} instead."
            )
            actor.set_params(actor_net_params)

        for epoch in range(self.train_epochs):
            obs, _ = env.reset()
            trajectory: List[Step] = []
            for step in range(self.train_seq_len):
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L145-L150
                # Action selection strategy depends on whether the actor is trainable:
                # - AbstractTrainableAgent: use select_action_for_training() which includes exploration noise (e.g., Gumbel-softmax sampling)
                # - AbstractAgent: use select_action() which is typically deterministic (e.g., for LLM-based agents that don't need exploration noise)
                if isinstance(actor, AbstractTrainableAgent):
                    action, action_prob = actor.select_action_for_training(obs)
                else:
                    action, action_prob = actor.select_action(obs)

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L152
                next_obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L154-L155
                action_vec = [0] * len(action_prob)
                action_vec[action] = 1

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L143
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L156-158
                trajectory.append(Step(state=obs, action=action_vec, action_prob=action_prob, reward=rew, step=step))

                obs = next_obs
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L159-160
                if done:
                    break
            # As the actor in central agent and worker agent may be different, we should not produce the training batch here
            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L162
            exp_queue.put((trajectory, done))

            actor_net_params = net_params_queue.get()
            if self.sync_params:
                # If sync_params=True, the worker agent MUST be AbstractTrainableAgent because we need set_params() to synchronize network parameters.
                assert isinstance(actor, AbstractTrainableAgent), (
                    "sync_params=True requires worker agent to be AbstractTrainableAgent "
                    "(needs set_params() for parameter synchronization). "
                    f"Got {type(actor).__name__} instead."
                )
                actor.set_params(actor_net_params)

    def train(self) -> None:
        """Start distributed training.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L167-L198
        """
        torch.set_num_threads(1)
        # inter-process communication queues
        net_params_queues = []
        exp_queues = []
        for i in range(self.num_agents):
            net_params_queues.append(mp.Queue(1))
            exp_queues.append(mp.Queue(1))

        # create a coordinator and multiple agent processes
        # (note: threading is not desirable due to python GIL)
        coordinator = mp.Process(target=self.central_agent,
                                 args=(net_params_queues, exp_queues))
        coordinator.start()

        agents = []
        for i in range(self.num_agents):
            agents.append(mp.Process(target=self.agent_worker,
                                     args=(i,
                                           net_params_queues[i],
                                           exp_queues[i])))
        for i in range(self.num_agents):
            agents[i].start()

        # wait until training is done
        coordinator.join()

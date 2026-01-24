"""ImitationTrainer class for distributed imitation learning.

This module implements the ImitationTrainer class for behavioral cloning / imitation learning,
where a student agent learns to imitate a teacher agent's decisions, following the architecture
from the original Pensieve-PPO implementation.

The key difference from standard distributed RL training (Trainer):
- Worker agents use a "teacher" agent (e.g., MPC, BBA, LLM-based) to collect trajectories
- Central agent (neural network) learns to imitate the teacher's decisions
- Environment outputs ImitationState with both student_state and teacher_state
- No parameter synchronization between central and worker agents

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import multiprocessing as mp
from typing import Callable, List

from ..gym import ImitationState
from .abc import AbstractAgent
from .trainable import Step, TrainingBatch
from .trainer import Trainer


class ImitationTrainer(Trainer):
    """Distributed trainer for imitation learning.

    This class implements a distributed imitation learning architecture with:
    - A central coordinator (student agent) that aggregates experiences and updates the model
    - Multiple parallel worker agents (teacher agents) that collect experiences

    The env outputs ImitationState:
    - teacher_state: used by worker agents (teacher) for action selection
    - student_state: sent to central agent for training the student neural network

    No parameter synchronization occurs between central and worker agents since
    they use different agent implementations (student learns to imitate teacher).

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
    """

    def __init__(
        self,
        *args,
        teacher_agent_factory: Callable[[], AbstractAgent],
        **kwargs
    ):
        """Initialize the imitation trainer.

        Args:
            *args: Positional arguments passed to Trainer.__init__.
                Note: The environment MUST output ImitationState containing both
                student_state and teacher_state.
            teacher_agent_factory: Factory function () -> AbstractAgent for worker agents.
                This is the "teacher" agent (e.g., MPC, BBA, LLM-based) whose decisions
                the student will learn to imitate.
            **kwargs: Keyword arguments passed to Trainer.__init__.
                See Trainer.__init__ for available parameters.
        """
        super().__init__(*args, **kwargs)
        self.teacher_agent_factory = teacher_agent_factory

    def central_agent(
        self,
        signal_queues: List[mp.Queue],
        exp_queues: List[mp.Queue],
    ) -> None:
        """Central coordinator that aggregates experiences and updates model.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L77-L127

        Args:
            signal_queues: Queues for sending signals to worker agents to proceed.
            exp_queues: Queues for receiving experiences from agents.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L79-L80
        assert len(signal_queues) == self.num_agents
        assert len(exp_queues) == self.num_agents

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L83-L85
        actor = self.agent_factory()

        # Model loading is now handled in create_agent via model_path parameter
        # # restore neural net parameters
        # # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L90-L100
        # if self.nn_model is not None:
        #     actor.load(self.nn_model)
        #     print('Model restored.')

        # while True:  # assemble training batches from agents, compute the gradients
        for epoch in range(1, self.train_epochs + 1):
            # Send signal to worker agents to proceed (no parameter sync needed for imitation learning)
            for i in range(self.num_agents):
                signal_queues[i].put(None)

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L102-L112
            training_batches: List[TrainingBatch] = []
            for i in range(self.num_agents):
                trajectory, done = exp_queues[i].get()
                # The trajectory contains:
                # - state: student_state (for training the student neural network)
                # - action: teacher's action (what we want to imitate)
                # - action_prob: teacher's action probabilities
                # - reward: reward from environment
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
        signal_queue: mp.Queue,
        exp_queue: mp.Queue,
    ) -> None:
        """Worker agent that collects experiences using the teacher agent.

        Reference:
            https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L130-L165

        Args:
            agent_id: Unique identifier for this agent.
            signal_queue: Queue for receiving signals from central agent to proceed.
            exp_queue: Queue for sending experiences.
        """
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L131-L140
        env = self.env_factory(agent_id)
        actor = self.teacher_agent_factory()

        signal_queue.get()

        for epoch in range(1, self.train_epochs + 1):
            obs, _ = env.reset()
            # obs should be ImitationState
            assert isinstance(obs, ImitationState), (
                f"Environment must output ImitationState, got {type(obs).__name__}. "
                "Use ImitationObserver in your environment."
            )

            trajectory: List[Step] = []
            for step in range(self.train_seq_len):
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L145-L150
                # Use teacher_state for teacher agent's decision making
                action, action_prob = actor.select_action(obs.teacher_state)

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L152
                next_obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L154-L155
                action_vec = [0] * len(action_prob)
                action_vec[action] = 1

                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L143
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L156-158
                # Store step with student_state for training the student agent
                # The action and action_prob are from the teacher, which the student will learn to imitate
                trajectory.append(Step(state=obs.student_state, action=action_vec, action_prob=action_prob, reward=rew, step=step, done=done))

                obs = next_obs
                # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L159-160
                if done:
                    break

            # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L162
            exp_queue.put((trajectory, done))

            signal_queue.get()

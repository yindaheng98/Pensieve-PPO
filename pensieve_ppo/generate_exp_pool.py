"""Experience pool generation script for Pensieve PPO.

This module implements experience pool generation using the imitation learning
infrastructure. It uses an actor agent (similar to the teacher in imitation learning)
to collect trajectories, and writes the training data to an experience pool file
instead of training a neural network.

The key concepts:
- Actor agent: The agent that makes decisions during rollouts (e.g., MPC, BBA, trained PPO).
  This is equivalent to the "teacher" agent in imitation learning.
- Observer agent: A wrapper that receives training batches and writes them to an experience
  pool file instead of performing actual training. This wraps a dummy AbstractTrainableAgent.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
"""

import argparse
import dataclasses
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .agent import AbstractTrainableAgent, ImitationTrainer, get_available_agents
from .agent.trainable import Step, TrainingBatch
from .imitate import prepare_imitation, add_teacher_arguments
from .args import add_env_agent_arguments, parse_env_agent_args, parse_options
from .train import add_training_arguments


@dataclass
class DictTrainingBatch(TrainingBatch):
    """A flexible training batch that stores data as a dictionary of lists.

    This batch type dynamically adapts its fields based on the state's fields,
    making it suitable for experience pool generation where the exact state
    structure may vary.

    Attributes:
        data: Dictionary mapping field names to lists of values.
    """
    data: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class ExperiencePool:
    """Experience pool for collecting trajectories.

    This is similar to the ExperiencePool in the NetLLM reference implementation.
    Stores training batch data collected during rollouts, with fields dynamically
    determined by the TrainingBatch dataclass structure.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/plm_special/data/exp_pool.py
    """
    data: Dict[str, List[Any]] = field(default_factory=dict)

    def add_batch(self, batch: DictTrainingBatch) -> None:
        """Add a training batch to the pool.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L365-L366

        Args:
            batch: A DictTrainingBatch instance.
        """
        for field_name, field_value in batch.data.items():
            if field_name not in self.data:
                self.data[field_name] = []
            self.data[field_name].extend(field_value)

    def __len__(self) -> int:
        if not self.data:
            return 0
        # Return length of first field
        first_key = next(iter(self.data))
        return len(self.data[first_key])

    def save(self, path: str) -> None:
        """Save the experience pool to disk.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L367-L368

        Args:
            path: Path to save the experience pool file.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class ExpPoolWriterAgent(AbstractTrainableAgent):
    """A wrapper agent that writes training data to an experience pool instead of training.

    This agent wraps any AbstractTrainableAgent and overrides the train_batch method
    to write the training batches to an experience pool file. All other methods are
    delegated to the wrapped agent.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
    """

    def __init__(
        self,
        wrapped_agent: AbstractTrainableAgent,
        exp_pool: ExperiencePool,
        exp_pool_path: str,
    ):
        """Initialize the experience pool writer agent.

        Args:
            wrapped_agent: The underlying AbstractTrainableAgent to wrap.
            exp_pool: The experience pool to write data to.
            exp_pool_path: Path to save the experience pool file.
        """
        self._wrapped_agent = wrapped_agent
        self._exp_pool = exp_pool
        self._exp_pool_path = exp_pool_path

    def get_params(self):
        raise NotImplementedError("ExpPoolWriterAgent.get_params should not be called")

    def set_params(self, params):
        raise NotImplementedError("ExpPoolWriterAgent.set_params should not be called")

    def select_action(self, state):
        raise NotImplementedError("ExpPoolWriterAgent.select_action should not be called")

    def select_action_for_training(self, state):
        raise NotImplementedError("ExpPoolWriterAgent.select_action_for_training should not be called")

    def produce_training_batch(
        self,
        trajectory: List[Step],
        done: bool,
    ) -> DictTrainingBatch:
        """Produce a training batch from a trajectory by extracting state fields.

        This method extracts all fields from each state in the trajectory and
        concatenates them into lists, creating a flexible DictTrainingBatch.

        Args:
            trajectory: List of steps collected during environment rollout.
            done: Whether the trajectory ended in a terminal state.

        Returns:
            DictTrainingBatch with fields extracted from states.
        """
        if not trajectory:
            return DictTrainingBatch(data={})

        data: Dict[str, List[Any]] = {}

        for step in trajectory:
            state = step.state
            # Extract all fields from state dataclass
            for f in dataclasses.fields(state):
                field_name = f.name
                field_value = getattr(state, field_name)
                if field_name not in data:
                    data[field_name] = []
                data[field_name].append(field_value)

        return DictTrainingBatch(data=data)

    def save(self, path: str = None) -> None:
        """Save the experience pool instead of the model.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L367-L368

        Args:
            path: Ignored. Always saves to the configured exp_pool_path.
        """
        self._exp_pool.save(self._exp_pool_path)

    def train_batch(
        self,
        training_batches: List[DictTrainingBatch],
        epoch: int,
    ) -> Dict[str, float]:
        """Write training batch data to the experience pool instead of training.

        This method adds each training batch to the experience pool.

        Reference:
            https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L365-L366

        Args:
            training_batches: List of DictTrainingBatch from workers.
            epoch: Current training epoch.

        Returns:
            Dictionary containing experience collection metrics.
        """
        total_samples = 0
        for batch in training_batches:
            self._exp_pool.add_batch(batch)
            # Estimate samples from first list field
            if batch.data:
                first_key = next(iter(batch.data))
                total_samples += len(batch.data[first_key])

        return {
            'exp_pool_size': len(self._exp_pool),
            'new_samples': total_samples,
            'epoch': epoch,
        }


class ExpPoolWriterAgentFactory:
    """Factory class for creating ExpPoolWriterAgent instances.

    This callable class wraps an original agent factory and creates ExpPoolWriterAgent
    instances that write to a shared experience pool.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
    """

    def __init__(
        self,
        original_factory: Callable[[], AbstractTrainableAgent],
        exp_pool: ExperiencePool,
        exp_pool_path: str,
    ):
        """Initialize the factory.

        Args:
            original_factory: The original agent factory function.
            exp_pool: The experience pool to write data to.
            exp_pool_path: Path to save the experience pool file.
        """
        self._original_factory = original_factory
        self._exp_pool = exp_pool
        self._exp_pool_path = exp_pool_path

    def __call__(self) -> ExpPoolWriterAgent:
        """Create an ExpPoolWriterAgent instance."""
        wrapped_agent = self._original_factory()
        return ExpPoolWriterAgent(
            wrapped_agent=wrapped_agent,
            exp_pool=self._exp_pool,
            exp_pool_path=self._exp_pool_path,
        )


def exp_pool_epoch_end_callback(epoch: int, actor: ExpPoolWriterAgent, train_info: Dict[str, Any]) -> None:
    """Callback for logging epoch information and saving experience pool.

    This callback is invoked at the end of each epoch during experience pool generation.
    It outputs the epoch number and training info, and saves the experience pool.

    Reference:
        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py#L31-L75

    Args:
        epoch: Current training epoch.
        actor: The actor agent (ExpPoolWriterAgent).
        train_info: Dictionary containing training/collection metrics.
    """
    # Output epoch and train_info
    info_str = ', '.join(f'{k}={v}' for k, v in train_info.items())
    print(f'Epoch {epoch}: {info_str}')

    # Save experience pool
    actor.save()


def prepare_exp_pool_generation(
    *args,
    exp_pool_path: str,
    **kwargs,
) -> Tuple[ImitationTrainer, ExperiencePool]:
    """Prepare trainer for experience pool generation.

    This function calls prepare_imitation and modifies the returned trainer's
    agent_factory to use ExpPoolWriterAgent for writing data to an experience pool.

    Architecture:
    - Actor agent (--teacher-agent-name): Makes decisions during rollouts.
      This is equivalent to the "teacher" agent in imitation learning.
    - Observer agent: A wrapper that receives training batches and writes them to
      an experience pool file instead of performing actual training.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L315-L369

    Args:
        *args: Positional arguments passed to prepare_imitation.
        exp_pool_path: Path to save the experience pool file.
        **kwargs: Keyword arguments passed to prepare_imitation.

    Returns:
        Tuple of (configured ImitationTrainer, ExperiencePool).
    """
    # Call prepare_imitation to get a configured trainer
    trainer = prepare_imitation(*args, **kwargs, on_epoch_end=exp_pool_epoch_end_callback)  # Epoch end callback for logging and saving

    # Create experience pool
    # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L334
    exp_pool = ExperiencePool()

    # Replace the agent factory in the trainer with ExpPoolWriterAgentFactory
    trainer.agent_factory = ExpPoolWriterAgentFactory(
        original_factory=trainer.agent_factory,
        exp_pool=exp_pool,
        exp_pool_path=exp_pool_path,
    )

    return trainer, exp_pool


def add_exp_pool_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for experience pool configuration.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument('--exp-pool-path', type=str, default='./exp_pool/exp_pool.pkl',
                        dest='exp_pool_path',
                        help="Path to save the experience pool file (default: ./exp_pool/exp_pool.pkl)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experience pool using actor agent rollouts')
    add_env_agent_arguments(parser, available_agents=get_available_agents())
    add_training_arguments(parser)
    # Actor arguments use --teacher-* prefix (equivalent to teacher in imitation learning)
    add_teacher_arguments(parser, available_agents=get_available_agents())
    add_exp_pool_arguments(parser)
    args = parser.parse_args()

    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)
    args.teacher_agent_options = parse_options(args.teacher_agent_options)

    # Prepare trainer for experience pool generation
    # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L315-L369
    trainer, exp_pool = prepare_exp_pool_generation(
        trace_folder=args.train_trace_folder,
        # Observer agent parameters (the agent that records data)
        # This uses the main agent name for the underlying trainable agent interface
        student_name=args.agent_name,
        student_model_path=args.model_path,
        student_device=args.device,
        student_agent_options=args.agent_options,
        # Actor agent parameters (the agent that makes decisions)
        # Uses --teacher-* arguments (equivalent to teacher in imitation learning)
        teacher_name=args.teacher_agent_name,
        teacher_model_path=args.teacher_model_path,
        teacher_device=args.teacher_device,
        teacher_agent_options=args.teacher_agent_options,
        # Shared parameters
        levels_quality=args.levels_quality,
        state_history_len=args.state_history_len,
        initial_level=args.initial_level,
        env_options=args.env_options,
        # Training parameters (control how much data to collect)
        output_dir=args.output_dir,
        parallel_workers=args.parallel_workers,
        steps_per_epoch=args.steps_per_epoch,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        pretrained_model_path=args.pretrained_model_path,
        on_save_model=None,  # No model saving callback needed
        # Experience pool parameters
        exp_pool_path=args.exp_pool_path,
    )

    # Start experience pool generation
    # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L372-L397
    print(f"Starting experience pool generation with {args.parallel_workers} parallel workers...")
    print(f"Actor agent: {args.teacher_agent_name}")
    print(f"Observer agent: {args.agent_name}")
    print(f"Experience pool path: {args.exp_pool_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total epochs: {args.train_epochs}")
    print(f"Steps per epoch: {args.steps_per_epoch}")
    trainer.train()

    # Load the final saved experience pool from disk
    # Note: The exp_pool object in this process is empty because the actual data
    # collection happens in a subprocess. The subprocess saves the data periodically
    # via ExpPoolWriterAgent.save() calls.
    # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L367-L369
    if os.path.exists(args.exp_pool_path):
        with open(args.exp_pool_path, 'rb') as f:
            final_exp_pool = pickle.load(f)
        print(f"\nDone. Experience pool saved at: {args.exp_pool_path}")
        print(f"Total samples collected: {len(final_exp_pool.data.get(next(iter(final_exp_pool.data)), []))}" if final_exp_pool.data else "0")
    else:
        print(f"\nWarning: Experience pool file not found at {args.exp_pool_path}")
        print("This may happen if no epochs were saved. Check model_save_interval setting.")

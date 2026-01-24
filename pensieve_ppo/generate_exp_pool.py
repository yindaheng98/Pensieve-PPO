"""Experience pool generation script for Pensieve PPO.

This module implements experience pool generation using the imitation learning
infrastructure. It uses an actor agent (similar to the teacher in imitation learning)
to collect trajectories, and writes the training data to an experience pool file
instead of training a neural network.

The key concepts:
- Actor agent: The agent that makes decisions during rollouts (e.g., MPC, BBA, trained PPO).
  This is equivalent to the "teacher" agent in imitation learning.
- Observer agent: ExpPoolWriterAgent that receives training batches and writes them to an
  experience pool file instead of performing actual training.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
    https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
"""

import argparse
import os
from typing import Any, Dict

from .agent import ImitationTrainer, get_available_agents
from .imitate import prepare_imitation, add_teacher_arguments
from .args import add_env_agent_arguments, parse_env_agent_args, parse_options
from .train import add_training_arguments
from .exp_pool import ExperiencePool, ExpPoolWriterAgent

# Default path for experience pool file
EXP_POOL_PATH = './exp_pool/exp_pool.pkl'


class ExpPoolWriterAgentFactory:
    """Factory class for creating ExpPoolWriterAgent instances.

    This callable class creates ExpPoolWriterAgent instances that write to an
    experience pool file.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py
    """

    def __init__(self, exp_pool_path: str):
        """Initialize the factory.

        Args:
            exp_pool_path: Path to save the experience pool file.
        """
        self._exp_pool_path = exp_pool_path

    def __call__(self) -> ExpPoolWriterAgent:
        """Create an ExpPoolWriterAgent instance."""
        return ExpPoolWriterAgent(exp_pool_path=self._exp_pool_path)


def exp_pool_epoch_end_callback(epoch: int, actor: ExpPoolWriterAgent, train_info: Dict[str, Any]) -> None:
    """Callback for logging epoch information.

    This callback is invoked at the end of each epoch during experience pool generation.
    It outputs the epoch number and training info.

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


def exp_pool_save_callback(epoch: int, model_path: str, actor: ExpPoolWriterAgent) -> None:
    """Callback for saving experience pool when model save interval is reached.

    This callback is invoked when the model would be saved (based on model_save_interval).
    It saves the experience pool to disk.

    Args:
        epoch: Current training epoch.
        model_path: Path where model would be saved (ignored, we use exp_pool_path).
        actor: The actor agent (ExpPoolWriterAgent).
    """
    print(f'Epoch {epoch}: Saving experience pool...')
    actor.save()


def prepare_exp_pool_generation(
    *args,
    exp_pool_path: str,
    **kwargs,
) -> ImitationTrainer:
    """Prepare trainer for experience pool generation.

    This function calls prepare_imitation and modifies the returned trainer's
    agent_factory to use ExpPoolWriterAgent for writing data to an experience pool.

    Architecture:
    - Actor agent (--teacher-agent-name): Makes decisions during rollouts.
      This is equivalent to the "teacher" agent in imitation learning.
    - Observer agent: ExpPoolWriterAgent that receives training batches and writes
      them to an experience pool file instead of performing actual training.

    Reference:
        https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L315-L369

    Args:
        *args: Positional arguments passed to prepare_imitation.
        exp_pool_path: Path to save the experience pool file.
        **kwargs: Keyword arguments passed to prepare_imitation.

    Returns:
        Configured ImitationTrainer.
    """
    # Call prepare_imitation to get a configured trainer
    trainer = prepare_imitation(*args, **kwargs)

    # Replace the agent factory in the trainer with ExpPoolWriterAgentFactory
    trainer.agent_factory = ExpPoolWriterAgentFactory(exp_pool_path=exp_pool_path)

    return trainer


def add_exp_pool_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for experience pool configuration.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument('--exp-pool-path', type=str, default=EXP_POOL_PATH,
                        dest='exp_pool_path',
                        help=f"Path to the experience pool file (default: {EXP_POOL_PATH})")


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
    trainer = prepare_exp_pool_generation(
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
        max_steps_per_epoch=args.max_steps_per_epoch,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        on_epoch_end=exp_pool_epoch_end_callback,
        on_save_model=exp_pool_save_callback,
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
    print(f"Max steps per epoch: {args.max_steps_per_epoch}")
    trainer.train()

    # Load the final saved experience pool from disk
    # Note: The data collection happens in a subprocess. The subprocess saves the data
    # periodically via ExpPoolWriterAgent.save() calls.
    # Reference: https://github.com/duowuyms/NetLLM/blob/105bcf070f2bec808f7b14f8f5a953de6e4e6e54/adaptive_bitrate_streaming/generate_exp_pool.py#L367-L369
    if os.path.exists(args.exp_pool_path):
        final_exp_pool = ExperiencePool.load(args.exp_pool_path)
        print(f"\nDone. Experience pool saved at: {args.exp_pool_path}")
        print(f"Total samples collected: {len(final_exp_pool)}")
    else:
        print(f"\nWarning: Experience pool file not found at {args.exp_pool_path}")
        print("This may happen if no epochs were saved. Check model_save_interval setting.")

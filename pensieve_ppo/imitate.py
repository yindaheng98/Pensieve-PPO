"""Imitation learning script for Pensieve PPO.

This module implements the complete imitation learning pipeline using distributed
parallel agents, where worker agents use a teacher agent (e.g., LLM-based agent)
to collect trajectories, and a central student agent (neural network) learns to
imitate those decisions through behavioral cloning.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import argparse
import os
from typing import Callable, Optional, Union

import torch

from .agent import AbstractTrainableAgent, Trainer, get_available_agents
from .defaults import create_agent_factory_with_default, create_env_agent_factory_with_default, S_INFO, S_LEN, VIDEO_BIT_RATE, TRAIN_TRACES, DEFAULT_QUALITY
from .args import add_env_agent_arguments, parse_env_agent_args, parse_options
from .test import add_testing_arguments
from .train import TestingCallback, add_training_arguments, NUM_AGENTS, TRAIN_SEQ_LEN, TRAIN_EPOCH, MODEL_SAVE_INTERVAL, SUMMARY_DIR


def prepare_imitation(
    output_dir: str = SUMMARY_DIR,
    parallel_workers: int = NUM_AGENTS,
    steps_per_epoch: int = TRAIN_SEQ_LEN,
    train_epochs: int = TRAIN_EPOCH,
    model_save_interval: int = MODEL_SAVE_INTERVAL,
    pretrained_model_path: str = None,
    on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = None,
    # Student agent parameters
    name: str = 'ppo',
    model_path: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
    agent_options: dict = {},
    # Teacher agent parameters
    teacher_name: str = 'bba',
    teacher_model_path: Optional[str] = None,
    teacher_device: Optional[Union[torch.device, str]] = None,
    teacher_agent_options: dict = {},
    # Teacher environment parameters
    teacher_trace_folder: Optional[str] = None,
    teacher_env_options: dict = {},
    # Shared compatibility parameters (must be same for student and teacher)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
) -> Trainer:
    """Prepare trainer for distributed imitation learning.

    This function creates a Trainer configured for imitation learning, where:
    - Worker agents use a teacher agent (e.g., LLM-based agent) to collect trajectories
    - Central student agent (neural network) learns to imitate teacher decisions
    - No parameter synchronization occurs (sync_params=False)

    Args:
        output_dir: Directory for saving logs and model checkpoints.
        parallel_workers: Number of parallel worker agents.
        steps_per_epoch: Number of environment steps per epoch per worker.
        train_epochs: Total number of training epochs.
        model_save_interval: Interval for saving model checkpoints.
        pretrained_model_path: Path to pre-trained model to resume from (for student agent).
        on_save_model: Callback function invoked when model is saved.
                     Signature: (epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None
        name: Agent name for student agent.
        model_path: Path to load pre-trained model weights for student agent.
        device: PyTorch device for student agent.
        agent_options: Additional kwargs for student agent.
        teacher_name: Agent name for teacher agent.
        teacher_model_path: Path to pre-trained model for teacher agent.
        teacher_device: PyTorch device for teacher agent.
        teacher_agent_options: Additional kwargs for teacher agent.
        teacher_trace_folder: Folder containing network bandwidth trace files for teacher.
        teacher_env_options: Additional kwargs for teacher environment.
        levels_quality: Video bitrate levels (shared between student and teacher).
        state_history_len: Number of past observations in state (shared between student and teacher).

    Returns:
        Configured Trainer instance ready for imitation learning.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create student agent factory (no env needed for student in imitation learning)
    agent_factory = create_agent_factory_with_default(
        model_path=model_path,
        name=name,
        device=device,
        levels_quality=levels_quality,
        state_history_len=state_history_len,
        agent_options=agent_options,
    )

    # Create teacher env_factory and agent_factory
    teacher_env_factory, teacher_agent_factory = create_env_agent_factory_with_default(
        trace_folder=teacher_trace_folder,
        train=True,
        model_path=teacher_model_path,
        name=teacher_name,
        device=teacher_device,
        levels_quality=levels_quality,
        state_history_len=state_history_len,
        env_options=teacher_env_options,
        agent_options=teacher_agent_options,
    )

    # Create trainer with imitation learning configuration
    # - Use teacher_env_factory for worker environments
    # - Use teacher_agent_factory for worker agents (teacher)
    # - Use student_agent_factory for central agent (student)
    # - Set sync_params=False for imitation learning mode
    trainer = Trainer(
        env_factory=teacher_env_factory,
        agent_factory=agent_factory,
        parallel_workers=parallel_workers,
        steps_per_epoch=steps_per_epoch,
        train_epochs=train_epochs,
        model_save_interval=model_save_interval,
        output_dir=output_dir,
        pretrained_model_path=pretrained_model_path,
        on_save_model=on_save_model,
        agent_factory_for_worker=teacher_agent_factory,
        sync_params=False,  # Imitation learning: no parameter sync
    )

    return trainer


def add_teacher_env_agent_arguments(parser: argparse.ArgumentParser, available_agents: list) -> None:
    """Add arguments for teacher agent and environment configuration (for imitation learning).

    This function adds teacher-specific arguments to the teacher subcommand.
    Arguments are NOT prefixed with --teacher since they are in the teacher subcommand.

    Args:
        parser: ArgumentParser (teacher subcommand) to add arguments to.
        available_agents: List of available agent names for validation.
    """
    subparsers = parser.add_subparsers(dest='subcommand', help='Subcommands')
    parser = subparsers.add_parser('teacher', help='Teacher agent and environment configuration')
    parser.add_argument('--agent-name', type=str, default='bba',
                        choices=available_agents, dest='teacher_agent_name',
                        help=f"Algorithm to use for teacher agent (default: 'bba')")
    parser.add_argument('--model-path', type=str, default=None, dest='teacher_model_path',
                        help="Path to load pre-trained model weights for teacher agent (default: None)")
    parser.add_argument('--device', type=str, default=None, dest='teacher_device',
                        help="PyTorch device for teacher agent, e.g. 'cuda', 'cpu' (default: None, auto-select)")
    parser.add_argument('--initial-level', type=int, default=DEFAULT_QUALITY, dest='teacher_initial_level',
                        help=f"Initial quality level index on reset for teacher env (default: {DEFAULT_QUALITY})")
    parser.add_argument('-o', '--agent-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE', dest='teacher_agent_options',
                        help="Extra teacher agent kwargs, e.g. learning_rate=1e-4 gamma=0.99")
    parser.add_argument('-e', '--env-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE', dest='teacher_env_options',
                        help="Extra teacher env kwargs, e.g. rebuf_penalty=4.3 smooth_penalty=1.0")


def parse_env_agent_teacher_args(args: argparse.Namespace) -> argparse.Namespace:
    """Post-process parsed arguments for environment, agent, and teacher agent configuration.

    This function handles:
    - Parsing additional options (agent_options, teacher_agent_options, env_options, teacher_env_options)
    - Setting global random seed if specified.

    Args:
        args: Parsed argument namespace from argparse.

    Returns:
        The modified argument namespace.
    """
    # Parse teacher_agent_options and teacher_env_options
    args.teacher_agent_options = parse_options(args.teacher_agent_options)
    args.teacher_env_options = parse_options(args.teacher_env_options)

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pensieve agent via imitation learning')

    # Add student agent arguments (standard agent arguments)
    add_env_agent_arguments(parser, available_agents=get_available_agents())
    add_testing_arguments(parser)
    add_training_arguments(parser)
    add_teacher_env_agent_arguments(parser, available_agents=get_available_agents())

    args = parser.parse_args()

    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)
    parse_env_agent_teacher_args(args)

    # Create on_save_model callback (reuse from train.py)
    on_save_model = TestingCallback(args=args, output_dir=args.output_dir)

    # Prepare trainer for imitation learning
    trainer = prepare_imitation(
        output_dir=args.output_dir,
        parallel_workers=args.parallel_workers,
        steps_per_epoch=args.steps_per_epoch,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        pretrained_model_path=args.pretrained_model_path,
        on_save_model=on_save_model,
        # Student agent parameters
        name=args.agent_name,
        model_path=None,  # Student agent starts from scratch
        device=args.device,
        agent_options=args.agent_options,
        # Teacher agent parameters
        teacher_name=args.teacher_agent_name,
        teacher_model_path=args.teacher_model_path,
        teacher_device=args.teacher_device,
        teacher_agent_options=args.teacher_agent_options,
        # Teacher environment parameters
        teacher_trace_folder=args.teacher_trace_folder,
        teacher_env_options=args.teacher_env_options,
        # Shared compatibility parameters
        levels_quality=args.levels_quality,
        state_history_len=args.state_history_len,
    )

    # Start imitation learning
    print(f"Starting imitation learning with {args.parallel_workers} parallel workers...")
    print(f"Student agent: {args.agent_name}")
    print(f"Teacher agent: {args.teacher_agent_name}")
    print(f"Output directory: {args.output_dir}")
    trainer.train()

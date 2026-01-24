"""Imitation learning script for Pensieve PPO.

This module implements the complete imitation learning pipeline using distributed
parallel agents, where worker agents use a teacher agent (e.g., BBA, MPC, LLM-based)
to collect trajectories, and a central student agent (neural network) learns to
imitate those decisions through behavioral cloning.

The key difference from standard distributed RL training (train.py):
- Worker agents use a "teacher" agent to collect trajectories
- Central agent (neural network) learns to imitate the teacher's decisions
- Environment outputs ImitationState with both student_state and teacher_state
- No parameter synchronization between central and worker agents

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import argparse
import os
from typing import Callable


from .agent import AbstractTrainableAgent, ImitationTrainer, EpochEndCallback, SaveModelCallback, get_available_agents
from .defaults import create_imitation_env_agent_factory_with_default
from .args import add_env_agent_arguments, parse_env_agent_args, parse_options
from .test import add_testing_arguments
from .train import TestingCallback, add_training_arguments, NUM_AGENTS, TRAIN_SEQ_LEN, TRAIN_EPOCH, MODEL_SAVE_INTERVAL, SUMMARY_DIR


def prepare_imitation(
    *args,
    output_dir: str = SUMMARY_DIR,
    parallel_workers: int = NUM_AGENTS,
    max_steps_per_epoch: int = TRAIN_SEQ_LEN,
    train_epochs: int = TRAIN_EPOCH,
    model_save_interval: int = MODEL_SAVE_INTERVAL,
    on_epoch_end: Callable[[int, AbstractTrainableAgent, dict], None] = EpochEndCallback(),
    on_save_model: Callable[[int, str, AbstractTrainableAgent], None] = SaveModelCallback(),
    **kwargs,
) -> ImitationTrainer:
    """Prepare trainer for distributed imitation learning.

    Wrapper for create_imitation_env_agent_factory_with_default with train=True.
    See create_imitation_env_agent_factory_with_default for available parameters.

    This function creates an ImitationTrainer configured for imitation learning, where:
    - Worker agents use a teacher agent (e.g., BBA, MPC, LLM-based) to collect trajectories
    - Central student agent (neural network) learns to imitate teacher decisions
    - Environment outputs ImitationState with both student_state and teacher_state
    - No parameter synchronization occurs between central and worker agents

    Args:
        *args: Positional arguments passed to create_imitation_env_agent_factory_with_default.
        output_dir: Directory for saving logs and model checkpoints.
        parallel_workers: Number of parallel worker agents.
        max_steps_per_epoch: Maximum number of environment steps per epoch per worker.
            The actual number may be less if the episode terminates or truncates early.
        train_epochs: Total number of training epochs.
        model_save_interval: Interval for saving model checkpoints.
        on_epoch_end: Callback function invoked at the end of each epoch.
                     Signature: (epoch: int, agent: AbstractTrainableAgent, info: dict) -> None
        on_save_model: Callback function invoked when model is saved.
                     Signature: (epoch: int, model_path: str, agent: AbstractTrainableAgent) -> None
        **kwargs: Keyword arguments passed to create_imitation_env_agent_factory_with_default.

    Returns:
        Configured ImitationTrainer instance ready for imitation learning.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create factories for imitation learning
    env_factory, student_agent_factory, teacher_agent_factory = create_imitation_env_agent_factory_with_default(
        *args, train=True, **kwargs,
    )

    # Create ImitationTrainer
    # - env_factory creates environments with ImitationObserver (student + teacher states)
    # - agent_factory creates student agent (neural network to train)
    # - teacher_agent_factory creates teacher agent (expert to imitate)
    trainer = ImitationTrainer(
        env_factory=env_factory,
        agent_factory=student_agent_factory,
        teacher_agent_factory=teacher_agent_factory,
        parallel_workers=parallel_workers,
        max_steps_per_epoch=max_steps_per_epoch,
        train_epochs=train_epochs,
        model_save_interval=model_save_interval,
        output_dir=output_dir,
        on_epoch_end=on_epoch_end,
        on_save_model=on_save_model,
    )

    return trainer


def add_teacher_arguments(parser: argparse.ArgumentParser, available_agents: list) -> None:
    """Add arguments for teacher agent configuration (for imitation learning).

    Args:
        parser: ArgumentParser to add arguments to.
        available_agents: List of available agent names for validation.
    """
    parser.add_argument('--teacher-agent-name', type=str, default='bba',
                        choices=available_agents, dest='teacher_agent_name',
                        help=f"Algorithm to use for teacher agent (default: 'bba')")
    parser.add_argument('--teacher-model-path', type=str, default=None, dest='teacher_model_path',
                        help="Path to load pre-trained model weights for teacher agent (default: None)")
    parser.add_argument('--teacher-device', type=str, default=None, dest='teacher_device',
                        help="PyTorch device for teacher agent, e.g. 'cuda', 'cpu' (default: None, auto-select)")
    parser.add_argument('--teacher-agent-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE', dest='teacher_agent_options',
                        help="Extra teacher agent kwargs, e.g. learning_rate=1e-4 gamma=0.99")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pensieve agent via imitation learning')
    add_env_agent_arguments(parser, available_agents=get_available_agents())
    add_testing_arguments(parser)
    add_training_arguments(parser)
    add_teacher_arguments(parser, available_agents=get_available_agents())
    args = parser.parse_args()

    # Post-process arguments (parse options, set seed)
    parse_env_agent_args(args)
    args.teacher_agent_options = parse_options(args.teacher_agent_options)

    # Create on_save_model callback (reuse from train.py)
    on_save_model = TestingCallback(args=args, output_dir=args.output_dir)

    # Prepare trainer for imitation learning
    trainer = prepare_imitation(
        trace_folder=args.train_trace_folder,
        # Student agent parameters
        student_name=args.agent_name,
        student_model_path=args.model_path,
        student_device=args.device,
        student_agent_options=args.agent_options,
        # Teacher agent parameters
        teacher_name=args.teacher_agent_name,
        teacher_model_path=args.teacher_model_path,
        teacher_device=args.teacher_device,
        teacher_agent_options=args.teacher_agent_options,
        # Shared parameters
        levels_quality=args.levels_quality,
        state_history_len=args.state_history_len,
        initial_level=args.initial_level,
        env_options=args.env_options,
        # Training parameters
        output_dir=args.output_dir,
        parallel_workers=args.parallel_workers,
        max_steps_per_epoch=args.max_steps_per_epoch,
        train_epochs=args.train_epochs,
        model_save_interval=args.model_save_interval,
        on_save_model=on_save_model,
    )

    # Start imitation learning
    print(f"Starting imitation learning with {args.parallel_workers} parallel workers...")
    print(f"Student agent: {args.agent_name}")
    print(f"Teacher agent: {args.teacher_agent_name}")
    print(f"Output directory: {args.output_dir}")
    trainer.train()

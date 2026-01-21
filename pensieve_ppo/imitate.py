"""Imitation learning script for Pensieve PPO.

This module implements the complete imitation learning pipeline using distributed
parallel agents, where worker agents use a teacher agent (e.g., LLM-based agent)
to collect trajectories, and a central student agent (neural network) learns to
imitate those decisions through behavioral cloning.

Reference:
    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/train.py
"""

import argparse


from .agent import get_available_agents
from .defaults import DEFAULT_QUALITY
from .args import add_env_agent_arguments, parse_env_agent_args, parse_options
from .test import add_testing_arguments
from .train import TestingCallback, add_training_arguments


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

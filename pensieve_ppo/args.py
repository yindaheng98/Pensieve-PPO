"""Argument parsing utilities for Pensieve PPO."""

import argparse
from typing import Any, Dict, List

import numpy as np

from .defaults import VIDEO_BIT_RATE, TEST_TRACES, DEFAULT_QUALITY, S_INFO, S_LEN

# Default random seed
RANDOM_SEED = 42


def add_env_agent_arguments(parser: argparse.ArgumentParser, available_agents: List[str]) -> None:
    """Add arguments for environment and agent configuration.

    These arguments correspond to the parameters of create_env_agent_with_default and
    create_env_agent_factory_with_default in pensieve_ppo/defaults.py. See those functions
    for parameter descriptions.

    Args:
        parser: ArgumentParser to add arguments to.
        available_agents: List of available agent names for validation.
    """
    parser.add_argument('--test-trace-folder', type=str, default=TEST_TRACES,
                        help=f"Folder containing network bandwidth trace files for testing "
                             f"(default: '{TEST_TRACES}')")
    parser.add_argument('--agent-name', type=str, default='ppo',
                        choices=available_agents,
                        help=f"Algorithm to use (default: 'ppo')")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to load pre-trained model weights (default: None)")
    parser.add_argument('--device', type=str, default=None,
                        help="PyTorch device for computation, e.g. 'cuda', 'cpu' (default: None, auto-select)")
    parser.add_argument('--levels-quality', type=float, nargs='+', default=VIDEO_BIT_RATE,
                        metavar='BITRATE',
                        help=f"Video bitrate levels in Kbps, determines action_dim=len(levels_quality) "
                             f"(default: {VIDEO_BIT_RATE})")
    parser.add_argument('--state-history-len', type=int, default=S_LEN,
                        help=f"Number of past observations in state, determines state_dim=({S_INFO}, state_history_len) "
                             f"(default: {S_LEN})")
    parser.add_argument('--initial-level', type=int, default=DEFAULT_QUALITY,
                        help=f"Initial quality level index on reset (default: {DEFAULT_QUALITY})")
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help=f"Global random seed for Numpy and PyTorch (default: {RANDOM_SEED})")
    parser.add_argument('-o', '--agent-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help="Extra agent kwargs, e.g. learning_rate=1e-4 gamma=0.99")
    parser.add_argument('-e', '--env-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help="Extra env kwargs, e.g. rebuf_penalty=4.3 smooth_penalty=1.0")


def parse_options(options: list) -> Dict[str, Any]:
    """Parse additional options from command line using eval."""
    result: Dict[str, Any] = {}
    for opt in options:
        if '=' not in opt:
            raise ValueError(f"Invalid option format: {opt}. Expected KEY=VALUE")
        key, value = opt.split('=', 1)
        result[key] = eval(value)
    return result


def parse_env_agent_args(args: argparse.Namespace) -> argparse.Namespace:
    """Post-process parsed arguments for environment and agent configuration.

    This function handles:
    - Parsing additional options (agent_options, env_options)
    - Setting global random seed if specified.

    Args:
        args: Parsed argument namespace from argparse.

    Returns:
        The modified argument namespace.
    """
    # Parse additional options
    args.agent_options = parse_options(args.agent_options)
    args.env_options = parse_options(args.env_options)

    # Set global random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    return args

"""Argument parsing utilities for Pensieve PPO."""

import argparse
import keyword
import os
import sys
from typing import Any, Dict, List

import numpy as np

from .defaults import (
    DEFAULT_REGISTRY_PACKAGES,
    REGISTRY_PACKAGES_ENV,
    TEST_TRACES,
    import_registry_packages,
)

# Default random seed
RANDOM_SEED = 42


def is_valid_import_name(name: str) -> bool:
    """Return whether name is a valid absolute or relative Python module path."""
    module_path = name.lstrip('.')
    return bool(module_path) and all(
        part.isidentifier() and not keyword.iskeyword(part)
        for part in module_path.split('.')
    )


def prepare_registry_package(parser: argparse.ArgumentParser) -> None:
    """Add registry package args and import packages requested on the command line."""
    parser.add_argument(
        '--registry-package',
        '--import-package',
        action='append',
        default=argparse.SUPPRESS,
        dest='registry_packages',
        metavar='PACKAGE',
        help=(
            'Package to import before building command arguments. '
            'Can be repeated. Defaults to .quality_ladder.'
        ),
    )
    preparse_args = [arg for arg in sys.argv[1:] if arg not in ('-h', '--help')]
    args, _ = parser.parse_known_args(preparse_args)

    registry_packages = getattr(args, 'registry_packages', DEFAULT_REGISTRY_PACKAGES)
    assert all(is_valid_import_name(package) for package in registry_packages), (
        f"Invalid registry package name(s): {registry_packages}"
    )
    os.environ[REGISTRY_PACKAGES_ENV] = os.pathsep.join(registry_packages)
    import_registry_packages(registry_packages)


def add_env_agent_arguments(parser: argparse.ArgumentParser, available_agents: List[str]) -> None:
    """Add arguments for environment and agent configuration.

    These arguments correspond to the parameters of create_env_agent and
    create_env_agent_factory in pensieve_ppo/defaults.py. See those functions
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
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help=f"Global random seed for Numpy and PyTorch (default: {RANDOM_SEED})")
    parser.add_argument('-o', '--agent-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help="Extra agent kwargs, e.g. learning_rate=1e-4 gamma=0.99 device='cuda'")
    parser.add_argument('--observer-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help="Extra observer kwargs, e.g. state_history_len=6 rebuf_penalty=4.3")
    parser.add_argument('--player-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help="Extra video player kwargs, e.g. name='envivio' max_chunks=48")


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
    - Parsing additional options (agent_options, observer_options, player_options)
    - Setting global random seed if specified.

    Args:
        args: Parsed argument namespace from argparse.

    Returns:
        The modified argument namespace.
    """
    # Parse additional options
    args.agent_options = parse_options(args.agent_options)
    args.observer_options = parse_options(args.observer_options)
    args.player_options = parse_options(args.player_options)

    # Set global random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    return args

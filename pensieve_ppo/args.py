"""Argument parsing utilities for Pensieve PPO."""

import argparse
from typing import Any, Dict, Optional

import numpy as np
import torch

from .defaults import VIDEO_BIT_RATE
from .gym.env import S_INFO, S_LEN

# Default random seed (matching common.py RANDOM_SEED)
RANDOM_SEED = 42


class SetSeedAction(argparse.Action):
    """Argparse action that sets global random seed on parse."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            np.random.seed(values)
        setattr(namespace, self.dest, values)


def add_env_agent_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for create_env_agent_with_default and create_env_agent_factory_with_default.

    These arguments correspond to the parameters of create_env_agent_with_default and
    create_env_agent_factory_with_default in pensieve_ppo/defaults.py. See those functions
    for parameter descriptions.
    """
    parser.add_argument('--trace-folder', type=str, default=None,
                        help='Path to trace folder (default: None, auto-selects based on train mode)')
    parser.add_argument('--train', action='store_true',
                        help='Use training mode (default: True for factory, False for single env/agent)')
    parser.add_argument('--agent-name', type=str, default='ppo',
                        help="Agent name (default: 'ppo')")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model (default: None)')
    parser.add_argument('--device', type=str, default=None,
                        help='PyTorch device (e.g., "cuda", "cpu") (default: None)')
    parser.add_argument('--levels-quality', type=float, nargs='+', default=None,
                        metavar='BITRATE',
                        help=f'Video bitrate levels in Kbps (default: {VIDEO_BIT_RATE})')
    parser.add_argument('--state-history-len', type=int, default=S_LEN,
                        help=f'State history length (S_LEN), S_INFO is fixed at {S_INFO} (default: {S_LEN})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, action=SetSeedAction,
                        help=f'Global random seed (default: {RANDOM_SEED})')
    parser.add_argument('-o', '--agent-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Additional agent options as key=value pairs')
    parser.add_argument('-e', '--env-options', type=str, nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Additional environment options as key=value pairs')

    # Apply default seed immediately so callers remain transparent
    np.random.seed(RANDOM_SEED)


def parse_options(options: list) -> Dict[str, Any]:
    """Parse additional options from command line using eval."""
    result: Dict[str, Any] = {}
    for opt in options:
        if '=' not in opt:
            raise ValueError(f"Invalid option format: {opt}. Expected KEY=VALUE")
        key, value = opt.split('=', 1)
        result[key] = eval(value)
    return result

"""Argument parsing utilities for Pensieve PPO."""

import argparse
from typing import Any, Dict

import numpy as np

from .agent import get_available_agents
from .defaults import VIDEO_BIT_RATE, TRAIN_TRACES, TEST_TRACES
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
                        help=f"Folder containing network bandwidth trace files for simulation "
                             f"(default: '{TRAIN_TRACES}' for train, '{TEST_TRACES}' for test)")
    parser.add_argument('--agent-name', type=str, default='ppo',
                        choices=get_available_agents(),
                        help="RL algorithm to use (default: 'ppo')")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to load pre-trained model weights (default: None)")
    parser.add_argument('--device', type=str, default=None,
                        help="PyTorch device for computation, e.g. 'cuda', 'cpu' (default: None, auto-select)")
    parser.add_argument('--levels-quality', type=float, nargs='+', default=None,
                        metavar='BITRATE',
                        help=f"Video bitrate levels in Kbps, determines action_dim=len(levels_quality) "
                             f"(default: {VIDEO_BIT_RATE})")
    parser.add_argument('--state-history-len', type=int, default=S_LEN,
                        help=f"Number of past observations in state, determines state_dim=({S_INFO}, state_history_len) "
                             f"(default: {S_LEN})")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, action=SetSeedAction,
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

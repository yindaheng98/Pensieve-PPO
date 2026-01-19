"""Default parameter combinations for ABR environment and agent.

This module provides default parameter values and convenience functions
for creating ABREnv and Agent instances with Pensieve-PPO defaults.

The key compatibility parameters between env and agent are:
- state_dim[0] (S_INFO=6): Fixed in ABREnv, number of state features
- state_dim[1] (state_history_len): Must match env.state_history_len
- action_dim: Must equal len(levels_quality), which determines env.bitrate_levels
"""

from typing import Callable, Optional, Tuple

import torch

from .agent import AbstractAgent, AbstractTrainableAgent, create_agent, create_env
from .agent.rl.observer import S_INFO, S_LEN
from .gym import ABREnv


# Default constants from original Pensieve-PPO implementation
# Source: https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/

# From src/core.py
TOTAL_VIDEO_CHUNKS = 48  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L9
VIDEO_SIZE_FILE_PREFIX = './src/envivio/video_size_'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L17

# From src/env.py
VIDEO_BIT_RATE = [300., 750., 1200., 1850., 2850., 4300.]  # Kbps, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L13
DEFAULT_QUALITY = 1  # default video quality without agent, https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L19

# From src/load_trace.py and src/test.py
TRAIN_TRACES = './src/train/'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/load_trace.py#L4
TEST_TRACES = './src/test/'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L26


def create_env_with_default(
    name: str = 'ppo',
    levels_quality: list = VIDEO_BIT_RATE,
    trace_folder: Optional[str] = None,
    video_size_file_prefix: str = VIDEO_SIZE_FILE_PREFIX,
    max_chunks: int = TOTAL_VIDEO_CHUNKS,
    train: bool = True,
    # Env parameters
    initial_level: int = DEFAULT_QUALITY,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv with default Pensieve parameters.

    Wraps `create_env` with default values matching the original Pensieve implementation.
    If trace_folder is None, auto-selects TRAIN_TRACES or TEST_TRACES based on train flag.
    """
    if trace_folder is None:
        trace_folder = TRAIN_TRACES if train else TEST_TRACES

    return create_env(
        name=name,
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        video_size_file_prefix=video_size_file_prefix,
        max_chunks=max_chunks,
        train=train,
        initial_level=initial_level,
        **kwargs,
    )


class PicklableEnvFactory:
    """Callable factory for creating ABREnv instances."""

    def __init__(self, *args, env_options: dict = {}, **kwargs):
        self.args = args
        self.env_options = env_options
        self.kwargs = kwargs

    def __call__(self, pid: int) -> ABREnv:
        env_options = self.env_options.copy()
        random_seed = env_options['random_seed'] + pid if 'random_seed' in env_options and env_options['random_seed'] is not None else None
        env_options = {**env_options, 'random_seed': random_seed}
        return create_env_with_default(name=self.name, *self.args, **self.kwargs, **env_options)


class PicklableAgentFactory:
    """Callable factory for creating AbstractTrainableAgent instances (for training)."""

    def __init__(self, *args, agent_options: dict = {}, **kwargs):
        self.args = args
        self.agent_options = agent_options
        self.kwargs = kwargs

    def __call__(self) -> AbstractTrainableAgent:
        return create_agent(*self.args, **self.kwargs, **self.agent_options)  # type: ignore[return-value]


def create_env_agent_factory_with_default(
    trace_folder: Optional[str] = None,
    train: bool = True,
    model_path: Optional[str] = None,
    name: str = 'ppo',
    device: Optional[torch.device] = None,
    # Compatibility parameters (shared between env and agent)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
    # Additional options
    env_options: dict = {},
    agent_options: dict = {},
) -> Tuple[Callable[[int], ABREnv], Callable[[], AbstractTrainableAgent]]:
    """Create env_factory and agent_factory with default parameters.

    Ensures env-agent compatibility by deriving shared parameters:
    - state_dim = (S_INFO, state_history_len), action_dim = len(levels_quality)

    Returns:
        (env_factory, agent_factory):
        - env_factory(agent_id: int) -> ABREnv
        - agent_factory() -> AbstractTrainableAgent
    """
    # Derive compatibility parameters
    state_dim = (S_INFO, state_history_len)
    action_dim = len(levels_quality)

    # Create environment factory
    env_factory = PicklableEnvFactory(
        name=name,
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        train=train,
        state_history_len=state_history_len,
        env_options=env_options,
    )

    # Create agent factory
    agent_factory = PicklableAgentFactory(
        name=name,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        model_path=model_path,
        agent_options=agent_options,
    )

    return env_factory, agent_factory


def create_env_agent_with_default(
    *args,
    train: bool = False,
    **kwargs,
) -> Tuple[ABREnv, AbstractAgent]:
    """Create a compatible env and agent pair with default parameters.

    Ensures env-agent compatibility by deriving shared parameters:
    - state_dim = (S_INFO, state_history_len), action_dim = len(levels_quality)

    Returns:
        (env, agent) - guaranteed to be compatible.
    """
    env_factory, agent_factory = create_env_agent_factory_with_default(*args, train=train, **kwargs)
    return env_factory(pid=0), agent_factory()

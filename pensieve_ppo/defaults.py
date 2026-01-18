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

from .agent import AbstractAgent, AbstractTrainableAgent, create_agent
from .gym import ABREnv, create_env
from .agent.rl.observer import (
    RLABRStateObserver,
    S_INFO,
    S_LEN,
    REBUF_PENALTY,
    SMOOTH_PENALTY,
    BUFFER_NORM_FACTOR,
)


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
    levels_quality: list = VIDEO_BIT_RATE,
    trace_folder: Optional[str] = None,
    video_size_file_prefix: str = VIDEO_SIZE_FILE_PREFIX,
    max_chunks: int = TOTAL_VIDEO_CHUNKS,
    train: bool = True,
    # Observer parameters
    rebuf_penalty: float = REBUF_PENALTY,
    smooth_penalty: float = SMOOTH_PENALTY,
    state_history_len: int = S_LEN,
    buffer_norm_factor: float = BUFFER_NORM_FACTOR,
    # Env parameters
    initial_level: int = 0,
    **kwargs,
) -> ABREnv:
    """Create an ABREnv with default Pensieve parameters.

    Wraps `create_env` with default values matching the original Pensieve implementation.
    If trace_folder is None, auto-selects TRAIN_TRACES or TEST_TRACES based on train flag.
    """
    if trace_folder is None:
        trace_folder = TRAIN_TRACES if train else TEST_TRACES

    # Create the RL observer with specified parameters
    observer = RLABRStateObserver(
        levels_quality=levels_quality,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
        state_history_len=state_history_len,
        buffer_norm_factor=buffer_norm_factor,
    )

    return create_env(
        observer=observer,
        initial_level=initial_level,
        trace_folder=trace_folder,
        video_size_file_prefix=video_size_file_prefix,
        max_chunks=max_chunks,
        train=train,
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
        return create_env_with_default(*self.args, **self.kwargs, **env_options)


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
    agent_name: str = 'ppo',
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
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        train=train,
        state_history_len=state_history_len,
        env_options=env_options,
    )

    # Create agent factory
    agent_factory = PicklableAgentFactory(
        name=agent_name,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        model_path=model_path,
        agent_options=agent_options,
    )

    return env_factory, agent_factory


def create_env_agent_with_default(
    trace_folder: Optional[str] = None,
    train: bool = False,
    model_path: Optional[str] = None,
    agent_name: str = 'ppo',
    device: Optional[torch.device] = None,
    # Compatibility parameters (shared between env and agent)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
    # Additional options
    env_options: dict = {},
    agent_options: dict = {},
) -> Tuple[ABREnv, AbstractAgent]:
    """Create a compatible env and agent pair with default parameters.

    Ensures env-agent compatibility by deriving shared parameters:
    - state_dim = (S_INFO, state_history_len), action_dim = len(levels_quality)

    Returns:
        (env, agent) - guaranteed to be compatible.
    """
    # Derive compatibility parameters
    # state_dim[0] is S_INFO (fixed at 6 in ABREnv)
    # state_dim[1] is state_history_len
    # action_dim is len(levels_quality)
    state_dim = (S_INFO, state_history_len)
    action_dim = len(levels_quality)

    # Create environment with compatibility parameters
    env = create_env_with_default(
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        train=train,
        state_history_len=state_history_len,
        **env_options,
    )

    # Create agent with derived compatibility parameters
    agent = create_agent(
        name=agent_name,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        model_path=model_path,
        **agent_options,
    )

    return env, agent

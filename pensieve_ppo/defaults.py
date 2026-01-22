"""Default parameter combinations for ABR environment and agent.

This module provides default parameter values and convenience functions
for creating ABREnv and Agent instances with Pensieve-PPO defaults.

The key compatibility parameters between env and agent are:
- state_dim[0] (S_INFO=6): Fixed in ABREnv, number of state features
- state_dim[1] (state_history_len): Must match env.state_history_len
- action_dim: Must equal len(levels_quality), which determines env.bitrate_levels
"""

from typing import Callable, Optional, Tuple, Union

import torch

from .agent import (
    AbstractAgent,
    AbstractTrainableAgent,
    create_agent,
    create_env,
    create_imitation_env,
)
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

    Wraps `create_env` with default values matching the original
    Pensieve implementation. If trace_folder is None, auto-selects TRAIN_TRACES
    or TEST_TRACES based on train flag.

    Returns:
        Configured ABREnv instance.
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

    def __init__(self, *args, random_seed=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.random_seed = random_seed

    def __call__(self, pid: int) -> ABREnv:
        random_seed = (self.random_seed + pid) if self.random_seed is not None else None
        return create_env_with_default(*self.args, random_seed=random_seed, **self.kwargs)


class PicklableAgentFactory:
    """Callable factory for creating AbstractAgent or AbstractTrainableAgent instances."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Union[AbstractAgent, AbstractTrainableAgent]:
        return create_agent(*self.args, **self.kwargs)


def create_agent_factory_with_default(
    model_path: Optional[str] = None,
    name: str = 'ppo',
    device: Optional[Union[torch.device, str]] = None,
    # Compatibility parameters (shared between env and agent)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
    # Additional options
    agent_options: dict = {},
) -> Callable[[], Union[AbstractAgent, AbstractTrainableAgent]]:
    """Create agent_factory with default parameters.

    This function only creates an agent factory, without creating an environment factory.
    Useful when you need to create an agent factory separately (e.g., for imitation learning).

    Ensures compatibility by deriving shared parameters:
    - state_dim = (S_INFO, state_history_len), action_dim = len(levels_quality)

    Returns:
        agent_factory() -> AbstractAgent | AbstractTrainableAgent
    """
    # Derive compatibility parameters
    state_dim = (S_INFO, state_history_len)
    action_dim = len(levels_quality)

    # Create agent factory
    agent_factory = PicklableAgentFactory(
        name=name,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        model_path=model_path,
        **agent_options,
    )

    return agent_factory


def create_env_agent_factory_with_default(
    trace_folder: Optional[str] = None,
    train: bool = True,
    # Agent parameters
    model_path: Optional[str] = None,
    name: str = 'ppo',
    device: Optional[Union[torch.device, str]] = None,
    # Compatibility parameters (shared between env and agent)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
    # Additional options
    env_options: dict = {},
    agent_options: dict = {},
) -> Tuple[PicklableEnvFactory, PicklableAgentFactory]:
    """Create env_factory and agent_factory with default parameters.

    Ensures env-agent compatibility by deriving shared parameters:
    - state_dim = (S_INFO, state_history_len), action_dim = len(levels_quality)

    Returns:
        (env_factory, agent_factory):
        - env_factory(agent_id: int) -> ABREnv
        - agent_factory() -> AbstractTrainableAgent
    """
    # Create environment factory
    env_factory = PicklableEnvFactory(
        name=name,
        levels_quality=levels_quality,
        trace_folder=trace_folder,
        train=train,
        state_history_len=state_history_len,
        **env_options,
    )

    # Create agent factory using create_agent_factory_with_default
    agent_factory = create_agent_factory_with_default(
        model_path=model_path,
        name=name,
        device=device,
        levels_quality=levels_quality,
        state_history_len=state_history_len,
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


class PicklableImitationEnvFactory:
    """Callable factory for creating ABREnv instances with ImitationObserver.

    This factory creates environments that output ImitationState, containing
    both student_state and teacher_state for imitation learning.
    """

    def __init__(
        self,
        student_name: str = 'ppo',
        teacher_name: str = 'bba',
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the imitation environment factory.

        Args:
            student_name: Agent name for student observer (e.g., 'ppo').
            teacher_name: Agent name for teacher observer (e.g., 'bba', 'mpc').
            random_seed: Base random seed (will be offset by pid).
            **kwargs: Additional kwargs passed to create_imitation_env.
        """
        self.student_name = student_name
        self.teacher_name = teacher_name
        self.random_seed = random_seed
        self.kwargs = kwargs

    def __call__(self, pid: int) -> ABREnv:
        """Create an ABREnv with ImitationObserver.

        Args:
            pid: Process ID for random seed offset.

        Returns:
            Configured ABREnv instance with ImitationObserver.
        """
        random_seed = (self.random_seed + pid) if self.random_seed is not None else None

        return create_imitation_env(
            student_name=self.student_name,
            teacher_name=self.teacher_name,
            random_seed=random_seed,
            **self.kwargs,
        )


def create_imitation_env_agent_factory_with_default(
    trace_folder: Optional[str] = None,
    train: bool = True,
    # Student agent parameters
    student_name: str = 'ppo',
    student_model_path: Optional[str] = None,
    student_device: Optional[Union[torch.device, str]] = None,
    student_agent_options: dict = {},
    # Teacher agent parameters
    teacher_name: str = 'bba',
    teacher_model_path: Optional[str] = None,
    teacher_device: Optional[Union[torch.device, str]] = None,
    teacher_agent_options: dict = {},
    # Compatibility parameters (shared between student and teacher)
    levels_quality: list = VIDEO_BIT_RATE,
    state_history_len: int = S_LEN,
    # Additional options
    random_seed: Optional[int] = None,
    initial_level: int = DEFAULT_QUALITY,
    env_options: dict = {},
) -> Tuple[PicklableImitationEnvFactory, PicklableAgentFactory, PicklableAgentFactory]:
    """Create env_factory, student_agent_factory, and teacher_agent_factory for imitation learning.

    This function creates factories for imitation learning where:
    - env_factory creates environments with ImitationObserver (student + teacher states)
    - student_agent_factory creates the student agent (neural network to train)
    - teacher_agent_factory creates the teacher agent (expert to imitate)

    Ensures compatibility by deriving shared parameters:
    - state_dim = (S_INFO, state_history_len), action_dim = len(levels_quality)

    Args:
        trace_folder: Folder containing network bandwidth trace files.
        train: Whether in training mode (affects trace iteration).
        student_name: Agent name for student (e.g., 'ppo').
        student_model_path: Path to pre-trained model for student agent.
        student_device: PyTorch device for student agent.
        student_agent_options: Additional kwargs for student agent.
        teacher_name: Agent name for teacher (e.g., 'bba', 'mpc').
        teacher_model_path: Path to pre-trained model for teacher agent.
        teacher_device: PyTorch device for teacher agent.
        teacher_agent_options: Additional kwargs for teacher agent.
        levels_quality: Quality metric list for each bitrate level.
        state_history_len: Number of past observations in state.
        random_seed: Base random seed (will be offset by pid for env).
        initial_level: Initial quality level index on reset.
        env_options: Additional kwargs for environment/simulator.

    Returns:
        (env_factory, student_agent_factory, teacher_agent_factory):
        - env_factory(pid: int) -> ABREnv with ImitationObserver
        - student_agent_factory() -> AbstractTrainableAgent (student)
        - teacher_agent_factory() -> AbstractAgent (teacher)
    """
    # Auto-select trace folder based on train flag
    if trace_folder is None:
        trace_folder = TRAIN_TRACES if train else TEST_TRACES

    # Create imitation environment factory
    env_factory = PicklableImitationEnvFactory(
        student_name=student_name,
        teacher_name=teacher_name,
        random_seed=random_seed,
        # Observer args (shared)
        levels_quality=levels_quality,
        state_history_len=state_history_len,
        # Simulator args
        trace_folder=trace_folder,
        train=train,
        initial_level=initial_level,
        **env_options,
    )

    # Create student agent factory
    student_agent_factory = create_agent_factory_with_default(
        model_path=student_model_path,
        name=student_name,
        device=student_device,
        levels_quality=levels_quality,
        state_history_len=state_history_len,
        agent_options=student_agent_options,
    )

    # Create teacher agent factory
    teacher_agent_factory = create_agent_factory_with_default(
        model_path=teacher_model_path,
        name=teacher_name,
        device=teacher_device,
        levels_quality=levels_quality,
        state_history_len=state_history_len,
        agent_options=teacher_agent_options,
    )

    return env_factory, student_agent_factory, teacher_agent_factory

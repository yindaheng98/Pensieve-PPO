"""Default parameter combinations for ABR environment and agent.

This module provides default parameter values and convenience functions
for creating ABREnv and Agent instances with Pensieve-PPO defaults.

"""

from typing import Optional, Tuple, Union

from .agent import (
    AbstractAgent,
    AbstractTrainableAgent,
    create_agent,
    create_env,
    create_imitation_env,
)
from .gym import ABREnv


# From src/load_trace.py and src/test.py
TRAIN_TRACES = './src/train/'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/load_trace.py#L4
TEST_TRACES = './src/test/'  # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L26


class PicklableEnvFactory:
    """Callable factory for creating ABREnv instances."""

    def __init__(
        self,
        name: str,
        trace_folder: Optional[str] = None,
        train: bool = True,
        random_seed: Optional[int] = None,
        observer_kwargs: dict = {},
        player_kwargs: dict = {},
    ):
        if trace_folder is None:
            trace_folder = TRAIN_TRACES if train else TEST_TRACES

        self.name = name
        self.trace_folder = trace_folder
        self.train = train
        self.random_seed = random_seed
        self.observer_kwargs = dict(observer_kwargs)
        self.player_kwargs = dict(player_kwargs)

    def __call__(self, pid: int) -> ABREnv:
        random_seed = (self.random_seed + pid) if self.random_seed is not None else None
        return create_env(
            name=self.name,
            trace_folder=self.trace_folder,
            train=self.train,
            random_seed=random_seed,
            observer_kwargs=self.observer_kwargs,
            player_kwargs=self.player_kwargs,
        )


class PicklableAgentFactory:
    """Callable factory for creating AbstractAgent or AbstractTrainableAgent instances."""

    def __init__(
        self,
        name: str,
        model_path: Optional[str] = None,
        agent_kwargs: dict = {},
    ):
        self.name = name
        self.model_path = model_path
        self.agent_kwargs = dict(agent_kwargs)

    def __call__(self) -> Union[AbstractAgent, AbstractTrainableAgent]:
        return create_agent(
            name=self.name,
            model_path=self.model_path,
            agent_kwargs=self.agent_kwargs,
        )


def create_env_agent_factory(
    name: str = 'ppo',
    trace_folder: Optional[str] = None,
    train: bool = True,
    random_seed: Optional[int] = None,
    observer_kwargs: dict = {},
    player_kwargs: dict = {},
    model_path: Optional[str] = None,
    agent_kwargs: dict = {},
) -> Tuple[PicklableEnvFactory, PicklableAgentFactory]:
    """Create compatible environment and agent factories.

    Returns:
        (env_factory, agent_factory):
        - env_factory(agent_id: int) -> ABREnv
        - agent_factory() -> AbstractTrainableAgent
    """
    env_factory = PicklableEnvFactory(
        name=name,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
        observer_kwargs=observer_kwargs,
        player_kwargs=player_kwargs,
    )

    agent_factory = PicklableAgentFactory(
        name=name,
        model_path=model_path,
        agent_kwargs=agent_kwargs,
    )

    return env_factory, agent_factory


def create_env_agent(
    name: str = 'ppo',
    trace_folder: Optional[str] = None,
    train: bool = True,
    random_seed: Optional[int] = None,
    observer_kwargs: dict = {},
    player_kwargs: dict = {},
    model_path: Optional[str] = None,
    agent_kwargs: dict = {},
) -> Tuple[ABREnv, AbstractAgent]:
    """Create a compatible env and agent pair."""
    env_factory, agent_factory = create_env_agent_factory(
        name=name,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
        observer_kwargs=observer_kwargs,
        player_kwargs=player_kwargs,
        model_path=model_path,
        agent_kwargs=agent_kwargs,
    )
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
        trace_folder: Optional[str] = None,
        train: bool = True,
        random_seed: Optional[int] = None,
        student_observer_kwargs: dict = {},
        teacher_observer_kwargs: dict = {},
        player_kwargs: dict = {},
    ):
        if trace_folder is None:
            trace_folder = TRAIN_TRACES if train else TEST_TRACES

        self.student_name = student_name
        self.teacher_name = teacher_name
        self.trace_folder = trace_folder
        self.train = train
        self.random_seed = random_seed
        self.student_observer_kwargs = dict(student_observer_kwargs)
        self.teacher_observer_kwargs = dict(teacher_observer_kwargs)
        self.player_kwargs = dict(player_kwargs)

    def __call__(self, pid: int) -> ABREnv:
        random_seed = (self.random_seed + pid) if self.random_seed is not None else None
        return create_imitation_env(
            student_name=self.student_name,
            teacher_name=self.teacher_name,
            trace_folder=self.trace_folder,
            train=self.train,
            random_seed=random_seed,
            student_observer_kwargs=self.student_observer_kwargs,
            teacher_observer_kwargs=self.teacher_observer_kwargs,
            player_kwargs=self.player_kwargs,
        )


def create_imitation_env_agent_factory(
    student_name: str = 'ppo',
    teacher_name: str = 'bba',
    trace_folder: Optional[str] = None,
    train: bool = True,
    random_seed: Optional[int] = None,
    student_observer_kwargs: dict = {},
    teacher_observer_kwargs: dict = {},
    player_kwargs: dict = {},
    student_model_path: Optional[str] = None,
    teacher_model_path: Optional[str] = None,
    student_agent_kwargs: dict = {},
    teacher_agent_kwargs: dict = {},
) -> Tuple[PicklableImitationEnvFactory, PicklableAgentFactory, PicklableAgentFactory]:
    """Create env_factory, student_agent_factory, and teacher_agent_factory for imitation learning.

    This function creates factories for imitation learning where:
    - env_factory creates environments with ImitationObserver (student + teacher states)
    - student_agent_factory creates the student agent (neural network to train)
    - teacher_agent_factory creates the teacher agent (expert to imitate)

    Args:
        trace_folder: Folder containing network bandwidth trace files.
        train: Whether in training mode (affects trace iteration).
        student_name: Agent name for student (e.g., 'ppo').
        student_model_path: Path to pre-trained model for student agent.
        teacher_name: Agent name for teacher (e.g., 'bba', 'mpc').
        teacher_model_path: Path to pre-trained model for teacher agent.
        student_agent_kwargs: Additional kwargs for student agent.
        teacher_agent_kwargs: Additional kwargs for teacher agent.
        random_seed: Base random seed (will be offset by pid for env).

    Returns:
        (env_factory, student_agent_factory, teacher_agent_factory):
        - env_factory(pid: int) -> ABREnv with ImitationObserver
        - student_agent_factory() -> AbstractTrainableAgent (student)
        - teacher_agent_factory() -> AbstractAgent (teacher)
    """
    env_factory = PicklableImitationEnvFactory(
        student_name=student_name,
        teacher_name=teacher_name,
        trace_folder=trace_folder,
        train=train,
        random_seed=random_seed,
        student_observer_kwargs=student_observer_kwargs,
        teacher_observer_kwargs=teacher_observer_kwargs,
        player_kwargs=player_kwargs,
    )

    student_agent_factory = PicklableAgentFactory(
        name=student_name,
        model_path=student_model_path,
        agent_kwargs=student_agent_kwargs,
    )

    teacher_agent_factory = PicklableAgentFactory(
        name=teacher_name,
        model_path=teacher_model_path,
        agent_kwargs=teacher_agent_kwargs,
    )

    return env_factory, student_agent_factory, teacher_agent_factory

"""Imitation Learning Observer Components.

This module provides state and observer classes for imitation learning,
enabling any combination of student and teacher observers through composition
rather than inheritance.

The key insight is that imitation learning requires:
1. A student observer: provides state for training and computes rewards
2. A teacher observer: provides state for teacher's decision making

By composing these observers, we can achieve any imitation combination:
- RL student imitating BBA teacher
- RL student imitating MPC teacher
- etc.

Example:
    >>> # Create individual observers
    >>> rl_observer = RLABRStateObserver(levels_quality=VIDEO_BIT_RATE)
    >>> bba_observer = BBAStateObserver(levels_quality=VIDEO_BIT_RATE)
    >>>
    >>> # Combine for imitation learning
    >>> imitation_observer = ImitationObserver(
    ...     student_observer=rl_observer,
    ...     teacher_observer=bba_observer,
    ... )
    >>>
    >>> # Use in environment
    >>> env = ABREnv(simulator=simulator, observer=imitation_observer)
    >>> state, info = env.reset()
    >>> # state.student_state is RLState (for training RL agent)
    >>> # state.teacher_state is BBAState (for BBA agent's decision)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from gymnasium import spaces

from .env import AbstractABRStateObserver, ABREnv, State
from ..core.simulator import StepResult
from ..core.video import VideoChunkRequestType, VideoPlayer


@dataclass
class ImitationState(State):
    """State class for imitation learning.

    This dataclass combines states from two different observers, enabling
    imitation learning where a student agent learns from a teacher agent's
    decisions.

    The student_state is used for training (e.g., updating RL policy),
    while teacher_state is used by the teacher agent to select actions.

    Attributes:
        student_state: State for the student agent (e.g., RLState for RL training).
        teacher_state: State for the teacher agent (e.g., BBAState for BBA decisions).
    """
    student_state: State
    teacher_state: State


class ImitationObserver(AbstractABRStateObserver[VideoChunkRequestType]):
    """Observer that combines student and teacher observers for imitation learning.

    This observer enables imitation learning by composing two different observers:
    - student_observer: Provides state for training and computes rewards
    - teacher_observer: Provides state for teacher's decision making

    The observation_space is taken from the student_observer since that's
    what will be used for training. The reward is also computed by the
    student_observer.

    This design allows any observer to be used as either student or teacher,
    without requiring observers to inherit from each other.

    Attributes:
        student_observer: Observer for the student agent.
        teacher_observer: Observer for the teacher agent.
    """

    @classmethod
    def get_constructor_args(cls) -> List[str]:
        """Get the list of all constructor argument names.

        Raises:
            NotImplementedError: ImitationObserver should not be constructed via registry.
        """
        raise NotImplementedError(
            "ImitationObserver should not be constructed via registry. "
            "Use direct instantiation with student_observer and teacher_observer."
        )

    def __init__(
        self,
        student_observer: AbstractABRStateObserver[VideoChunkRequestType],
        teacher_observer: AbstractABRStateObserver[VideoChunkRequestType],
    ):
        """Initialize the imitation observer.

        Args:
            student_observer: Observer for the student agent. Its observation_space
                            will be used as the environment's observation space.
            teacher_observer: Observer for the teacher agent. Used only to generate
                           states for teacher's decision making.
        """
        teacher_observer.validate_request_cls_match(student_observer)
        self.student_observer = student_observer
        self.teacher_observer = teacher_observer

    def validate_video_player(
        self,
        video_player: VideoPlayer[VideoChunkRequestType],
    ) -> None:
        """Validate that both composed observers support the video player's request type."""
        self.student_observer.validate_video_player(video_player)
        self.teacher_observer.validate_video_player(video_player)

    @property
    def observation_space(self) -> spaces.Box:
        """Gymnasium observation space for the state.

        Returns the student's observation space since that's what will be
        used for training.

        Returns:
            Gymnasium Box space from the student observer.
        """
        return self.student_observer.observation_space

    def reset(
        self,
        env: ABREnv[VideoChunkRequestType],
        initial_chunk_request: VideoChunkRequestType,
    ) -> Tuple[ImitationState, Dict[str, Any]]:
        """Reset both observers and return combined initial state.

        Args:
            env: The ABREnv instance to observe.
            initial_chunk_request: Initial video chunk request.

        Returns:
            Tuple of (ImitationState containing both states, combined info_dict).
        """
        student_state, student_info = self.student_observer.reset(env, initial_chunk_request)
        teacher_state, teacher_info = self.teacher_observer.reset(env, initial_chunk_request)

        state = ImitationState(
            student_state=student_state,
            teacher_state=teacher_state,
        )

        info = {
            'student_info': student_info,
            'teacher_info': teacher_info,
        }

        return state, info

    def observe(
        self,
        env: ABREnv[VideoChunkRequestType],
        chunk_request: VideoChunkRequestType,
        result: StepResult,
    ) -> Tuple[ImitationState, float, Dict[str, Any]]:
        """Process simulator result using both observers.

        The reward is computed by the student_observer since that's what
        will be used for training. Both observers update their states.

        Args:
            env: The ABREnv instance to observe.
            chunk_request: Current video chunk request.
            result: Result from simulator.step().

        Returns:
            Tuple of (ImitationState, reward from student, combined info_dict).
        """
        # Student observer provides reward for training
        student_state, reward, student_info = self.student_observer.observe(
            env, chunk_request, result
        )

        # Teacher observer provides state for teacher's decision (reward ignored)
        teacher_state, _, teacher_info = self.teacher_observer.observe(
            env, chunk_request, result
        )

        state = ImitationState(
            student_state=student_state,
            teacher_state=teacher_state,
        )

        info = {
            'student_info': student_info,
            'teacher_info': teacher_info,
        }

        return state, reward, info

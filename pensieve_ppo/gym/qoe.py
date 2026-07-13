"""QoE-based ABR state observer primitives.

This module contains reward/QoE calculation that depends only on generic
simulator outputs plus a concrete quality value supplied by subclasses.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..core.simulator import StepResult
from ..core.video import VideoChunkRequest
from .env import ABREnv, AbstractABRStateObserver, State


# Normalization constants
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
BUFFER_NORM_FACTOR = 10.0

# Reward parameters
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L17
REBUF_PENALTY = 4.3  # 1 sec rebuffering penalty
SMOOTH_PENALTY = 1.0  # penalty for bitrate changes


@dataclass
class QoEState(State):
    """Generic QoE state produced from simulator results.

    Subclasses can inherit from this state and add algorithm-specific fields
    while reusing the QoE inputs.
    """
    quality: float
    last_quality: float
    buffer_size_norm: float
    qoe: float


class QoEObserver(AbstractABRStateObserver):
    """Observer that computes QoE reward from generic simulator outputs.

    Concrete subclasses provide the quality value for the downloaded chunk.
    This class intentionally does not depend on any video-player-specific
    representation such as quality ladders.
    """

    def __init__(
        self,
        rebuf_penalty: float = REBUF_PENALTY,
        smooth_penalty: float = SMOOTH_PENALTY,
        buffer_norm_factor: float = BUFFER_NORM_FACTOR,
    ):
        """Initialize QoE calculation parameters."""
        self.rebuf_penalty = rebuf_penalty
        self.smooth_penalty = smooth_penalty
        self.buffer_norm_factor = buffer_norm_factor
        self.last_quality: Optional[float] = None

    def reset(
        self,
        env: ABREnv,
    ) -> None:
        """Reset QoE history for a new episode."""
        self.last_quality = None

    @abstractmethod
    def compute_quality(
        self,
        env: ABREnv,
        chunk_request: VideoChunkRequest,
        result: StepResult,
    ) -> float:
        """Compute the concrete quality value for a simulator result."""
        pass

    def observe(
        self,
        env: ABREnv,
        chunk_request: VideoChunkRequest,
        result: StepResult,
    ) -> Tuple[QoEState, float, Dict[str, Any]]:
        """Process simulator result and compute QoE reward."""
        quality = float(self.compute_quality(env, chunk_request, result))
        last_quality = self.last_quality if self.last_quality is not None else quality

        # Unpack result (matches original variable names)
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L75-L78
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L48-L51
        rebuf = result.rebuffer

        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L83-L87
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/test.py#L80-L84
        # reward is video quality - rebuffer penalty - smooth penalty
        qoe = (
            quality
            - self.rebuf_penalty * rebuf
            - self.smooth_penalty * abs(quality - last_quality)
        )

        state = QoEState(
            quality=quality,
            last_quality=last_quality,
            buffer_size_norm=result.buffer_size / self.buffer_norm_factor,
            qoe=qoe,
        )

        self.last_quality = quality
        info = {
            'quality': float(state.quality),
        }

        return state, qoe, info

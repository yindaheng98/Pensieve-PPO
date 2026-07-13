"""Quality-ladder QoE observer primitives."""

from typing import Any, Dict, Tuple

from ..core.simulator import StepResult
from ..gym.env import ABREnv
from ..gym.qoe import QoEObserver, QoEState
from .abc import QualityLadderRequest
from .player import QualityLadderResolvedChunk


# Quality-ladder quality unit normalization
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L14
M_IN_K = 1000.0


class QualityLadderQoEObserver(QoEObserver):
    """QoE observer for quality-ladder video chunks."""

    def compute_quality(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
    ) -> float:
        """Compute the quality-ladder quality value in QoE reward units."""
        resolved_chunk = result.resolved_chunk
        if not isinstance(resolved_chunk, QualityLadderResolvedChunk):
            raise TypeError(
                "Quality-ladder QoE observers require QualityLadderResolvedChunk, "
                f"got {type(resolved_chunk).__name__}"
            )
        return resolved_chunk.quality / M_IN_K

    def observe(
        self,
        env: ABREnv,
        chunk_request: QualityLadderRequest,
        result: StepResult,
    ) -> Tuple[QoEState, float, Dict[str, Any]]:
        """Process quality-ladder simulator result and compute QoE reward."""
        return super().observe(env, chunk_request, result)

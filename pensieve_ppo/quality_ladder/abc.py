"""Quality-ladder typing helpers and request classes."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..agent.abc import ActionDecision
from ..core.video.player import VideoChunkRequest


@dataclass(frozen=True)
class QualityLadderData:
    """Loaded quality ladder data.

    `video_size` and `video_quality` are shaped as
    [bitrate_levels, total_chunks]. `video_length` is shaped as
    [total_chunks] and stores each chunk's playback duration in milliseconds.
    """

    video_size: NDArray[np.int64]
    video_quality: NDArray[np.float64]
    video_length: NDArray[np.float64]


class QualityLadderLoader(Protocol):
    """Callable protocol for quality ladder data loaders."""

    def __call__(self, *args: Any, **kwargs: Any) -> QualityLadderData:
        """Load video size and quality matrices."""
        ...


@dataclass(frozen=True)
class QualityLadderRequest(VideoChunkRequest):
    """Request for a video chunk at a quality ladder level."""

    level: int


@dataclass(frozen=True)
class QualityLadderActionDecision(ActionDecision):
    """Agent decision containing a quality-ladder chunk request action."""

    action: QualityLadderRequest

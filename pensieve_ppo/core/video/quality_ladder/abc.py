"""Quality ladder loader typing helpers."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QualityLadderData:
    """Loaded quality ladder data with matching size and quality matrices."""

    video_size: NDArray[np.int64]
    video_quality: NDArray[np.float64]


class QualityLadderLoader(Protocol):
    """Callable protocol for quality ladder data loaders."""

    def __call__(self, *args: Any, **kwargs: Any) -> QualityLadderData:
        """Load video size and quality matrices."""
        ...

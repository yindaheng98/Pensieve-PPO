"""TraceSimulator wrapper that adds noise to delay."""

from typing import Optional

import numpy as np

from ..simulator import TraceSimulator
from ..wrapper import TraceSimulatorWrapper


# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L15
NOISE_LOW = 0.9
NOISE_HIGH = 1.1


class NoiseTraceSimulator(TraceSimulatorWrapper):
    """Wrapper that adds multiplicative noise to download delay.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L92

    This wrapper overrides `download_chunk` to apply multiplicative noise
    to the download delay, simulating network variability.
    """

    def __init__(
        self,
        simulator: TraceSimulator,
        noise_low: float = NOISE_LOW,
        noise_high: float = NOISE_HIGH,
        random_seed: Optional[int] = None,
    ):
        """Initialize the noise wrapper.

        Args:
            simulator: The TraceSimulator to wrap
            noise_low: Lower bound of noise multiplier (default: 0.9)
            noise_high: Upper bound of noise multiplier (default: 1.1)
            random_seed: Random seed for reproducibility
        """
        super().__init__(simulator)
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.seed(random_seed)

    def seed(self, seed: Optional[int]) -> None:
        """Set random seed for noise generation."""
        self.rng = np.random.RandomState(seed)

    def download_chunk(self, video_chunk_size: int) -> float:
        """Simulate downloading with multiplicative noise on delay.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L92

        Args:
            video_chunk_size: Size of chunk to download in bytes

        Returns:
            Download delay in milliseconds (with noise applied)
        """
        # Get base delay from wrapped simulator
        delay = super().download_chunk(video_chunk_size)

        # Add multiplicative noise to delay
        # https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L92
        delay *= self.rng.uniform(self.noise_low, self.noise_high)

        return delay

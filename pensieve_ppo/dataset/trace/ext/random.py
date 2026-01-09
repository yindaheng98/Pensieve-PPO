"""TraceSimulator wrapper that adds random trace selection."""

from typing import Optional

import numpy as np

from ..simulator import TraceSimulator
from ..wrapper import TraceSimulatorWrapper


class RandomTraceSimulator(TraceSimulatorWrapper):
    """Wrapper that adds random trace selection and random start point.

    https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L32

    This wrapper overrides `reset` and `on_video_finished` to:
    - Randomly select traces instead of sequential iteration
    - Randomly select starting point within each trace
    """

    def __init__(
        self,
        simulator: TraceSimulator,
        random_seed: Optional[int] = None,
    ):
        """Initialize the random trace wrapper.

        Args:
            simulator: The TraceSimulator to wrap
            random_seed: Random seed for reproducibility
        """
        super().__init__(simulator)
        self.seed(random_seed)

    def seed(self, seed: Optional[int]) -> None:
        """Set random seed for trace selection."""
        self.rng = np.random.RandomState(seed)

    def reset(self) -> None:
        """Reset with random trace selection and random start point.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L32-L40

        Args:
            trace_idx: If provided, use this trace. Otherwise, select randomly.
        """
        super().reset()

        # pick a random trace file
        self.trace_idx = self.rng.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = self.rng.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def on_video_finished(self) -> None:
        """Handle video end with random trace selection.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L145-L153
        """
        super().on_video_finished()

        # pick a random trace file
        self.trace_idx = self.rng.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the video
        # note: trace file starts with time 0
        self.mahimahi_ptr = self.rng.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

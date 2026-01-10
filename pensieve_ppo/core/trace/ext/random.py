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
        # Apply random initialization to the base simulator
        self._apply_random_init()

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for trace selection.

        Args:
            seed: Random seed. If None, uses global np.random module.
        """
        if seed is None:
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(seed)

    def _apply_random_init(self) -> None:
        """Apply random trace selection to the base simulator.

        This is called during __init__ and reset() to randomize
        the trace index and starting pointer.
        """
        rng = self.rng
        base = self.unwrapped

        # pick a random trace file
        base.trace_idx = rng.randint(len(base.all_cooked_time))
        base.cooked_time = base.all_cooked_time[base.trace_idx]
        base.cooked_bw = base.all_cooked_bw[base.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        base.mahimahi_ptr = rng.randint(1, len(base.cooked_bw))
        base.last_mahimahi_time = base.cooked_time[base.mahimahi_ptr - 1]

    def reset(self) -> None:
        """Reset with random trace selection and random start point.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L32-L40
        """
        super().reset()
        self._apply_random_init()

    def on_video_finished(self) -> None:
        """Handle video end with random trace selection.

        https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/core.py#L145-L153
        """
        super().on_video_finished()
        self._apply_random_init()

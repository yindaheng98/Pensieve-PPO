"""Default parameter combinations for ABR environment.

This module provides default parameter values that can be used when creating
ABREnv instances for training, testing, or evaluation.
"""

import numpy as np


# Default quality metric for each bitrate level (Pensieve's original values)
# This is a quality indicator for state/reward calculation (e.g., bitrate values in Kbps)
# https://github.com/godka/Pensieve-PPO/blob/a1b2579ca325625a23fe7d329a186ef09e32a3f1/src/env.py#L12
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps

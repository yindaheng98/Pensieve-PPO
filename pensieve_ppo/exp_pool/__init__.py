"""Experience pool module for collecting and storing trajectories.

This module provides utilities for managing experience data collected during
agent rollouts.
"""

from .abc import DictTrainingBatch
from .pool import ExperiencePool

__all__ = ['DictTrainingBatch', 'ExperiencePool']
